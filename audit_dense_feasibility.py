"""Audit whether dense eight-aircraft scenarios are operationally feasible.

This is a diagnostic tool, not the paper-scale rolling experiment.  It builds
one fixed set of eight-aircraft resets from the dense training configuration
and evaluates MAPPO and a deterministic FCFS+A* diagnostic baseline on exactly
those resets.  The report separates route reachability, action-mask blocking,
timeouts, deadlocks, and collisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fixed_scenarios(config: Config, episodes: int, seed: int) -> list[dict[str, Any]]:
    """Sample once from the configured reset distribution and serialize it."""
    sampler = MARLEnv(config)
    scenarios: list[dict[str, Any]] = []
    for episode in range(episodes):
        scenario_seed = seed + episode
        set_seed(scenario_seed)
        sampler.reset()
        graph = sampler.graph_env
        starts = [int(value) for value in graph.pos]
        goals = [int(value) for value in graph.goal]
        delays = [int(value) for value in graph.start_delays]
        reachable = [
            bool(agent.planned_path and (len(agent.planned_path) > 1 or starts[i] == goals[i]))
            for i, agent in enumerate(sampler.agents)
        ]
        same_start_release = Counter(zip(starts, delays))
        scenarios.append({
            "episode": episode,
            "seed": scenario_seed,
            "starts": starts,
            "goals": goals,
            "delays": delays,
            "reachable": reachable,
            "duplicate_start_release_pairs": int(sum(
                count - 1 for count in same_start_release.values() if count > 1
            )),
            "shadow_release_adjustments": int(
                getattr(sampler.graph_env, "shadow_release_adjustments", 0)
            ),
        })
    return scenarios


def active_agents(env: MARLEnv) -> list[int]:
    graph = env.graph_env
    return [
        i for i in range(env.num_agents)
        if not graph.done[i] and not graph.crashed[i] and graph.start_delays[i] <= 0
    ]


def apply_shared_safety_filter(env: MARLEnv, actions: list[int], order: list[int]) -> None:
    """Apply the same one-step node/edge reservation to both methods."""
    graph = env.graph_env
    reserved_nodes: set[int] = set()
    reserved_edges: set[tuple[int, int]] = set()
    for agent_id in order:
        if graph.on_edge[agent_id] or actions[agent_id] >= 5:
            if not graph.on_edge[agent_id]:
                reserved_nodes.add(graph.pos[agent_id])
            continue
        pos = graph.pos[agent_id]
        if actions[agent_id] >= len(graph.adj[pos]):
            actions[agent_id] = 7
            reserved_nodes.add(pos)
            continue
        next_node, _ = graph.adj[pos][actions[agent_id]]
        if next_node in reserved_nodes or (next_node, pos) in reserved_edges:
            actions[agent_id] = 7
            reserved_nodes.add(pos)
        else:
            reserved_nodes.add(next_node)
            reserved_edges.add((pos, next_node))


def fcfs_astar_actions(
    env: MARLEnv,
    masks: list[np.ndarray],
    release_rank: list[int],
) -> tuple[list[int], list[int]]:
    """Choose the feasible A* successor in FCFS order for the diagnostic run."""
    graph = env.graph_env
    actions = [7] * env.num_agents
    active = active_agents(env)
    order = sorted(
        active,
        key=lambda i: (-graph.priority[i], release_rank[i], i),
    )
    for agent_id in order:
        if graph.on_edge[agent_id]:
            continue
        pos = graph.pos[agent_id]
        goal = graph.goal[agent_id]
        candidates: list[tuple[float, int]] = []
        for action, (next_node, edge_index) in enumerate(graph.adj[pos][:5]):
            if masks[agent_id][action] <= 0:
                continue
            edge_cost = graph.edges[edge_index].length
            tail_cost = graph.get_shortest_path_distance(next_node, goal)
            if tail_cost >= 0:
                candidates.append((edge_cost + tail_cost, action))
        if candidates:
            actions[agent_id] = min(candidates)[1]
    return actions, order


def action_target(graph, agent_id: int, action: int) -> int | None:
    """Return the target node for a routing action, if one exists."""
    if action < 0 or action >= 5 or graph.on_edge[agent_id]:
        return None
    neighbors = graph.adj[graph.pos[agent_id]]
    if action >= len(neighbors):
        return None
    return int(neighbors[action][0])


def build_candidate_snapshot(
    env: MARLEnv,
    observation: np.ndarray,
    mask: np.ndarray,
    agent_id: int,
) -> str:
    """Serialize decision-time candidate features for later micro analysis."""
    graph = env.graph_env
    pos = graph.pos[agent_id]
    candidates = []
    for action, (node, _) in enumerate(graph.adj[pos][:5]):
        offset = action * 7
        candidates.append({
            "action": action,
            "node": int(node),
            "legal": int(mask[action] > 0),
            "path_cost": round(float(observation[offset + 1]), 4),
            "astar_reference": int(observation[offset + 2] > 0.5),
            "immediate_risk": int(observation[offset + 3] > 0.5),
            "corridor_locked": int(observation[offset + 4] > 0.5),
            "downstream_occupied": int(observation[offset + 5] > 0.5),
            "corridor_load": round(float(observation[offset + 6]), 4),
        })
    return json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))


def observe_decisions(
    env: MARLEnv,
    observations: list[np.ndarray],
    masks: list[np.ndarray],
    raw_actions: list[int],
    final_actions: list[int],
    active: list[int],
    step: int,
    counters: dict[str, int],
    wait_streaks: list[int],
    traces: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Classify every node decision before the environment advances."""
    graph = env.graph_env
    pending_trace_rows: list[dict[str, Any]] = []
    for agent_id in active:
        if graph.on_edge[agent_id]:
            wait_streaks[agent_id] = 0
            continue

        counters["decision_steps"] += 1
        legal_actions = [int(action) for action in np.flatnonzero(masks[agent_id][:5] > 0)]
        raw_action = int(raw_actions[agent_id])
        final_action = int(final_actions[agent_id])
        reference_node = env.agents[agent_id].get_next_planned_node(graph)
        reference_action = next(
            (
                action for action, (node, _) in enumerate(graph.adj[graph.pos[agent_id]][:5])
                if node == reference_node
            ),
            None,
        )

        if not legal_actions:
            wait_reason = "forced_wait_no_legal_route"
            counters["no_route_action_steps"] += 1
            if len(graph.adj[graph.pos[agent_id]]) == 1:
                counters["gate_blocked_steps"] += 1
            else:
                for next_node, _ in graph.adj[graph.pos[agent_id]]:
                    corridor = graph.corridor_mgr.get_corridor(graph.pos[agent_id], next_node)
                    if corridor is None:
                        continue
                    direction = corridor.edge_direction(graph.pos[agent_id], next_node)
                    if not graph.corridor_mgr.can_enter(corridor, direction, agent_id=agent_id):
                        counters["corridor_blocked_steps"] += 1
                        break
        elif raw_action == 7:
            wait_reason = "policy_wait_with_legal_route"
        elif raw_action in legal_actions and final_action == 7:
            wait_reason = "safety_filtered_wait"
        elif final_action in legal_actions:
            wait_reason = "move"
        else:
            wait_reason = "other_nonmove"

        if wait_reason == "forced_wait_no_legal_route":
            counters["forced_wait_steps"] += 1
        elif wait_reason == "policy_wait_with_legal_route":
            counters["policy_wait_steps"] += 1
        elif wait_reason == "safety_filtered_wait":
            counters["safety_filtered_wait_steps"] += 1
        elif wait_reason == "move":
            counters["route_move_steps"] += 1
            if reference_action is not None and final_action == reference_action:
                counters["reference_move_steps"] += 1
            else:
                counters["alternative_move_steps"] += 1

        if wait_reason != "move":
            wait_streaks[agent_id] += 1
        else:
            wait_streaks[agent_id] = 0
        counters["max_wait_streak"] = max(
            counters["max_wait_streak"], wait_streaks[agent_id]
        )

        if traces is not None:
            row = {
                "episode": None,
                "step": step,
                "agent_id": agent_id,
                "active_count": len(active),
                "position_before": int(graph.pos[agent_id]),
                "goal": int(graph.goal[agent_id]),
                "speed_before": round(float(graph.speed[agent_id]), 4),
                "progress_before": round(float(graph.progress[agent_id]), 4),
                "priority": round(float(graph.priority[agent_id]), 4),
                "remaining_distance": round(float(
                    graph.get_shortest_path_distance(graph.pos[agent_id], graph.goal[agent_id])
                ), 4),
                "reference_next_node": reference_node,
                "reference_action": reference_action,
                "legal_route_actions": "|".join(map(str, legal_actions)),
                "legal_route_count": len(legal_actions),
                "raw_action": raw_action,
                "raw_target_node": action_target(graph, agent_id, raw_action),
                "final_action": final_action,
                "final_target_node": action_target(graph, agent_id, final_action),
                "wait_reason": wait_reason,
                "wait_streak": wait_streaks[agent_id],
                "candidate_features": build_candidate_snapshot(
                    env, observations[agent_id], masks[agent_id], agent_id
                ),
            }
            pending_trace_rows.append(row)
    return pending_trace_rows


def finalize_trace_rows(
    env: MARLEnv,
    rows: list[dict[str, Any]],
    episode: int,
) -> None:
    """Attach post-action movement and terminal outcomes to trace rows."""
    graph = env.graph_env
    for row in rows:
        agent_id = int(row["agent_id"])
        row["episode"] = episode
        row["position_after"] = int(graph.pos[agent_id])
        row["progress_after"] = round(float(graph.progress[agent_id]), 4)
        row["crashed_after"] = int(graph.crashed[agent_id])
        row["completed_after"] = int(
            graph.done[agent_id]
            and not graph.crashed[agent_id]
            and graph.pos[agent_id] == graph.goal[agent_id]
        )


def observe_mask_constraints(env: MARLEnv, masks: list[np.ndarray], counters: dict[str, int]) -> None:
    graph = env.graph_env
    for agent_id in active_agents(env):
        if graph.on_edge[agent_id]:
            continue
        counters["decision_steps"] += 1
        if np.any(masks[agent_id][:5] > 0):
            continue
        counters["no_route_action_steps"] += 1
        pos = graph.pos[agent_id]
        if len(graph.adj[pos]) == 1:
            counters["gate_blocked_steps"] += 1
        else:
            for next_node, _ in graph.adj[pos]:
                corridor = graph.corridor_mgr.get_corridor(pos, next_node)
                if corridor is None:
                    continue
                direction = corridor.edge_direction(pos, next_node)
                if not graph.corridor_mgr.can_enter(corridor, direction, agent_id=agent_id):
                    counters["corridor_blocked_steps"] += 1
                    break


def run_episode(
    env: MARLEnv,
    scenario: dict[str, Any],
    method: str,
    trainer: MAPPOTrainerImproved | None,
    shared_safety_filter: bool,
    capture_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(int(scenario["seed"]))
    observations, masks = env.reset(
        starts=scenario["starts"], goals=scenario["goals"], delays=scenario["delays"]
    )
    global_state = env.get_global_state()
    release_rank_by_agent = [0] * env.num_agents
    for rank, agent_id in enumerate(sorted(range(env.num_agents), key=lambda i: (scenario["delays"][i], i))):
        release_rank_by_agent[agent_id] = rank

    counters: dict[str, int] = defaultdict(int)
    max_active = 0
    total_active = 0
    steps = 0
    wait_streaks = [0] * env.num_agents
    trace_rows: list[dict[str, Any]] = []
    for step in range(1, env.config.env.max_steps + 1):
        currently_active = active_agents(env)
        max_active = max(max_active, len(currently_active))
        total_active += len(currently_active)

        if method == "mappo":
            assert trainer is not None
            raw_actions, _, _ = trainer.select_actions(
                observations, masks, global_state, deterministic=True
            )
            order = sorted(currently_active, key=lambda i: (-env.graph_env.priority[i], release_rank_by_agent[i], i))
        else:
            raw_actions, order = fcfs_astar_actions(env, masks, release_rank_by_agent)
        actions = list(raw_actions)
        if shared_safety_filter:
            apply_shared_safety_filter(env, actions, order)

        pending_trace_rows = observe_decisions(
            env,
            observations,
            masks,
            raw_actions,
            actions,
            currently_active,
            step,
            counters,
            wait_streaks,
            trace_rows if capture_trace else None,
        )

        observations, _, dones, masks, _ = env.step(actions)
        global_state = env.get_global_state()
        if capture_trace:
            finalize_trace_rows(env, pending_trace_rows, int(scenario["episode"]))
            trace_rows.extend(pending_trace_rows)
        steps = step
        if all(dones):
            break

    graph = env.graph_env
    safe_completed = sum(
        graph.done[i] and not graph.crashed[i] and graph.pos[i] == graph.goal[i]
        for i in range(env.num_agents)
    )
    crashed_agents = int(sum(graph.crashed))
    timed_out = steps >= env.config.env.max_steps and not all(graph.done)
    record = {
        "episode": scenario["episode"],
        "seed": scenario["seed"],
        "method": method,
        "all_routes_reachable": int(all(scenario["reachable"])),
        "unreachable_agents": int(env.num_agents - sum(scenario["reachable"])),
        "duplicate_start_release_pairs": scenario["duplicate_start_release_pairs"],
        "safe_completed": int(safe_completed),
        "safe_completion_rate": safe_completed / env.num_agents,
        "full_success": int(safe_completed == env.num_agents and crashed_agents == 0),
        "crashed_agents": crashed_agents,
        "collision_events": int(graph.conflicts),
        "deadlock_events": len(env._deadlock_events),
        "timed_out": int(timed_out),
        "steps": steps,
        "max_active": max_active,
        "mean_active": total_active / max(steps, 1),
        "decision_steps": counters["decision_steps"],
        "no_route_action_steps": counters["no_route_action_steps"],
        "gate_blocked_steps": counters["gate_blocked_steps"],
        "corridor_blocked_steps": counters["corridor_blocked_steps"],
        "forced_wait_steps": counters["forced_wait_steps"],
        "policy_wait_steps": counters["policy_wait_steps"],
        "safety_filtered_wait_steps": counters["safety_filtered_wait_steps"],
        "route_move_steps": counters["route_move_steps"],
        "reference_move_steps": counters["reference_move_steps"],
        "alternative_move_steps": counters["alternative_move_steps"],
        "max_wait_streak": counters["max_wait_streak"],
    }
    return record, trace_rows


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)
    result: dict[str, Any] = {}
    for method, method_records in by_method.items():
        count = len(method_records)
        numeric_fields = [
            "safe_completion_rate", "full_success", "crashed_agents", "collision_events",
            "deadlock_events", "timed_out", "steps", "max_active", "mean_active",
            "no_route_action_steps", "gate_blocked_steps", "corridor_blocked_steps",
            "forced_wait_steps", "policy_wait_steps", "safety_filtered_wait_steps",
            "route_move_steps", "reference_move_steps", "alternative_move_steps",
            "max_wait_streak",
        ]
        metrics = {
            field: round(float(np.mean([record[field] for record in method_records])), 4)
            for field in numeric_fields
        }
        decision_steps = sum(record["decision_steps"] for record in method_records)
        blocked_steps = sum(record["no_route_action_steps"] for record in method_records)
        forced_waits = sum(record["forced_wait_steps"] for record in method_records)
        policy_waits = sum(record["policy_wait_steps"] for record in method_records)
        filtered_waits = sum(record["safety_filtered_wait_steps"] for record in method_records)
        route_moves = sum(record["route_move_steps"] for record in method_records)
        reference_moves = sum(record["reference_move_steps"] for record in method_records)
        result[method] = {
            "episodes": count,
            "safe_completion_pct": round(100 * metrics["safe_completion_rate"], 2),
            "full_success_pct": round(100 * metrics["full_success"], 2),
            "timeout_pct": round(100 * metrics["timed_out"], 2),
            "mean_crashed_agents": metrics["crashed_agents"],
            "mean_collision_events": metrics["collision_events"],
            "mean_deadlock_events": metrics["deadlock_events"],
            "mean_steps": metrics["steps"],
            "mean_max_active": metrics["max_active"],
            "mean_active": metrics["mean_active"],
            "no_route_action_pct": round(100 * blocked_steps / max(decision_steps, 1), 2),
            "mean_gate_blocked_steps": metrics["gate_blocked_steps"],
            "mean_corridor_blocked_steps": metrics["corridor_blocked_steps"],
            "forced_wait_pct": round(100 * forced_waits / max(decision_steps, 1), 2),
            "policy_wait_pct": round(100 * policy_waits / max(decision_steps, 1), 2),
            "safety_filtered_wait_pct": round(100 * filtered_waits / max(decision_steps, 1), 2),
            "route_move_pct": round(100 * route_moves / max(decision_steps, 1), 2),
            "reference_move_share_pct": round(100 * reference_moves / max(route_moves, 1), 2),
            "mean_max_wait_streak": metrics["max_wait_streak"],
        }
    return result


def load_trainer(env: MARLEnv, config: Config, model_path: str, device: str) -> MAPPOTrainerImproved:
    trainer = MAPPOTrainerImproved(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=config.training.hidden_dim,
        device=device,
        critic_state_schema=env.critic_state_schema,
    )
    trainer.load(model_path, load_optimizer=False)
    return trainer


def select_trace_episodes(
    records: list[dict[str, Any]], method: str, limit: int
) -> list[dict[str, Any]]:
    """Select reproducible failure cases for detailed step-level replay."""
    candidates = [record for record in records if record["method"] == method]
    candidates.sort(
        key=lambda record: (
            -record["timed_out"],
            record["safe_completed"],
            -record["policy_wait_steps"],
            -record["forced_wait_steps"],
            -record["crashed_agents"],
            record["episode"],
        )
    )
    return candidates[:max(limit, 0)]


def summarize_micro_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate action-level traces into a compact diagnostic report."""
    if not rows:
        return {}
    wait_reasons = Counter(row["wait_reason"] for row in rows)
    forced_positions = Counter(
        int(row["position_before"])
        for row in rows
        if row["wait_reason"] == "forced_wait_no_legal_route"
    )
    forced_agents = Counter(
        (int(row["episode"]), int(row["agent_id"]))
        for row in rows
        if row["wait_reason"] == "forced_wait_no_legal_route"
    )
    reference_moves = sum(
        row["wait_reason"] == "move"
        and row["reference_action"] is not None
        and row["final_action"] == row["reference_action"]
        for row in rows
    )
    all_moves = sum(row["wait_reason"] == "move" for row in rows)
    return {
        "trace_decisions": len(rows),
        "wait_reason_counts": dict(wait_reasons),
        "reference_move_share_pct": round(
            100 * reference_moves / max(all_moves, 1), 2
        ),
        "top_forced_wait_positions": [
            {"node": node, "count": count}
            for node, count in forced_positions.most_common(10)
        ],
        "top_forced_wait_agents": [
            {"episode": episode, "agent_id": agent_id, "count": count}
            for (episode, agent_id), count in forced_agents.most_common(10)
        ],
        "max_observed_wait_streak": max(int(row["wait_streak"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dense-scenario feasibility")
    parser.add_argument("--config", default="configs/fine_tune_dense_v1.yaml")
    parser.add_argument("--model", default="checkpoints/stage3_dense_v1/best_eval.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--methods", nargs="+", choices=["mappo", "fcfs"], default=["mappo", "fcfs"])
    parser.add_argument("--output-dir", default="results/diagnostics/dense_v1_feasibility_audit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--trace-method",
        choices=["mappo", "fcfs"],
        default="mappo",
        help="Method for which representative failure traces are replayed",
    )
    parser.add_argument(
        "--trace-episodes",
        type=int,
        default=5,
        help="Number of representative episodes to replay with action traces; zero disables tracing",
    )
    parser.add_argument(
        "--gate-shadow-mode",
        choices=["extended", "direct", "off"],
        default=None,
        help="Override the gate-release compatibility rule for a diagnostic ablation",
    )
    parser.add_argument(
        "--shared-safety-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same conservative one-step reservation filter to both methods",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    config = Config.from_yaml(args.config)
    if args.gate_shadow_mode is not None:
        config.env.gate_shadow_mode = args.gate_shadow_mode
    if config.env.num_agents != 8:
        raise ValueError("This audit is designed for the fixed eight-agent environment")
    scenarios = build_fixed_scenarios(config, args.episodes, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenarios.json").write_text(
        json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    for method in args.methods:
        env = MARLEnv(config)
        trainer = load_trainer(env, config, args.model, args.device) if method == "mappo" else None
        for index, scenario in enumerate(scenarios, start=1):
            record, _ = run_episode(
                env, scenario, method, trainer, args.shared_safety_filter
            )
            records.append(record)
            if index % 25 == 0 or index == len(scenarios):
                print(f"{method}: {index}/{len(scenarios)} episodes complete")

    with (output_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    trace_selection = []
    trace_rows: list[dict[str, Any]] = []
    if args.trace_episodes > 0 and args.trace_method in args.methods:
        selected = select_trace_episodes(records, args.trace_method, args.trace_episodes)
        scenario_by_episode = {int(scenario["episode"]): scenario for scenario in scenarios}
        trace_env = MARLEnv(config)
        trace_trainer = (
            load_trainer(trace_env, config, args.model, args.device)
            if args.trace_method == "mappo" else None
        )
        for item in selected:
            scenario = scenario_by_episode[int(item["episode"])]
            _, episode_trace = run_episode(
                trace_env,
                scenario,
                args.trace_method,
                trace_trainer,
                args.shared_safety_filter,
                capture_trace=True,
            )
            trace_rows.extend(episode_trace)
        trace_selection = selected
        (output_dir / f"{args.trace_method}_trace_selection.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if trace_rows:
            with (output_dir / f"{args.trace_method}_micro_trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
                writer.writeheader()
                writer.writerows(trace_rows)
            (output_dir / f"{args.trace_method}_micro_trace_summary.json").write_text(
                json.dumps(summarize_micro_trace(trace_rows), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    report = {
        "config": args.config,
        "model": args.model,
        "episodes": args.episodes,
        "seed": args.seed,
        "methods": args.methods,
        "shared_safety_filter": args.shared_safety_filter,
        "gate_shadow_mode": config.env.gate_shadow_mode,
        "micro_trace": {
            "method": args.trace_method,
            "selected_episodes": [item["episode"] for item in trace_selection],
            "trace_rows": len(trace_rows),
        },
        "scenario_feasibility": {
            "all_routes_reachable_pct": round(100 * np.mean([all(item["reachable"]) for item in scenarios]), 2),
            "mean_unreachable_agents": round(float(np.mean([
                len(item["reachable"]) - sum(item["reachable"]) for item in scenarios
            ])), 4),
            "mean_duplicate_start_release_pairs": round(float(np.mean([
                item["duplicate_start_release_pairs"] for item in scenarios
            ])), 4),
            "mean_shadow_release_adjustments": round(float(np.mean([
                item["shadow_release_adjustments"] for item in scenarios
            ])), 4),
        },
        "results": summarize(records),
        "interpretation": (
            "If FCFS+A* also has low full success and high timeout on these fixed scenarios, "
            "the release/rule combination is a feasibility bottleneck. If FCFS+A* is materially "
            "better, the primary limitation is MAPPO coordination under the dense distribution."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
