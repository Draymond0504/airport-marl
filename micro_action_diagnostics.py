"""Action-level diagnostics for masked airport-taxi policies.

The trainer logs episode returns, but a return alone cannot distinguish a
genuine traffic constraint from a policy that repeatedly selects WAIT while a
clear route is available.  This module records that distinction without
changing the environment transition rules.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class ActionDiagnosticCollector:
    """Collect compact episode statistics and selected action traces."""

    def __init__(self, trace_episodes: int = 2) -> None:
        self.trace_episodes = max(int(trace_episodes), 0)
        self.episode_summaries: list[dict[str, Any]] = []
        self._episode_rows: list[dict[str, Any]] = []
        self._trace_candidates: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._counts: Counter[str] = Counter()
        self._wait_streaks: list[int] = []
        self._max_wait_streak = 0
        self._wait_prob_sum = 0.0
        self._move_prob_sum = 0.0
        self._clear_move_prob_sum = 0.0
        self._episode_rows = []

    def begin_episode(self, num_agents: int) -> None:
        self._reset_episode_state()
        self._wait_streaks = [0] * num_agents

    @staticmethod
    def _route_action(graph: Any, agent_id: int, node: int | None) -> int | None:
        if node is None:
            return None
        for action, (next_node, _) in enumerate(graph.adj[graph.pos[agent_id]][:5]):
            if next_node == node:
                return action
        return None

    @staticmethod
    def _candidate_snapshot(
        graph: Any,
        observation: np.ndarray,
        mask: np.ndarray,
        agent_id: int,
    ) -> str:
        """Persist the candidate-level input that led to a route decision."""
        candidates: list[dict[str, Any]] = []
        pos = graph.pos[agent_id]
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

    def observe(
        self,
        env: Any,
        observations: list[np.ndarray],
        masks: list[np.ndarray],
        actions: list[int],
        probabilities: np.ndarray,
        step: int,
    ) -> list[dict[str, Any]]:
        """Classify node decisions immediately before ``env.step``."""
        graph = env.graph_env
        pending: list[dict[str, Any]] = []
        active_count = sum(
            not graph.done[i] and not graph.crashed[i] and graph.start_delays[i] == 0
            for i in range(env.num_agents)
        )

        for agent_id in range(env.num_agents):
            if graph.done[agent_id] or graph.crashed[agent_id] or graph.start_delays[agent_id] > 0:
                continue
            if graph.on_edge[agent_id]:
                self._counts["edge_transit_steps"] += 1
                # Under automatic speed control, an aircraft on an edge has
                # no route action. Retain its progress so a later no-progress
                # event is attributed to physical downstream blocking rather
                # than to a policy WAIT decision.
                pending.append({
                    "_edge_progress_event": True,
                    "agent_id": agent_id,
                    "progress_before": float(graph.progress[agent_id]),
                })
                continue

            observation = observations[agent_id]
            mask = masks[agent_id]
            action = int(actions[agent_id])
            legal = [int(a) for a in np.flatnonzero(mask[:5] > 0)]
            # A clear route has no immediate occupancy/head-on warning.  It is
            # intentionally stricter than merely being action-mask legal.
            clear = [
                action_id for action_id in legal
                if observation[action_id * 7 + 3] < 0.5
                and observation[action_id * 7 + 4] < 0.5
            ]
            reference_node = env.agents[agent_id].get_next_planned_node(graph)
            reference_action = self._route_action(graph, agent_id, reference_node)
            is_reference_clear = reference_action in clear

            self._counts["node_decisions"] += 1
            self._wait_prob_sum += float(probabilities[agent_id, 7])
            self._move_prob_sum += float(np.sum(probabilities[agent_id, legal]))
            self._clear_move_prob_sum += float(np.sum(probabilities[agent_id, clear]))

            if not legal:
                category = "forced_wait_no_legal_route"
            elif action == 7 and clear:
                category = "avoidable_policy_wait"
            elif action == 7:
                category = "policy_wait_with_legal_route"
            elif action in legal:
                category = "route_choice"
            else:
                category = "other_nonroute_action"
            self._counts[category] += 1

            if category == "route_choice":
                self._wait_streaks[agent_id] = 0
                self._counts["reference_route_choice"] += int(action == reference_action)
            else:
                self._wait_streaks[agent_id] += 1
                self._max_wait_streak = max(
                    self._max_wait_streak, self._wait_streaks[agent_id]
                )
                if action == 7 and is_reference_clear:
                    self._counts["wait_despite_clear_reference"] += 1

            if self.trace_episodes <= 0:
                continue
            target = None
            if 0 <= action < 5 and action < len(graph.adj[graph.pos[agent_id]]):
                target = int(graph.adj[graph.pos[agent_id]][action][0])
            pending.append({
                "episode": None,
                "step": step,
                "agent_id": agent_id,
                "active_count": active_count,
                "position_before": int(graph.pos[agent_id]),
                "goal": int(graph.goal[agent_id]),
                "remaining_distance_before": round(float(
                    graph.get_shortest_path_distance(graph.pos[agent_id], graph.goal[agent_id])
                ), 4),
                "reference_next_node": reference_node,
                "reference_action": reference_action,
                "legal_route_actions": "|".join(map(str, legal)),
                "clear_route_actions": "|".join(map(str, clear)),
                "action": action,
                "target_node": target,
                "decision_category": category,
                "action_probability": round(float(probabilities[agent_id, action]), 6),
                "wait_probability": round(float(probabilities[agent_id, 7]), 6),
                "legal_move_probability": round(float(np.sum(probabilities[agent_id, legal])), 6),
                "clear_move_probability": round(float(np.sum(probabilities[agent_id, clear])), 6),
                "wait_streak": self._wait_streaks[agent_id],
                "candidate_features": self._candidate_snapshot(
                    graph, observation, mask, agent_id
                ),
            })
        return pending

    def finalize_step(
        self,
        env: Any,
        pending_rows: list[dict[str, Any]],
        rewards: list[float],
        episode: int,
    ) -> None:
        graph = env.graph_env
        trace_rows: list[dict[str, Any]] = []
        for row in pending_rows:
            agent_id = int(row["agent_id"])
            if row.get("_edge_progress_event"):
                if float(graph.progress[agent_id]) <= float(row["progress_before"]) + 1e-6:
                    self._counts["edge_no_progress_steps"] += 1
                    if float(row["progress_before"]) >= 0.98:
                        self._counts["edge_destination_hold_steps"] += 1
                continue
            row["episode"] = episode
            row["position_after"] = int(graph.pos[agent_id])
            row["on_edge_after"] = int(graph.on_edge[agent_id])
            row["progress_after"] = round(float(graph.progress[agent_id]), 4)
            row["reward"] = round(float(rewards[agent_id]), 4)
            row["crashed_after"] = int(graph.crashed[agent_id])
            row["completed_after"] = int(
                graph.done[agent_id]
                and not graph.crashed[agent_id]
                and graph.pos[agent_id] == graph.goal[agent_id]
            )
            trace_rows.append(row)
        self._episode_rows.extend(trace_rows)

    def finish_episode(
        self,
        episode: int,
        steps: int,
        safe_completed: int,
        crashed_agents: int,
        timed_out: bool,
    ) -> dict[str, Any]:
        decisions = max(self._counts["node_decisions"], 1)
        summary = {
            "episode": episode,
            "steps": steps,
            "safe_completed": safe_completed,
            "crashed_agents": crashed_agents,
            "timed_out": int(timed_out),
            "node_decisions": self._counts["node_decisions"],
            "route_choice_steps": self._counts["route_choice"],
            "forced_wait_steps": self._counts["forced_wait_no_legal_route"],
            "policy_wait_steps": self._counts["policy_wait_with_legal_route"],
            "avoidable_wait_steps": self._counts["avoidable_policy_wait"],
            "wait_despite_clear_reference_steps": self._counts["wait_despite_clear_reference"],
            "reference_route_choice_steps": self._counts["reference_route_choice"],
            "edge_transit_steps": self._counts["edge_transit_steps"],
            "edge_no_progress_steps": self._counts["edge_no_progress_steps"],
            "edge_destination_hold_steps": self._counts["edge_destination_hold_steps"],
            "max_wait_streak": self._max_wait_streak,
            "avoidable_wait_pct": round(100.0 * self._counts["avoidable_policy_wait"] / decisions, 3),
            "route_choice_pct": round(100.0 * self._counts["route_choice"] / decisions, 3),
            "edge_no_progress_pct": round(
                100.0 * self._counts["edge_no_progress_steps"]
                / max(self._counts["edge_transit_steps"], 1),
                3,
            ),
            "mean_wait_probability": round(self._wait_prob_sum / decisions, 6),
            "mean_legal_move_probability": round(self._move_prob_sum / decisions, 6),
            "mean_clear_move_probability": round(self._clear_move_prob_sum / decisions, 6),
        }
        # Prioritize episodes that expose policy idling, then unresolved runs.
        score = (
            4.0 * summary["avoidable_wait_steps"]
            + 2.0 * summary["policy_wait_steps"]
            + summary["max_wait_streak"]
            + 100.0 * summary["timed_out"]
        )
        self.episode_summaries.append(summary)
        if self.trace_episodes > 0 and self._episode_rows:
            self._trace_candidates.append((score, summary, list(self._episode_rows)))
        return summary

    def export(self, output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "episode_action_summary.csv"
        if self.episode_summaries:
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.episode_summaries[0]))
                writer.writeheader()
                writer.writerows(self.episode_summaries)

        total_decisions = sum(row["node_decisions"] for row in self.episode_summaries)
        def total(name: str) -> int:
            return int(sum(int(row[name]) for row in self.episode_summaries))

        report = dict(metadata)
        report.update({
            "episodes": len(self.episode_summaries),
            "node_decisions": total_decisions,
            "avoidable_wait_pct": round(100.0 * total("avoidable_wait_steps") / max(total_decisions, 1), 3),
            "policy_wait_pct": round(100.0 * total("policy_wait_steps") / max(total_decisions, 1), 3),
            "forced_wait_pct": round(100.0 * total("forced_wait_steps") / max(total_decisions, 1), 3),
            "route_choice_pct": round(100.0 * total("route_choice_steps") / max(total_decisions, 1), 3),
            "edge_no_progress_pct": round(
                100.0 * total("edge_no_progress_steps")
                / max(total("edge_transit_steps"), 1),
                3,
            ),
            "wait_despite_clear_reference_pct": round(
                100.0 * total("wait_despite_clear_reference_steps") / max(total_decisions, 1), 3
            ),
        })

        selected = sorted(self._trace_candidates, key=lambda item: item[0], reverse=True)[:self.trace_episodes]
        selected_episodes = [item[1]["episode"] for item in selected]
        trace_rows = [row for _, _, rows in selected for row in rows]
        report["representative_episodes"] = selected_episodes
        report["trace_rows"] = len(trace_rows)
        if trace_rows:
            with (output_dir / "representative_action_trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
                writer.writeheader()
                writer.writerows(trace_rows)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report
