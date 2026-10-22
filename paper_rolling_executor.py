"""Fixed-slot executor for paper experiments.

The MAPPO policy is trained with a fixed number of aircraft.  This module
keeps that population fixed and feeds a long flight schedule through a small
set of reusable slots.  Flights that have not been admitted do not exist in
the physical environment, observations, or action masks.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved


PRIORITY_TARGETS = {120, 180, 240, 300}


@dataclass
class FlightTrace:
    flight_index: int
    priority: int = 1
    spawn_step: int | None = None
    finish_step: int | None = None
    status: str = "queued"
    path_nodes: list[int] = field(default_factory=list)
    moving_steps: int = 0

    collision_count: int = 0
    queue_wait_steps: int = 0
    admission_hold_steps: int = 0
    admission_hold_events: int = 0
    max_stall_steps: int = 0
    ever_high_priority: bool = False
    _last_pos: int | None = None
    _last_on_edge: bool = False
    _last_progress: float = 0.0


class RollingMetrics:
    def __init__(self, flights: list[dict], algorithm: str, time_scale: float, snapshot_interval: int) -> None:
        self.flights = flights
        self.algorithm = algorithm
        self.time_scale = time_scale
        self.snapshot_interval = snapshot_interval
        self.records = [FlightTrace(i) for i in range(len(flights))]
        self.snapshots: list[dict] = []
        self.events: list[dict] = []
        self.inference_ms: list[float] = []
        self.replan_ms: list[float] = []
        self.edge_traversals: list[dict] = []
        self._last_success = 0
        self._wall_start = time.perf_counter()

    def activate(self, flight_index: int, step: int, release_step: int, priority: int) -> None:
        rec = self.records[flight_index]
        rec.spawn_step = step
        rec.status = "active"
        rec.priority = priority
        rec.queue_wait_steps = max(0, step - release_step)

    def record_admission_hold(self, flight_index: int, step: int, reason: str) -> None:
        rec = self.records[flight_index]
        rec.admission_hold_steps += 1
        rec.admission_hold_events += 1
        self.events.append({
            "step": step,
            "type": "admission_hold_tick",
            "flight_index": flight_index,
            "reason": reason,
        })

    def record_stall(self, flight_index: int, stall_steps: int) -> None:
        rec = self.records[flight_index]
        rec.max_stall_steps = max(rec.max_stall_steps, int(stall_steps))

    def observe_slot(self, flight_index: int, env: MARLEnv) -> None:
        rec = self.records[flight_index]
        pos = env.graph_env.pos[slot := self._slot_for_flight(flight_index)]
        on_edge = env.graph_env.on_edge[slot]
        progress = env.graph_env.progress[slot] if on_edge else 0.0
        rec.path_nodes.append(pos)
        if rec._last_pos is not None and (
            pos != rec._last_pos or on_edge != rec._last_on_edge or abs(progress - rec._last_progress) > 1e-4
        ):
            rec.moving_steps += 1
        rec._last_pos, rec._last_on_edge, rec._last_progress = pos, on_edge, progress

    # Slot ownership is assigned by RollingPaperExecutor before each observation.
    def set_slot_map(self, slot_to_flight: list[int | None]) -> None:
        self._slot_map = {flight: slot for slot, flight in enumerate(slot_to_flight) if flight is not None}

    def _slot_for_flight(self, flight_index: int) -> int:
        return self._slot_map[flight_index]

    def finish(self, flight_index: int, step: int, status: str) -> None:
        rec = self.records[flight_index]
        if rec.finish_step is None:
            rec.finish_step = step
            rec.status = status

    def snapshot(self, step: int, active: int) -> None:
        if step % self.snapshot_interval:
            return
        successes = sum(r.status == "completed" for r in self.records)
        self.snapshots.append({
            "step": step,
            "active": active,
            "cumulative_success": successes,
            "throughput_per_step": (successes - self._last_success) / self.snapshot_interval,
            "time_min": step * self.time_scale / 60.0,
        })
        self._last_success = successes

    def record_edge_traversal(
        self,
        step: int,
        flight_index: int,
        slot: int,
        edge_start: int,
        edge_end: int,
    ) -> None:
        """Record one observed directed edge completion for event-window plots."""
        self.edge_traversals.append({
            "step": step,
            "flight_index": flight_index,
            "slot": slot,
            "edge_start": edge_start,
            "edge_end": edge_end,
        })

    def summary(self, final_step: int, edge_len_map: dict[tuple[int, int], float]) -> dict:
        queued_before_timeout = sum(r.status == "queued" for r in self.records)
        active_before_timeout = sum(r.status == "active" for r in self.records)
        max_queue_wait_steps = max((r.queue_wait_steps for r in self.records), default=0)
        total_admission_hold_events = sum(r.admission_hold_events for r in self.records)
        for rec in self.records:
            if rec.status in {"queued", "active"}:
                rec.status = "timeout"
                rec.finish_step = final_step

        completed = [r for r in self.records if r.status == "completed"]
        crashed = [r for r in self.records if r.status == "collision"]

        total = len(self.records)

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        taxi_steps = [r.finish_step - r.spawn_step for r in completed if r.spawn_step is not None and r.finish_step is not None]
        moving_steps = [r.moving_steps for r in completed]
        wait_steps = [max(t - m, 0) for t, m in zip(taxi_steps, moving_steps)]
        queue_steps = [r.queue_wait_steps for r in completed]
        dists = []
        for rec in completed:
            nodes = [rec.path_nodes[0]] if rec.path_nodes else []
            for node in rec.path_nodes[1:]:
                if node != nodes[-1]:
                    nodes.append(node)
            dists.append(sum(edge_len_map.get((a, b), 0.0) for a, b in zip(nodes, nodes[1:])))

        high = [r for r in completed if r.ever_high_priority]
        normal = [r for r in completed if not r.ever_high_priority]
        high_taxi = [r.finish_step - r.spawn_step for r in high if r.spawn_step is not None and r.finish_step is not None]
        normal_taxi = [r.finish_step - r.spawn_step for r in normal if r.spawn_step is not None and r.finish_step is not None]
        high_min = mean(high_taxi) * self.time_scale / 60.0 if high_taxi else None
        normal_min = mean(normal_taxi) * self.time_scale / 60.0 if normal_taxi else None
        speedup = None
        if high_min is not None and normal_min is not None and normal_min > 0:
            speedup = (normal_min - high_min) / normal_min * 100.0

        hours = max(final_step * self.time_scale / 3600.0, 1e-9)
        completion_rate = 100.0 * len(completed) / total if total else 0.0
        active_samples = [int(row["active"]) for row in self.snapshots]
        return {
            "algorithm": self.algorithm,
            "completion_rate_pct": round(completion_rate, 2),
            "safe_completion_rate_pct": round(completion_rate, 2),
            "final_service_rate_pct": round(completion_rate, 2),
            "success_count": len(completed),
            "eventual_completion_rate_pct": round(completion_rate, 2),
            "eventual_success_count": len(completed),
            "total_count": total,
            "collision_count": sum(r.collision_count for r in self.records),

            "timeout_count": sum(r.status == "timeout" for r in self.records),
            "never_spawned_count": queued_before_timeout,
            "active_timeout_count": active_before_timeout,
            "max_queue_wait_steps": max_queue_wait_steps,
            "total_admission_hold_events": total_admission_hold_events,
            "max_stall_steps": max((r.max_stall_steps for r in self.records), default=0),
            "mean_active_aircraft": round(mean(active_samples), 3),
            "max_active_aircraft": max(active_samples, default=0),
            "avg_moving_min": round(mean(moving_steps) * self.time_scale / 60.0, 2),
            "avg_onroute_wait_min": round(mean(wait_steps) * self.time_scale / 60.0, 2),
            "avg_taxi_min": round(mean(taxi_steps) * self.time_scale / 60.0, 2),
            "avg_taxi_sec": round(mean(taxi_steps) * self.time_scale, 2),
            "avg_delay_min": round(mean(queue_steps) * self.time_scale / 60.0, 2),
            "avg_total_time_min": round((mean(taxi_steps) + mean(queue_steps)) * self.time_scale / 60.0, 2),
            "avg_wait_min": round(mean(queue_steps) * self.time_scale / 60.0, 2),
            "avg_path_dist": round(mean(dists), 2),
            "throughput_per_hour": round(len(completed) / hours, 2),
            "avg_inference_ms": round(mean(self.inference_ms), 3) if self.inference_ms else "N/A",
            "max_inference_ms": round(max(self.inference_ms), 3) if self.inference_ms else "N/A",
            "avg_replan_ms": round(mean(self.replan_ms), 3) if self.replan_ms else "N/A",
            "max_replan_ms": round(max(self.replan_ms), 3) if self.replan_ms else "N/A",
            "replan_count": len(self.replan_ms),
            "high_pri_count": len(high),
            "normal_pri_count": len(normal),
            "high_pri_avg_taxi_min": round(high_min, 2) if high_min is not None else None,
            "normal_pri_avg_taxi_min": round(normal_min, 2) if normal_min is not None else None,
            "high_pri_speedup_pct": round(speedup, 1) if speedup is not None else None,
            "wall_time_sec": round(time.perf_counter() - self._wall_start, 1),
            "execution_protocol": "fixed_slot_rolling",
        }

    def export(self, result_dir: str, final_step: int, edge_len_map: dict[tuple[int, int], float]) -> dict:
        os.makedirs(result_dir, exist_ok=True)
        summary = self.summary(final_step, edge_len_map)
        prefix = self.algorithm.replace("+", "_").replace(" ", "_")
        with open(os.path.join(result_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(result_dir, f"{prefix}_events.json"), "w", encoding="utf-8") as f:
            json.dump({"algorithm": self.algorithm, "events": self.events}, f, ensure_ascii=False, indent=2)
        if self.snapshots:
            with open(os.path.join(result_dir, f"{prefix}_timeseries.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.snapshots[0].keys())
                writer.writeheader()
                writer.writerows(self.snapshots)
        with open(os.path.join(result_dir, f"{prefix}_per_agent.csv"), "w", newline="", encoding="utf-8") as f:
            fields = ["agent_id", "flight_index", "priority", "spawn_step", "finish_step", "status", "taxi_steps", "taxi_sec", "moving_steps", "moving_sec", "onroute_wait_sec", "delay_steps", "wait_time_min", "admission_hold_steps", "admission_hold_events", "max_stall_steps", "path_dist", "collision_count", "crashed", "path_nodes"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.records:
                taxi = (r.finish_step - r.spawn_step) if r.status == "completed" and r.spawn_step is not None and r.finish_step is not None else None
                nodes = [r.path_nodes[0]] if r.path_nodes else []
                for node in r.path_nodes[1:]:
                    if node != nodes[-1]:
                        nodes.append(node)
                distance = sum(edge_len_map.get((a, b), 0.0) for a, b in zip(nodes, nodes[1:]))
                writer.writerow({
                    "agent_id": r.flight_index, "flight_index": r.flight_index, "priority": r.priority,
                    "spawn_step": r.spawn_step, "finish_step": r.finish_step, "status": r.status,
                    "taxi_steps": taxi, "taxi_sec": taxi * self.time_scale if taxi is not None else None,
                    "moving_steps": r.moving_steps, "moving_sec": r.moving_steps * self.time_scale,
                    "onroute_wait_sec": (taxi - r.moving_steps) * self.time_scale if taxi is not None else None,
                    "delay_steps": r.queue_wait_steps, "wait_time_min": round(r.queue_wait_steps * self.time_scale / 60.0, 3),
                    "admission_hold_steps": r.admission_hold_steps, "admission_hold_events": r.admission_hold_events, "max_stall_steps": r.max_stall_steps,
                    "path_dist": round(distance, 3), "collision_count": r.collision_count,
                    "path_nodes": str(nodes),
                })
        edge_path = os.path.join(result_dir, f"{prefix}_edge_traversals.csv")
        with open(edge_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "flight_index", "slot", "edge_start", "edge_end"],
            )
            writer.writeheader()
            writer.writerows(self.edge_traversals)
        return summary


class RollingScenario:
    """Scenario semantics expressed in scheduled-flight IDs, not slot IDs."""
    def __init__(
        self,
        scenario: str,
        flights: list[dict],
        max_steps: int,
        time_scale: float,
        metrics: RollingMetrics,
        schedule_compression: float = 1.0,
    ) -> None:
        self.scenario = scenario
        self.flights = flights
        self.max_steps = max_steps
        self.time_scale = time_scale
        self.metrics = metrics
        self.schedule_compression = schedule_compression
        self.release_steps = [int(round(f["t0"] * schedule_compression)) for f in flights]
        self.closure_plan = [(int(max_steps * 0.10), 800, (102, 103)), (int(max_steps * 0.55), 800, (304, 305))]
        self.triggered_closures: set[int] = set()
        self.priority_until: dict[int, int] = {}
        if scenario == "delay":
            self._apply_weather_delay()

    def _apply_weather_delay(self) -> None:
        candidates = [
            i for i, f in enumerate(self.flights)
            if f.get("type") == "ARR" and 11 * 3600 <= f.get("time_sec", -1) <= 13 * 3600
        ]
        if not candidates:
            return
        base_time = self.flights[0]["time_sec"]
        updated_releases = []
        for rank, flight_index in enumerate(candidates):
            offset = 3600 * rank / max(len(candidates) - 1, 1)
            release = int(round((13.5 * 3600 + offset - base_time) / self.time_scale * self.schedule_compression))
            self.release_steps[flight_index] = max(self.release_steps[flight_index], release)
            updated_releases.append(self.release_steps[flight_index])
        self.metrics.events.append({
            "step": 1, "type": "delay_release_plan", "affected_flights": candidates,
            "release_window": "13:30-14:30", "count": len(candidates),
            "first_release_step": min(updated_releases),
            "last_release_step": max(updated_releases),
            "duration": max(updated_releases) - min(updated_releases),
        })

    def before_step(self, step: int, env: MARLEnv, slot_to_flight: list[int | None]) -> bool:
        changed = False
        if self.scenario == "closure":
            for plan_index, (trigger, duration, edge_nodes) in enumerate(self.closure_plan):
                if plan_index in self.triggered_closures or step < trigger:
                    continue
                self.triggered_closures.add(plan_index)
                closed = []
                for edge_index, edge in enumerate(env.graph_env.edges):
                    if (edge.start, edge.end) in (edge_nodes, edge_nodes[::-1]):
                        edge.closed = True
                        env.graph_env.closure_timers[edge_index] = duration
                        closed.append(edge_index)
                self.metrics.events.append({"step": step, "type": "closure", "edge_nodes": edge_nodes, "duration": duration, "closed_edge_indices": closed})
                changed = True

        if self.scenario == "priority":
            for slot, flight_index in enumerate(slot_to_flight):
                if slot in self.priority_until and step >= self.priority_until[slot]:
                    env.graph_env.priority[slot] = 1
                    self.metrics.events.append({"step": step, "type": "priority_restore", "flight_index": flight_index, "slot": slot})
                    del self.priority_until[slot]
                if flight_index is None or flight_index not in PRIORITY_TARGETS:
                    continue
                rec = self.metrics.records[flight_index]
                if not rec.ever_high_priority:
                    env.graph_env.priority[slot] = 1001
                    rec.priority = 1001
                    rec.ever_high_priority = True
                    self.priority_until[slot] = step + 300
                    self.metrics.events.append({"step": step, "type": "priority_boost", "flight_index": flight_index, "slot": slot, "duration": 300})
        return changed


class RollingPaperExecutor:
    def __init__(self, args, flights: list[dict], config: Config, edge_len_map: dict[tuple[int, int], float], algorithm: str) -> None:
        self.args, self.flights, self.config, self.edge_len_map, self.algorithm = args, flights, config, edge_len_map, algorithm
        self.slots = args.active_limit
        if self.slots != config.env.num_agents:
            print(f"  [Rolling executor] override env.num_agents={config.env.num_agents} -> {self.slots}")
        config.env.num_agents = self.slots
        config.env.max_steps = args.max_steps
        config.env.taxiway_close_prob = 0.0
        config.env.delay_prob = 0.0
        config.env.priority_change_prob = 0.0
        # Operational evaluation keeps apron corridor protection but disables
        # training-only deadlock recovery and deadlock-based slot retirement.
        config.env.enable_deadlock_recovery = False
        self.env = MARLEnv(config)
        seed_starts = [flights[0]["start"]] * self.slots
        seed_goals = [flights[0]["goal"]] * self.slots
        self.env.reset(starts=seed_starts, goals=seed_goals, delays=[0] * self.slots)
        for slot in range(self.slots):
            self.env.deactivate_slot(slot)
        self.slot_to_flight: list[int | None] = [None] * self.slots
        self.stall_count = [0] * self.slots
        self.fcfs_routes: dict[int, list[int]] = {}
        self.held_flights: set[int] = set()
        self.last_admission_by_start: dict[int, int] = {}
        self.metrics = RollingMetrics(flights, algorithm, args.time_scale, args.snapshot_interval)
        self.scenario = RollingScenario(
            args.scenario,
            flights,
            args.max_steps,
            args.time_scale,
            self.metrics,
            schedule_compression=args.schedule_compression,
        )
        self._priority_event_step: int | None = None
        self._priority_target_flight: int | None = None
        self._priority_window_before = 100
        self._priority_window_after = 200
        self._priority_frame_buffer: deque[dict] = deque(
            maxlen=self._priority_window_before
        )
        self._priority_frames: list[dict] = []
        self.trainer = None
        if algorithm == "MAPPO":
            self.trainer = MAPPOTrainerImproved(
                num_agents=self.slots, obs_dim=self.env.obs_dim, action_dim=self.env.action_dim,
                global_state_dim=self.env.global_state_dim, hidden_dim=config.training.hidden_dim,
            )
            self.trainer.load(args.model, load_critic=False, load_optimizer=False)

    def _admit_due_flights(self, step: int) -> None:
        empty = [slot for slot, flight in enumerate(self.slot_to_flight) if flight is None]
        due = sorted(
            (i for i, release in enumerate(self.scenario.release_steps)
             if release <= step and self.metrics.records[i].status == "queued"),
            key=lambda flight_index: (self.scenario.release_steps[flight_index], flight_index),
        )
        if not empty or not due:
            if due and not empty:
                for flight_index in due:
                    self.metrics.record_admission_hold(flight_index, step, "no_empty_slot")
            return

        # Admission is an ATC gate-holding decision, not a collision outcome.
        # A second flight cannot materialize at an occupied entry node merely
        # because the schedule contains the same timestamp and origin.
        blocked_starts: set[int] = set()
        # Must match TaxiGraphEnv._detect_global_collisions: a squared
        # Euclidean distance below 3 is already a physical collision.  The
        # previous progress<0.2 proxy was too weak on short entry edges.
        clearance_sq = 3.0
        for active_slot, active_flight in enumerate(self.slot_to_flight):
            if active_flight is None:
                continue
            pos = self.env.graph_env.pos[active_slot]
            active_xy = self.env.graph_env._get_agent_coords(active_slot)
            for candidate_start in {self.flights[i]["start"] for i in due}:
                start_xy = self.env.graph_env.nodes[candidate_start].position
                if ((active_xy[0] - start_xy[0]) ** 2 + (active_xy[1] - start_xy[1]) ** 2) < clearance_sq:
                    blocked_starts.add(candidate_start)

        empty_iter = iter(empty)
        for flight_index in due:
            start_node = self.flights[flight_index]["start"]
            last_admission = self.last_admission_by_start.get(start_node)
            headway_blocked = (
                last_admission is not None
                and step - last_admission < self.args.entry_headway_steps
            )
            if start_node in blocked_starts or headway_blocked:
                self.metrics.record_admission_hold(
                    flight_index, step, "entry_headway" if headway_blocked else "occupied_entry_node"
                )
                if flight_index not in self.held_flights:
                    self.held_flights.add(flight_index)
                    self.metrics.events.append({
                        "step": step, "type": "admission_hold", "flight_index": flight_index,
                        "start": start_node,
                        "reason": "entry_headway" if headway_blocked else "occupied_entry_node",
                    })
                continue
            try:
                slot = next(empty_iter)
            except StopIteration:
                break
            flight = self.flights[flight_index]
            priority = 1
            self.env.activate_slot(slot, flight["start"], flight["goal"], priority=priority)
            self.slot_to_flight[slot] = flight_index
            self.metrics.activate(flight_index, step, self.scenario.release_steps[flight_index], priority)
            self.metrics.events.append({"step": step, "type": "admit", "flight_index": flight_index, "slot": slot})
            self.held_flights.discard(flight_index)
            blocked_starts.add(start_node)
            self.last_admission_by_start[start_node] = step

    def _replan_active_routes(self) -> None:
        start = time.perf_counter()
        graph = self.env.graph_env.get_graph_for_planner()
        for slot, flight_index in enumerate(self.slot_to_flight):
            if flight_index is None:
                continue
            pos = self.env.graph_env.pos[slot]
            goal = self.env.graph_env.goal[slot]
            path = self.env.planner.plan_path(pos, goal, graph) or [pos]
            if self.algorithm == "MAPPO":
                agent = self.env.agents[slot]
                agent.planned_path = path
                agent.planned_path_set = set(path)
                agent.path_index = 0
            else:
                self.fcfs_routes[slot] = path
        self.metrics.replan_ms.append((time.perf_counter() - start) * 1000.0)

    def _mappo_actions(self, obs, masks, active_slots: list[int]) -> list[int]:
        actions = [7] * self.slots
        if not active_slots:
            return actions
        started = time.perf_counter()
        with torch.no_grad():
            batch_obs = torch.as_tensor(np.asarray([obs[i] for i in active_slots]), dtype=torch.float32, device=self.trainer.device)
            batch_masks = torch.as_tensor(np.asarray([masks[i] for i in active_slots]), dtype=torch.float32, device=self.trainer.device)
            probs = self.trainer.actor(batch_obs, batch_masks).probs
            selected = torch.argmax(probs, dim=-1).cpu().numpy()
        self.metrics.inference_ms.append((time.perf_counter() - started) * 1000.0)
        for slot, action in zip(active_slots, selected):
            actions[slot] = int(action)
        return actions

    def _fcfs_actions(self, active_slots: list[int]) -> list[int]:
        actions = [7] * self.slots
        graph = self.env.graph_env.get_graph_for_planner()
        for slot in self._coordination_order(active_slots):
            if self.env.graph_env.on_edge[slot]:
                continue
            route = self.fcfs_routes.get(slot)
            pos, goal = self.env.graph_env.pos[slot], self.env.graph_env.goal[slot]
            if not route or pos not in route:
                started = time.perf_counter()
                route = self.env.planner.plan_path(pos, goal, graph) or [pos]
                self.metrics.replan_ms.append((time.perf_counter() - started) * 1000.0)
                self.fcfs_routes[slot] = route
            try:
                idx = route.index(pos)
                nxt = route[idx + 1]
            except (ValueError, IndexError):
                continue
            for action, (neighbor, _) in enumerate(self.env.graph_env.adj[pos]):
                if neighbor == nxt:
                    actions[slot] = action
                    break
        return actions

    def _coordination_order(self, active_slots: list[int]) -> list[int]:
        """Return the deterministic right-of-way order for the shared ATC filter."""
        if self.algorithm != "FCFS":
            return sorted(active_slots, key=lambda slot: (-self.env.graph_env.priority[slot], slot))
        return sorted(
            active_slots,
            key=lambda slot: (
                -self.env.graph_env.priority[slot],
                self.metrics.records[self.slot_to_flight[slot]].spawn_step or 0,
                self.slot_to_flight[slot],
            ),
        )

    def _safety_filter(self, actions: list[int], active_slots: list[int]) -> None:
        """Common deterministic ATC conflict filter, applied to both methods."""
        reserved_nodes: set[int] = set()
        reserved_edges: set[tuple[int, int]] = set()
        graph = self.env.graph_env

        # Aircraft already committed to an edge own that directed movement.
        # Reserve their destination in final approach to prevent a node queue
        # from becoming a circular edge-blocking queue.
        for slot in active_slots:
            if not graph.on_edge[slot] or graph.edge_idx[slot] is None:
                continue
            edge = graph.edges[graph.edge_idx[slot]]
            reserved_edges.add((edge.start, edge.end))
            if graph.progress[slot] >= graph.approach_reservation_progress:
                reserved_nodes.add(edge.end)

        for slot in self._coordination_order(active_slots):
            if graph.on_edge[slot] or actions[slot] >= 5:
                if not graph.on_edge[slot]:
                    reserved_nodes.add(graph.pos[slot])
                continue
            pos = graph.pos[slot]
            if actions[slot] >= len(graph.adj[pos]):
                actions[slot] = 7
                reserved_nodes.add(pos)
                continue
            nxt, _ = graph.adj[pos][actions[slot]]
            if (
                nxt in reserved_nodes
                or (nxt, pos) in reserved_edges
                or graph.is_node_entry_blocked(nxt, slot)
            ):
                actions[slot] = 7
                reserved_nodes.add(pos)
            else:
                reserved_nodes.add(nxt)
                reserved_edges.add((pos, nxt))

    def _retire_slot(self, slot: int, step: int, status: str) -> None:
        flight_index = self.slot_to_flight[slot]
        if flight_index is None:
            return
        rec = self.metrics.records[flight_index]
        if status == "collision":
            rec.collision_count += 1

        self.metrics.finish(flight_index, step, status)
        self.metrics.events.append({"step": step, "type": status, "flight_index": flight_index, "slot": slot})
        self.env.deactivate_slot(slot)
        self.slot_to_flight[slot] = None
        self.stall_count[slot] = 0
        self.fcfs_routes.pop(slot, None)
        self.scenario.priority_until.pop(slot, None)

    def _capture_priority_frame(self, step: int, actions: list[int]) -> None:
        """Capture a compact event-relative state frame for priority plots."""
        if self.args.scenario != "priority":
            return
        agents = []
        for slot, flight_index in enumerate(self.slot_to_flight):
            if flight_index is None:
                continue
            graph = self.env.graph_env
            edge_start = edge_end = None
            if graph.on_edge[slot]:
                edge = graph.edges[graph.edge_idx[slot]]
                edge_start, edge_end = edge.start, edge.end
            agents.append({
                "flight_index": flight_index,
                "slot": slot,
                "pos": graph.pos[slot],
                "on_edge": int(graph.on_edge[slot]),
                "progress": round(float(graph.progress[slot]) if graph.on_edge[slot] else 0.0, 4),
                "edge_start": edge_start,
                "edge_end": edge_end,
                "speed": round(float(graph.speed[slot]), 4),
                "action": int(actions[slot]),
                "priority": int(graph.priority[slot]),
                "is_vip": int(flight_index == self._priority_target_flight),
            })
        frame = {"step": step, "agents": agents}
        if self._priority_event_step is None:
            self._priority_frame_buffer.append(frame)
            return
        if step <= self._priority_event_step + self._priority_window_after:
            self._priority_frames.append(frame)

    def _start_priority_capture(self, step: int, events: list[dict]) -> None:
        if self._priority_event_step is not None:
            return
        for event in events:
            if event.get("type") != "priority_boost":
                continue
            self._priority_event_step = step
            self._priority_target_flight = event.get("flight_index")
            self._priority_frames = list(self._priority_frame_buffer)
            return

    def _export_priority_micro_log(self) -> None:
        if self.args.scenario != "priority" or self._priority_event_step is None:
            return
        payload = {
            "scenario": "priority",
            "algorithm": self.algorithm,
            "event_step": self._priority_event_step,
            "target_flight_index": self._priority_target_flight,
            "window_before_steps": self._priority_window_before,
            "window_after_steps": self._priority_window_after,
            "time_scale": self.args.time_scale,
            "frames": self._priority_frames,
        }
        with open(
            os.path.join(self.args.result_dir, f"{self.algorithm}_micro_priority.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def run(self) -> dict:
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        print(f"\n  [{self.algorithm}] fixed-slot rolling evaluation: {len(self.flights)} flights, {self.slots} slots")
        final_step = 0
        for step in range(self.args.max_steps):
            final_step = step
            event_count_before = len(self.metrics.events)
            closure_changed = self.scenario.before_step(step, self.env, self.slot_to_flight)
            self._start_priority_capture(step, self.metrics.events[event_count_before:])
            self._admit_due_flights(step)
            if closure_changed:
                self._replan_active_routes()
            active_slots = [slot for slot, flight in enumerate(self.slot_to_flight) if flight is not None]
            self.metrics.set_slot_map(self.slot_to_flight)
            for flight_index in (f for f in self.slot_to_flight if f is not None):
                self.metrics.observe_slot(flight_index, self.env)
            self.metrics.snapshot(step, len(active_slots))
            if not active_slots:
                if all(r.status != "queued" for r in self.metrics.records):
                    break
                continue
            obs, masks = self.env._build_observations_and_masks()
            actions = self._mappo_actions(obs, masks, active_slots) if self.algorithm == "MAPPO" else self._fcfs_actions(active_slots)
            self._safety_filter(actions, active_slots)
            prior_states = {
                slot: (
                    self.env.graph_env.pos[slot],
                    self.env.graph_env.on_edge[slot],
                    self.env.graph_env.edge_idx[slot],
                    self.env.graph_env.progress[slot],
                )
                for slot in active_slots
            }
            self.env.step(actions)
            self._capture_priority_frame(step + 1, actions)
            for slot in list(active_slots):
                flight_index = self.slot_to_flight[slot]
                if flight_index is None:
                    continue
                prior_pos, prior_on_edge, prior_edge_index, prior_progress = prior_states[slot]
                prior_edge = None
                if prior_on_edge and prior_edge_index is not None:
                    edge = self.env.graph_env.edges[prior_edge_index]
                    prior_edge = (edge.start, edge.end)
                if (
                    prior_edge is not None
                    and not self.env.graph_env.on_edge[slot]
                    and self.env.graph_env.pos[slot] == prior_edge[1]
                ):
                    self.metrics.record_edge_traversal(
                        step + 1, flight_index, slot, prior_edge[0], prior_edge[1]
                    )
                if self.env.graph_env.crashed[slot]:
                    self._retire_slot(slot, step + 1, "collision")
                    continue
                if self.env.graph_env.done[slot]:
                    self._retire_slot(slot, step + 1, "completed")
                    continue
                graph = self.env.graph_env
                moved = graph.pos[slot] != prior_pos
                if prior_on_edge and graph.on_edge[slot]:
                    moved = moved or graph.edge_idx[slot] != prior_edge_index or graph.progress[slot] > prior_progress + 1e-5
                elif not prior_on_edge and graph.on_edge[slot]:
                    moved = moved or graph.progress[slot] > 1e-5
                self.stall_count[slot] = 0 if moved else self.stall_count[slot] + 1
                self.metrics.record_stall(flight_index, self.stall_count[slot])

            if step and step % 500 == 0:
                completed = sum(r.status == "completed" for r in self.metrics.records)
                print(f"  Step {step:<6} | active {len(active_slots):<2} | completed {completed}/{len(self.flights)}")
        summary = self.metrics.export(self.args.result_dir, final_step, self.edge_len_map)
        self._export_priority_micro_log()
        print(f"  [{self.algorithm}] safe completion: {summary['completion_rate_pct']}% ({summary['success_count']}/{summary['total_count']})")
        return summary


def run_mappo_rolling(args, flights, config, edge_len_map, result_dir):
    return RollingPaperExecutor(args, flights, config, edge_len_map, "MAPPO").run()


def run_fcfs_rolling(args, flights, config, edge_len_map, result_dir):
    return RollingPaperExecutor(args, flights, config, edge_len_map, "FCFS").run()
