"""Layered rolling evaluation with explicit proposal and execution stages.

The policy or FCFS planner first proposes an action. A shared execution layer
then applies deterministic protection and stall-triggered replanning. The
original RollingPaperExecutor remains unchanged for backward compatibility.
"""

from __future__ import annotations

import csv
import json
import os
import time
from copy import deepcopy

from paper_rolling_executor import RollingPaperExecutor


class LayeredRollingPaperExecutor(RollingPaperExecutor):
    """Rolling executor with common protection and recovery for both methods."""

    def __init__(self, args, flights, config, edge_len_map, algorithm):
        super().__init__(args, flights, config, edge_len_map, algorithm)
        self.stall_threshold_steps = max(int(getattr(args, "stall_threshold", 180)), 1)
        self.enable_stall_replanning = not bool(getattr(args, "disable_stall_replanning", False))
        self.enable_stall_path_recovery = not bool(
            getattr(args, "disable_stall_path_recovery", False)
        )
        self.stall_recovery_trigger_steps = max(
            int(getattr(args, "stall_recovery_trigger", 45)), 1
        )
        self.stall_recovery_cooldown_steps = max(
            int(getattr(args, "stall_recovery_cooldown", 15)), 1
        )
        # An active flight must eventually release its fixed rolling slot.
        # This is an operational service-window rule, not a collision or
        # stall-replanning event.
        self.operational_timeout_steps = max(
            int(getattr(args, "operational_timeout_steps", 1800)), 1
        )
        self.operational_timeout_events = 0
        self.allow_vacating_node_entry = not bool(
            getattr(args, "disable_vacating_node_entry", False)
        )
        self._last_stall_recovery_step = [-10_000] * self.slots
        self.raw_action_rows: list[dict] = []
        self.execution_events: list[dict] = []
        self.conflict_protection_events = 0
        self.stall_replanning_events = 0
        self.stall_path_recovery_events = 0
        self.stall_motion_recovery_events = 0
        self.clearance_release_recovery_events = 0
        self.blocker_release_recovery_events = 0
        self.recovery_kinds: dict[int, str] = {}
        self.safety_violation_count = 0
        self.last_filter_reasons: dict[int, str] = {}
        self.stall_recovery_diagnostics: list[dict] = []
        self.filter_reason_counts: dict[str, int] = {}
        self.collision_details: list[dict] = []
        self._collision_event_cursor = 0

    def _record_intervention(self, step, slot, raw_action, executed_action, reason):
        flight_index = self.slot_to_flight[slot]
        event = {
            "step": step,
            "slot": slot,
            "flight_index": flight_index,
            "type": "conflict_protection",
            "raw_action": int(raw_action),
            "executed_action": int(executed_action),
            "reason": reason,
        }
        self.execution_events.append(event)
        self.conflict_protection_events += 1
        self.filter_reason_counts[reason] = self.filter_reason_counts.get(reason, 0) + 1

    def _apply_safety_filter(self, actions: list[int], active_slots: list[int], step: int) -> list[int]:
        """Apply one common ATC filter and record every raw-action override."""
        raw_actions = list(actions)
        self.last_filter_reasons = {}
        reserved_edges: set[tuple[int, int]] = set()
        graph = self.env.graph_env
        occupied_nodes: set[int] = {
            graph.pos[slot] for slot in active_slots if not graph.on_edge[slot]
        }
        reserved_nodes: set[int] = set()
        vacating_nodes: set[int] = set()

        for slot in active_slots:
            if not graph.on_edge[slot] or graph.edge_idx[slot] is None:
                continue
            edge = graph.edges[graph.edge_idx[slot]]
            reserved_edges.add((edge.start, edge.end))
            # Reserve the downstream node as soon as an aircraft enters an
            # edge.  Delaying this reservation until final approach allows
            # two incident edges to converge inside the clearance zone.
            reserved_nodes.add(edge.end)

        for slot in self._coordination_order(active_slots):
            raw_action = raw_actions[slot]
            reason = None
            if graph.on_edge[slot] or raw_action >= 5:
                continue

            pos = graph.pos[slot]
            if raw_action >= len(graph.adj[pos]):
                actions[slot] = 7
                reason = "invalid_action"
                reserved_nodes.add(pos)
            else:
                nxt, _ = graph.adj[pos][raw_action]
                target_is_vacated = (
                    self.allow_vacating_node_entry
                    and nxt in vacating_nodes
                    and nxt in occupied_nodes
                )
                if nxt in occupied_nodes and not target_is_vacated:
                    actions[slot] = 7
                    reason = "occupied_node"
                elif nxt in reserved_nodes and not target_is_vacated:
                    actions[slot] = 7
                    reason = "reserved_node"
                elif (nxt, pos) in reserved_edges:
                    actions[slot] = 7
                    reason = "reserved_reverse_edge"
                elif not target_is_vacated and graph.is_node_entry_blocked(nxt, slot):
                    actions[slot] = 7
                    reason = "clearance_or_occupancy"
                else:
                    reserved_nodes.add(nxt)
                    reserved_edges.add((pos, nxt))
                    if pos in occupied_nodes:
                        vacating_nodes.add(pos)

            if reason is not None and actions[slot] != raw_action:
                self.last_filter_reasons[slot] = reason
                self._record_intervention(step, slot, raw_action, actions[slot], reason)
        return actions

    def _route_next_action(self, slot: int) -> int | None:
        """Return the action leading to the next node of the current route."""
        graph = self.env.graph_env
        if graph.on_edge[slot]:
            return None
        current = graph.pos[slot]
        route = (
            list(self.env.agents[slot].planned_path)
            if self.algorithm == "MAPPO"
            else list(self.fcfs_routes.get(slot, []))
        )
        if not route:
            return None
        try:
            idx = route.index(current)
        except ValueError:
            return None
        if idx + 1 >= len(route):
            return None
        next_node = route[idx + 1]
        for action_id, (neighbor, edge_idx) in enumerate(graph.adj.get(current, [])):
            if action_id >= 5:
                break
            if neighbor == next_node and not graph.edges[edge_idx].closed:
                return action_id
        return None

    def _find_blocker_for_node(self, slot: int, next_node: int, active_slots: list[int]) -> int | None:
        """Find the active aircraft currently reserving a desired downstream node."""
        graph = self.env.graph_env
        for other in active_slots:
            if other == slot:
                continue
            if not graph.on_edge[other] and graph.pos[other] == next_node:
                return other
            if graph.on_edge[other] and graph.edge_idx[other] is not None:
                edge = graph.edges[graph.edge_idx[other]]
                if edge.end == next_node:
                    return other
        return None

    def _blocker_release_action(self, blocker: int, step: int) -> int | None:
        """Return a safe route-follow action that lets a node blocker vacate."""
        graph = self.env.graph_env
        if graph.on_edge[blocker]:
            if getattr(graph, "clearance_blocked", [False] * self.slots)[blocker]:
                return None
            if getattr(graph, "turn_brake_active", [False] * self.slots)[blocker]:
                return None
            if float(graph.speed[blocker]) < float(getattr(graph, "max_speed", graph.speed[blocker])) - 1e-6:
                return 6
            return None
        action = self._route_next_action(blocker)
        if action is None:
            if not self._replan_slot(blocker, step, "stall"):
                return None
            action = self._route_next_action(blocker)
        return action if action is not None and action < 5 else None
    def _apply_stall_path_recovery(
        self, actions: list[int], active_slots: list[int], step: int
    ) -> dict[int, int]:
        """Propose a route-follow action after the first safety-filter pass."""
        if not self.enable_stall_path_recovery:
            return {}
        graph = self.env.graph_env
        self.recovery_kinds = {}
        proposals: dict[int, int] = {}
        for slot in active_slots:
            if self.stall_count[slot] < self.stall_recovery_trigger_steps:
                continue
            if step - self._last_stall_recovery_step[slot] < self.stall_recovery_cooldown_steps:
                continue
            first_filter_reason = self.last_filter_reasons.get(slot)
            # Rate-limit diagnostics as well as proposals; otherwise a single
            # blocked slot would append one identical row on every step.
            self._last_stall_recovery_step[slot] = step
            diag = {
                "step": step,
                "slot": slot,
                "flight_index": self.slot_to_flight[slot],
                "stall_count": int(self.stall_count[slot]),
                "on_edge": bool(graph.on_edge[slot]),
                "position": graph.pos[slot],
                "edge_idx": graph.edge_idx[slot],
                "progress": round(float(graph.progress[slot]), 6),
                "post_first_filter_action": int(actions[slot]),
                "first_filter_reason": first_filter_reason,
                "edge_hold_reason": getattr(graph, "edge_hold_reason", [None] * self.slots)[slot],
                "clearance_blocker_slot": getattr(graph, "clearance_blocker_by_agent", [None] * self.slots)[slot],
                "clearance_min_distance_sq": round(float(getattr(graph, "clearance_min_distance_sq_by_agent", [float("inf")] * self.slots)[slot]), 6)
                if getattr(graph, "clearance_min_distance_sq_by_agent", [float("inf")] * self.slots)[slot] != float("inf")
                else None,
            }
            if graph.on_edge[slot]:
                clearance_blocked = bool(
                    getattr(graph, "clearance_blocked", [False] * self.slots)[slot]
                )
                diag["clearance_blocked"] = clearance_blocked
                if clearance_blocked:
                    diag["outcome"] = "edge_clearance_hold"
                    self.stall_recovery_diagnostics.append(diag)
                    blocker = getattr(graph, "clearance_blocker_by_agent", [None] * self.slots)[slot]
                    if blocker is not None and blocker in active_slots and blocker != slot:
                        blocker_clearance = bool(
                            getattr(graph, "clearance_blocked", [False] * self.slots)[blocker]
                        )
                        blocker_reason = getattr(graph, "edge_hold_reason", [None] * self.slots)[blocker]
                        blocker_on_edge = bool(graph.on_edge[blocker])
                        blocker_speed = float(graph.speed[blocker]) if blocker_on_edge else 0.0
                        blocker_turn_brake = bool(
                            getattr(graph, "turn_brake_active", [False] * self.slots)[blocker]
                        )
                        if (
                            blocker_on_edge
                            and not blocker_clearance
                            and blocker_reason is None
                            and not blocker_turn_brake
                            and blocker_speed < float(getattr(graph, "max_speed", blocker_speed)) - 1e-6
                            and actions[blocker] != 6
                        ):
                            actions[blocker] = 6
                            proposals[blocker] = 6
                            self.recovery_kinds[blocker] = "clearance_release"
                            self._last_stall_recovery_step[blocker] = step
                            self.stall_recovery_diagnostics.append({
                                "step": step,
                                "slot": blocker,
                                "flight_index": self.slot_to_flight[blocker],
                                "stall_count": int(self.stall_count[blocker]),
                                "on_edge": True,
                                "position": graph.pos[blocker],
                                "edge_idx": graph.edge_idx[blocker],
                                "progress": round(float(graph.progress[blocker]), 6),
                                "post_first_filter_action": int(actions[blocker]),
                                "first_filter_reason": self.last_filter_reasons.get(blocker),
                                "outcome": "proposed_clearance_release",
                                "released_for_slot": slot,
                            })
                    continue
                speed = float(graph.speed[slot])
                turn_brake_active = bool(getattr(graph, "turn_brake_active", [False] * self.slots)[slot])
                if turn_brake_active and speed > float(getattr(graph, "turn_speed_limit", speed)):
                    diag["outcome"] = "edge_turn_brake_hold"
                    self.stall_recovery_diagnostics.append(diag)
                    continue
                if speed >= float(getattr(graph, "max_speed", speed)) - 1e-6:
                    diag["outcome"] = "edge_at_speed_limit"
                    self.stall_recovery_diagnostics.append(diag)
                    continue
                # Action 6 is an explicit acceleration request.  The physical
                # environment still applies gate/turn speed caps, following
                # distance, node occupancy and continuous clearance checks.
                actions[slot] = 6
                proposals[slot] = 6
                self.recovery_kinds[slot] = "motion"
                diag.update({"outcome": "proposed_edge_motion_recovery", "candidate_action": 6})
                self.stall_recovery_diagnostics.append(diag)
                continue
            if actions[slot] < 5:
                diag["outcome"] = "movement_already_admissible"
                self.stall_recovery_diagnostics.append(diag)
                continue
            route = (
                list(self.env.agents[slot].planned_path)
                if self.algorithm == "MAPPO"
                else list(self.fcfs_routes.get(slot, []))
            )
            diag["route_length"] = len(route)
            diag["route_contains_position"] = graph.pos[slot] in route
            action_id = self._route_next_action(slot)
            if action_id is None:
                diag["outcome"] = "no_route_successor"
                self.stall_recovery_diagnostics.append(diag)
                continue
            current = graph.pos[slot]
            next_node, edge_idx = graph.adj[current][action_id]
            blocker = self._find_blocker_for_node(slot, next_node, active_slots)
            if blocker is not None and blocker not in proposals:
                release_action = self._blocker_release_action(blocker, step)
                if release_action is not None:
                    actions[blocker] = release_action
                    proposals[blocker] = release_action
                    self.recovery_kinds[blocker] = "blocker_release"
                    self._last_stall_recovery_step[blocker] = step
                    diag["blocker_release_slot"] = blocker
                    diag["blocker_release_action"] = int(release_action)
                    self.stall_recovery_diagnostics.append({
                        "step": step,
                        "slot": blocker,
                        "flight_index": self.slot_to_flight[blocker],
                        "stall_count": int(self.stall_count[blocker]),
                        "on_edge": bool(graph.on_edge[blocker]),
                        "position": graph.pos[blocker],
                        "edge_idx": graph.edge_idx[blocker],
                        "progress": round(float(graph.progress[blocker]), 6),
                        "post_first_filter_action": int(release_action),
                        "outcome": "proposed_blocker_release",
                        "released_for_slot": slot,
                    })
            if graph.edges[edge_idx].closed:
                diag.update({"outcome": "route_edge_closed", "candidate_action": int(action_id), "candidate_node": next_node})
                self.stall_recovery_diagnostics.append(diag)
                continue
            actions[slot] = action_id
            proposals[slot] = action_id
            self.recovery_kinds[slot] = "path"
            self._last_stall_recovery_step[slot] = step
            diag.update({
                "outcome": "proposed_to_second_filter",
                "candidate_action": int(action_id),
                "candidate_node": next_node,
                "candidate_edge_idx": edge_idx,
            })
            self.stall_recovery_diagnostics.append(diag)
        return proposals

    def _record_stall_path_recovery(
        self, step: int, proposals: dict[int, int], actions: list[int]
    ) -> None:
        for slot, proposed_action in proposals.items():
            if actions[slot] != proposed_action:
                for row in reversed(self.stall_recovery_diagnostics):
                    if row.get("step") == step and row.get("slot") == slot:
                        row["outcome"] = "rejected_by_second_filter"
                        row["second_filter_reason"] = self.last_filter_reasons.get(slot)
                        break
                continue
            flight_index = self.slot_to_flight[slot]
            if flight_index is None:
                continue
            recovery_kind = self.recovery_kinds.get(slot, "path")
            self.execution_events.append({
                "step": step,
                "slot": slot,
                "flight_index": flight_index,
                "type": (
                    "stall_motion_recovery" if recovery_kind == "motion"
                    else "clearance_release_recovery" if recovery_kind == "clearance_release"
                    else "stall_path_recovery"
                ),
                "action": int(proposed_action),
            })
            if recovery_kind == "motion":
                self.stall_motion_recovery_events += 1
            elif recovery_kind == "clearance_release":
                self.clearance_release_recovery_events += 1
            elif recovery_kind == "blocker_release":
                self.blocker_release_recovery_events += 1
            else:
                self.stall_path_recovery_events += 1
            for row in reversed(self.stall_recovery_diagnostics):
                if row.get("step") == step and row.get("slot") == slot:
                    row["outcome"] = "accepted_by_second_filter"
                    break

    def _replan_slot(self, slot: int, step: int, trigger: str) -> bool:
        """Replan one slot without resetting or retiring its aircraft."""
        flight_index = self.slot_to_flight[slot]
        if flight_index is None:
            return False
        planner_graph = self.env.graph_env.get_graph_for_planner()
        graph_env = self.env.graph_env
        pos = graph_env.pos[slot]
        goal = graph_env.goal[slot]
        # Preserve an edge that is already physically committed; only its
        # suffix can be changed by replanning.
        committed_prefix = [pos]
        plan_start = pos
        committed_edge = None
        if graph_env.on_edge[slot] and graph_env.edge_idx[slot] is not None:
            edge = graph_env.edges[graph_env.edge_idx[slot]]
            committed_prefix = [edge.start, edge.end]
            plan_start = edge.end
            committed_edge = [edge.start, edge.end]
        blocked_nodes = set()
        if trigger == "stall":
            for other_slot, other_flight in enumerate(self.slot_to_flight):
                if other_slot == slot or other_flight is None:
                    continue
                if graph_env.done[other_slot] or graph_env.crashed[other_slot]:
                    continue
                if graph_env.on_edge[other_slot] and graph_env.edge_idx[other_slot] is not None:
                    blocked_nodes.add(graph_env.edges[graph_env.edge_idx[other_slot]].end)
                else:
                    blocked_nodes.add(graph_env.pos[other_slot])
            blocked_nodes.discard(pos)
            blocked_nodes.discard(plan_start)
            blocked_nodes.discard(goal)
        started = time.perf_counter()
        suffix = self.env.planner.plan_path(
            plan_start, goal, planner_graph, blocked_nodes=blocked_nodes
        ) or []
        # Dynamic occupancy is temporary. If it disconnects the graph, retain
        # a valid static route rather than converting a recoverable wait into a
        # false no-route condition.
        if not suffix and blocked_nodes:
            suffix = self.env.planner.plan_path(plan_start, goal, planner_graph) or []
        if suffix:
            path = committed_prefix[:-1] + suffix if committed_prefix[-1] == suffix[0] else committed_prefix + suffix
        else:
            path = committed_prefix
        self.metrics.replan_ms.append((time.perf_counter() - started) * 1000.0)
        success = bool(path and (pos == goal or path[-1] == goal))
        if self.algorithm == "MAPPO":
            agent = self.env.agents[slot]
            agent.planned_path = path or [pos]
            agent.planned_path_set = set(agent.planned_path)
            agent.path_index = 0
        else:
            self.fcfs_routes[slot] = path
        self.metrics.events.append({
            "step": step,
            "type": "stall_replanning" if trigger == "stall" else "route_replanning",
            "flight_index": flight_index,
            "slot": slot,
            "trigger": trigger,
            "success": success,
            "position": pos,
            "goal": goal,
            "committed_edge": committed_edge,
            "blocked_nodes": sorted(blocked_nodes),
        })
        if trigger == "stall":
            self.execution_events.append({
                "step": step,
                "slot": slot,
                "flight_index": flight_index,
                "type": "stall_replanning",
                "success": success,
                "position": pos,
                "goal": goal,
                "committed_edge": committed_edge,
                "blocked_nodes": sorted(blocked_nodes),
            })
            self.stall_replanning_events += 1
        return success

    def _replan_active_routes(self) -> None:
        for slot, flight_index in enumerate(self.slot_to_flight):
            if flight_index is not None:
                self._replan_slot(slot, self.env.graph_env.step_count, "closure")

    def _write_layered_outputs(self, summary: dict, final_step: int) -> dict:
        result_dir = self.args.result_dir
        os.makedirs(result_dir, exist_ok=True)
        hours = max(final_step * self.args.time_scale / 3600.0, 1e-9)
        completed = [r for r in self.metrics.records if r.status == "completed"]
        target_indices = {
            i for i, flight in enumerate(self.flights)
            if str(flight.get("agent_role", "TARGET")).upper() == "TARGET"
        }
        target_completed = [r for r in completed if r.flight_index in target_indices]
        transit_completed = [r for r in completed if r.flight_index not in target_indices]
        target_total = len(target_indices)
        transit_total = len(self.flights) - target_total
        target_rate = 100.0 * len(target_completed) / target_total if target_total else 0.0
        transit_rate = 100.0 * len(transit_completed) / transit_total if transit_total else 0.0
        target_throughput = len(target_completed) / hours
        all_throughput = len(completed) / hours

        # Paper metrics use released tasks as the denominator.  The full
        # schedule count remains available separately for audit purposes.
        released_indices = {
            i for i, release in enumerate(self.scenario.release_steps)
            if release <= final_step
        }
        released_completed = [r for r in completed if r.flight_index in released_indices]
        released_target_indices = released_indices & target_indices
        released_transit_indices = released_indices - target_indices
        released_target_completed = [
            r for r in released_completed if r.flight_index in target_indices
        ]
        released_transit_completed = [
            r for r in released_completed if r.flight_index not in target_indices
        ]
        released_count = len(released_indices)
        released_target_count = len(released_target_indices)
        released_transit_count = len(released_transit_indices)
        overall_rate = 100.0 * len(released_completed) / released_count if released_count else 0.0
        target_rate_released = (
            100.0 * len(released_target_completed) / released_target_count
            if released_target_count else 0.0
        )
        target_throughput = len(released_target_completed) / hours
        all_throughput = len(released_completed) / hours
        conflict_frequency = (
            100.0 * self.conflict_protection_events / released_count
            if released_count else 0.0
        )
        stall_replan_frequency = (
            100.0 * self.stall_replanning_events / released_count
            if released_count else 0.0
        )

        summary.update({
            "completion_rate_pct": round(overall_rate, 2),
            "safe_completion_rate_pct": round(overall_rate, 2),
            "final_service_rate_pct": round(overall_rate, 2),
            "total_schedule_task_count": len(self.flights),
            "released_task_count": released_count,
            "released_target_task_count": released_target_count,
            "released_transit_task_count": released_transit_count,
            "overall_task_completion_rate_pct": round(overall_rate, 2),
            "target_total_count": target_total,
            "target_success_count": len(released_target_completed),
            "target_aircraft_completion_rate_pct": round(target_rate_released, 2),
            "target_completion_rate_pct": round(target_rate_released, 2),
            "transit_total_count": transit_total,
            "transit_success_count": len(transit_completed),
            "transit_completion_rate_pct": round(transit_rate, 2),
            "all_flights_throughput_per_hour": round(all_throughput, 2),
            "target_throughput_per_hour": round(target_throughput, 2),
            "throughput_per_hour": round(target_throughput, 2),
            "raw_action_count": len(self.raw_action_rows),
            "conflict_protection_events": self.conflict_protection_events,
            "conflict_protection_intervention_count": self.conflict_protection_events,
            "conflict_protection_intervention_frequency": self.conflict_protection_events,
            "conflict_protection_intervention_frequency_pct": round(
                100.0 * self.conflict_protection_events / max(len(self.raw_action_rows), 1), 2
            ),
            "stall_replanning_events": self.stall_replanning_events,
            "stall_replanning_intervention_count": self.stall_replanning_events,
            "stall_replanning_intervention_frequency": self.stall_replanning_events,
            "stall_replanning_intervention_frequency_pct": round(
                100.0 * self.stall_replanning_events / max(len(self.raw_action_rows), 1), 2
            ),
            "replanning_intervention_count": self.stall_replanning_events,
            "R_stall": self.stall_replanning_events,
            "stall_replanning_definition": "count of replans triggered by persistent no-progress stall",
            "operational_timeout_steps": self.operational_timeout_steps,
            "operational_timeout_count": self.operational_timeout_events,
            "operational_timeout_definition": "active aircraft retired after exceeding the execution service window",
            "conflict_protection_definition": "count of nominal actions replaced by the execution safety filter",
            "conflict_protection_reason_counts": dict(sorted(self.filter_reason_counts.items())),
            "execution_mask_alignment": not bool(
                getattr(self.args, "disable_execution_mask_alignment", False)
            ),
            "approach_reservation_progress": round(
                float(getattr(self.env.graph_env, "approach_reservation_progress", 0.60)), 3
            ),
            "mean_executed_taxi_time_min": summary.get("avg_taxi_min", 0.0),
            "target_aircraft_throughput_per_hour": round(target_throughput, 2),
            "stall_path_recovery_events": self.stall_path_recovery_events,
            "stall_motion_recovery_events": self.stall_motion_recovery_events,
            "clearance_release_recovery_events": self.clearance_release_recovery_events,
            "blocker_release_recovery_events": self.blocker_release_recovery_events,
            "stall_recovery_diagnostic_count": len(self.stall_recovery_diagnostics),
            "stall_recovery_outcome_counts": {
                outcome: sum(1 for row in self.stall_recovery_diagnostics if row.get("outcome") == outcome)
                for outcome in sorted({row.get("outcome") for row in self.stall_recovery_diagnostics})
            },
            "safety_violation_count": self.safety_violation_count,
            "collision_event_details_count": len(self.collision_details),
            "execution_protocol": "layered_proposal_protection_replanning",
            "terminal_statuses": ["completed", "timeout"],
        })

        prefix = self.algorithm.replace("+", "_").replace(" ", "_")
        with open(os.path.join(result_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(result_dir, f"{prefix}_execution_events.json"), "w", encoding="utf-8") as handle:
            json.dump(self.execution_events, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(result_dir, f"{prefix}_collision_events.json"), "w", encoding="utf-8") as handle:
            json.dump(self.collision_details, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(result_dir, f"{prefix}_stall_recovery_diagnostics.json"), "w", encoding="utf-8") as handle:
            json.dump(self.stall_recovery_diagnostics, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(result_dir, f"{prefix}_raw_actions.csv"), "w", newline="", encoding="utf-8") as handle:
            fields = [
                "step", "slot", "flight_index", "raw_action", "executed_action",
                "action_changed", "position_before", "on_edge_before", "progress_before",
                "position_after", "on_edge_after", "progress_after",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.raw_action_rows)
        return summary

    def run(self) -> dict:
        import numpy as np

        np.random.seed(self.args.seed)
        import torch
        torch.manual_seed(self.args.seed)
        print(f"\n  [{self.algorithm}] layered rolling evaluation: {len(self.flights)} flights, {self.slots} slots")
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
            raw_actions = (
                self._mappo_actions(obs, masks, active_slots)
                if self.algorithm == "MAPPO"
                else self._fcfs_actions(active_slots)
            )
            executed_actions = list(raw_actions)
            self._apply_safety_filter(executed_actions, active_slots, step)
            recovery_proposals = self._apply_stall_path_recovery(
                executed_actions, active_slots, step
            )
            if recovery_proposals:
                self._apply_safety_filter(executed_actions, active_slots, step)
            self._record_stall_path_recovery(
                step, recovery_proposals, executed_actions
            )
            prior_states = {
                slot: (
                    self.env.graph_env.pos[slot],
                    self.env.graph_env.on_edge[slot],
                    self.env.graph_env.edge_idx[slot],
                    self.env.graph_env.progress[slot],
                )
                for slot in active_slots
            }
            self.env.step(executed_actions)
            collision_events = getattr(self.env.graph_env, "collision_events", [])
            if len(collision_events) > self._collision_event_cursor:
                for event in collision_events[self._collision_event_cursor:]:
                    detail = dict(event)
                    detail["slot_to_flight"] = {
                        str(slot): self.slot_to_flight[slot]
                        for slot in active_slots
                        if self.slot_to_flight[slot] is not None
                    }
                    self.collision_details.append(detail)
                self._collision_event_cursor = len(collision_events)
            self._capture_priority_frame(step + 1, executed_actions)

            for slot in list(active_slots):
                flight_index = self.slot_to_flight[slot]
                if flight_index is None:
                    continue
                before = prior_states[slot]
                graph = self.env.graph_env
                moved = graph.pos[slot] != before[0]
                if before[1] and graph.on_edge[slot]:
                    moved = moved or graph.edge_idx[slot] != before[2] or graph.progress[slot] > before[3] + 1e-5
                elif not before[1] and graph.on_edge[slot]:
                    moved = moved or graph.progress[slot] > 1e-5
                self.raw_action_rows.append({
                    "step": step,
                    "slot": slot,
                    "flight_index": flight_index,
                    "raw_action": int(raw_actions[slot]),
                    "executed_action": int(executed_actions[slot]),
                    "action_changed": int(raw_actions[slot] != executed_actions[slot]),
                    "position_before": before[0],
                    "on_edge_before": int(before[1]),
                    "progress_before": round(float(before[3]), 6),
                    "position_after": graph.pos[slot],
                    "on_edge_after": int(graph.on_edge[slot]),
                    "progress_after": round(float(graph.progress[slot]), 6),
                })
                if graph.crashed[slot]:
                    self.safety_violation_count += 1
                    self._retire_slot(slot, step + 1, "collision")
                    continue
                if graph.done[slot]:
                    self._retire_slot(slot, step + 1, "completed")
                    continue
                rec = self.metrics.records[flight_index]
                if (
                    rec.spawn_step is not None
                    and step + 1 - rec.spawn_step >= self.operational_timeout_steps
                ):
                    self.execution_events.append({
                        "step": step + 1,
                        "slot": slot,
                        "flight_index": flight_index,
                        "type": "operational_timeout",
                        "elapsed_steps": step + 1 - rec.spawn_step,
                        "timeout_limit_steps": self.operational_timeout_steps,
                    })
                    self.operational_timeout_events += 1
                    self._retire_slot(slot, step + 1, "timeout")
                    continue
                self.stall_count[slot] = 0 if moved else self.stall_count[slot] + 1
                self.metrics.record_stall(flight_index, self.stall_count[slot])
                if self.enable_stall_replanning and self.stall_count[slot] >= self.stall_threshold_steps:
                    self._replan_slot(slot, step + 1, "stall")
                    self.stall_count[slot] = 0

            if step and step % 500 == 0:
                completed = sum(r.status == "completed" for r in self.metrics.records)
                print(f"  Step {step:<6} | active {len(active_slots):<2} | completed {completed}/{len(self.flights)}")

        summary = self.metrics.export(self.args.result_dir, final_step, self.edge_len_map)
        self._export_priority_micro_log()
        return self._write_layered_outputs(summary, final_step)


def run_layered_mappo(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "MAPPO").run()


def run_layered_fcfs(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "FCFS").run()
