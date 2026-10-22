"""Compatibility fix for the layered executor's planner/environment split."""

from __future__ import annotations

import time

from layered_rolling_executor import LayeredRollingPaperExecutor as _LayeredBase


class LayeredRollingPaperExecutor(_LayeredBase):
    def _replan_slot(self, slot: int, step: int, trigger: str) -> bool:
        flight_index = self.slot_to_flight[slot]
        if flight_index is None:
            return False
        graph_env = self.env.graph_env
        planner_graph = graph_env.get_graph_for_planner()
        pos = graph_env.pos[slot]
        goal = graph_env.goal[slot]
        started = time.perf_counter()
        path = self.env.planner.plan_path(pos, goal, planner_graph) or []
        self.metrics.replan_ms.append((time.perf_counter() - started) * 1000.0)
        success = bool(path)
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
            })
            self.stall_replanning_events += 1
        return success


def run_layered_mappo(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "MAPPO").run()


def run_layered_fcfs(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "FCFS").run()
