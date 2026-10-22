"""Diagnostic wrapper for the layered executor.

It preserves v3 behavior and records the environment's detailed collision
payload before the base executor retires the involved slots.
"""

from __future__ import annotations

from layered_rolling_executor_v3 import LayeredRollingPaperExecutor as _V3Executor


class LayeredRollingPaperExecutor(_V3Executor):
    def _retire_slot(self, slot: int, step: int, status: str) -> None:
        if status == "collision":
            details = list(getattr(self.env.graph_env, "collision_events", []))
            payload = details[-1] if details else None
            self.execution_events.append({
                "step": step,
                "slot": slot,
                "flight_index": self.slot_to_flight[slot],
                "type": "safety_violation_detail",
                "collision": payload,
            })
        super()._retire_slot(slot, step, status)


def run_layered_mappo(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "MAPPO").run()


def run_layered_fcfs(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "FCFS").run()
