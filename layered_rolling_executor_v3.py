"""Layered executor with per-aircraft operational timeout handling.

The previous executor only converted active flights to ``timeout`` after the
global horizon.  A stalled aircraft could therefore occupy a rolling slot for
the entire experiment and prevent all queued flights from being admitted.
This wrapper retires an aircraft after a bounded operational mission window,
while preserving the terminal status as ``timeout`` rather than collision or
deadlock.
"""

from __future__ import annotations

from layered_rolling_executor_v2 import LayeredRollingPaperExecutor as _V2Executor


class LayeredRollingPaperExecutor(_V2Executor):
    def __init__(self, args, flights, config, edge_len_map, algorithm):
        super().__init__(args, flights, config, edge_len_map, algorithm)
        configured = getattr(args, "flight_timeout_steps", None)
        self.flight_timeout_steps = int(
            configured if configured is not None
            else max(5 * self.stall_threshold_steps, 900)
        )

    def _retire_expired_flights(self, step: int) -> None:
        """Release rolling slots whose bounded mission window has expired."""
        for slot, flight_index in enumerate(list(self.slot_to_flight)):
            if flight_index is None:
                continue
            record = self.metrics.records[flight_index]
            if record.spawn_step is None:
                continue
            if step - record.spawn_step < self.flight_timeout_steps:
                continue

            self.execution_events.append({
                "step": step,
                "slot": slot,
                "flight_index": flight_index,
                "type": "flight_timeout",
                "timeout_window_steps": self.flight_timeout_steps,
            })
            self._retire_slot(slot, step, "timeout")

    def _admit_due_flights(self, step: int) -> None:
        self._retire_expired_flights(step)
        super()._admit_due_flights(step)


def run_layered_mappo(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "MAPPO").run()


def run_layered_fcfs(args, flights, config, edge_len_map, result_dir):
    args.result_dir = result_dir
    return LayeredRollingPaperExecutor(args, flights, config, edge_len_map, "FCFS").run()
