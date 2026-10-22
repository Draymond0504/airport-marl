"""Operational rolling runner with corrected edge logging and stronger events.

This compatibility entry point preserves the original runner while applying
the requested protocol changes before dispatching to its main function.
"""

from __future__ import annotations

import csv
import os

import paper_rolling_executor as paper
from paper_rolling_executor import RollingScenario
from layered_rolling_executor import LayeredRollingPaperExecutor


# Three representative flights, distributed across the schedule.
paper.PRIORITY_TARGETS = {120, 180, 240}


_ORIGINAL_SCENARIO_INIT = RollingScenario.__init__


def _scenario_init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_SCENARIO_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)
    if scenario == "closure":
        # Preserve the effective-closure trigger locations used in the final
        # paper protocol; only extend the active duration from 800 to 1200.
        service_horizon = max(self.release_steps, default=max_steps)
        self.closure_plan = [
            (max(1, int(service_horizon * 0.35)), 2400, (102, 103)),
            (max(2, int(service_horizon * 0.70)), 2400, (304, 305)),
        ]


RollingScenario.__init__ = _scenario_init


_ORIGINAL_EXECUTOR_INIT = LayeredRollingPaperExecutor.__init__


def _executor_init(self, args, flights, config, edge_len_map, algorithm):
    _ORIGINAL_EXECUTOR_INIT(self, args, flights, config, edge_len_map, algorithm)
    self.edge_traversal_rows = []
    self.edge_occupancy_rows = []
    original_step = self.env.step

    def traced_step(actions):
        graph = self.env.graph_env
        before = {}
        occupancy_step = int(graph.step_count)
        for slot in range(self.slots):
            if graph.on_edge[slot] and graph.edge_idx[slot] is not None:
                edge = graph.edges[graph.edge_idx[slot]]
                flight_index = self.slot_to_flight[slot]
                if flight_index is not None:
                    self.edge_occupancy_rows.append({
                        "step": occupancy_step,
                        "flight_index": flight_index,
                        "slot": slot,
                        "edge_start": edge.start,
                        "edge_end": edge.end,
                    })
            if not graph.on_edge[slot] or graph.edge_idx[slot] is None:
                before[slot] = None
                continue
            edge = graph.edges[graph.edge_idx[slot]]
            before[slot] = (edge.start, edge.end)
        result = original_step(actions)
        graph_after = self.env.graph_env
        for slot, edge_pair in before.items():
            if edge_pair is None:
                continue
            if not graph_after.on_edge[slot] and graph_after.pos[slot] == edge_pair[1]:
                flight_index = self.slot_to_flight[slot]
                if flight_index is not None:
                    self.edge_traversal_rows.append({
                        "step": self.env.graph_env.step_count,
                        "flight_index": flight_index,
                        "slot": slot,
                        "edge_start": edge_pair[0],
                        "edge_end": edge_pair[1],
                    })
        return result

    self.env.step = traced_step


LayeredRollingPaperExecutor.__init__ = _executor_init


_ORIGINAL_WRITE = LayeredRollingPaperExecutor._write_layered_outputs


def _write_with_edge_log(self, summary, final_step):
    result = _ORIGINAL_WRITE(self, summary, final_step)
    prefix = self.algorithm.replace("+", "_").replace(" ", "_")
    path = os.path.join(self.args.result_dir, f"{prefix}_edge_traversals.csv")
    fields = ["step", "flight_index", "slot", "edge_start", "edge_end"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(self.edge_traversal_rows)
    occupancy_path = os.path.join(self.args.result_dir, f"{prefix}_edge_occupancy.csv")
    with open(occupancy_path, "w", newline="", encoding="utf-8") as handle:
        fields = ["step", "flight_index", "slot", "edge_start", "edge_end"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(self.edge_occupancy_rows)
    return result


LayeredRollingPaperExecutor._write_layered_outputs = _write_with_edge_log


if __name__ == "__main__":
    import run_layered_paper_exp as base

    base.main()
