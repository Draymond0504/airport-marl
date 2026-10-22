from __future__ import annotations

import time
import json
import os
import csv
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentRecord:
    agent_id: int
    spawn_step: Optional[int] = None
    finish_step: Optional[int] = None
    crashed: bool = False
    path_nodes: list = field(default_factory=list)
    taxi_steps: Optional[int] = None
    taxi_sec: Optional[float] = None
    moving_steps: int = 0
    moving_sec: Optional[float] = None
    onroute_wait_sec: Optional[float] = None
    path_dist: float = 0.0
    priority: int = 1
    delay_steps: int = 0
    wait_time_min: float = 0.0
    collision_count: int = 0
    deadlock_count: int = 0
    _last_pos: Optional[int] = field(default=None, repr=False)
    _last_on_edge: bool = field(default=False, repr=False)
    _last_progress: float = field(default=0.0, repr=False)


class SimulationTracker:
    def __init__(self, num_agents, snapshot_interval=10, time_step_sec=5.0, algorithm_name="MAPPO"):
        self.num_agents = num_agents
        self.snapshot_interval = snapshot_interval
        self.time_step_sec = time_step_sec
        self.algorithm_name = algorithm_name
        self.agents = [AgentRecord(agent_id=i) for i in range(num_agents)]
        self.snapshots = []
        self._inference_start = None
        self.inference_latencies_ms = []
        self.replan_latencies_ms = []
        self._replan_start = None
        self.deadlock_count = 0
        self.collision_count = 0
        self._wall_start = time.time()
        self._last_success_cnt = 0

    def record_step(self, step, graph_env):
        active_count = 0
        start_delays = getattr(graph_env, "start_delays", [0] * self.num_agents)
        for i in range(self.num_agents):
            rec = self.agents[i]
            if rec.spawn_step is None and start_delays[i] == 0 and not graph_env.done[i]:
                rec.spawn_step = step
                rec._last_pos = graph_env.pos[i]
                rec._last_on_edge = graph_env.on_edge[i]
                rec._last_progress = graph_env.progress[i] if graph_env.on_edge[i] else 0.0
                rec.moving_steps = 0
            if not graph_env.done[i] and start_delays[i] == 0:
                rec.path_nodes.append(graph_env.pos[i])
                active_count += 1
                cur_pos = graph_env.pos[i]
                cur_on_edge = graph_env.on_edge[i]
                cur_prog = graph_env.progress[i] if cur_on_edge else 0.0
                if rec._last_pos is not None:
                    moved = (cur_pos != rec._last_pos
                             or cur_on_edge != rec._last_on_edge
                             or abs(cur_prog - rec._last_progress) > 1e-4)
                    if moved:
                        rec.moving_steps += 1
                rec._last_pos = cur_pos
                rec._last_on_edge = cur_on_edge
                rec._last_progress = cur_prog
            if rec.finish_step is None and graph_env.done[i]:
                rec.finish_step = step
                rec.crashed = graph_env.crashed[i]

        if step % self.snapshot_interval == 0:
            success_cnt = sum(1 for i in range(self.num_agents)
                              if graph_env.done[i] and not graph_env.crashed[i])
            delta = success_cnt - self._last_success_cnt
            throughput = delta / self.snapshot_interval
            self._last_success_cnt = success_cnt
            self.snapshots.append({
                "step": step,
                "active": active_count,
                "cumulative_success": success_cnt,
                "throughput_per_step": throughput,
                "time_min": step * self.time_step_sec / 60.0,
            })

    def start_inference(self):
        self._inference_start = time.perf_counter()

    def end_inference(self):
        if self._inference_start is not None:
            self.inference_latencies_ms.append(
                (time.perf_counter() - self._inference_start) * 1000.0)
            self._inference_start = None

    def start_replan(self):
        self._replan_start = time.perf_counter()

    def end_replan(self):
        if self._replan_start is not None:
            self.replan_latencies_ms.append(
                (time.perf_counter() - self._replan_start) * 1000.0)
            self._replan_start = None

    def record_deadlock(self, count=1, agent_id: Optional[int] = None):
        self.deadlock_count += count
        if agent_id is not None and 0 <= agent_id < self.num_agents:
            self.agents[agent_id].deadlock_count += count

    def record_collision(self, count=1, agent_id: Optional[int] = None):
        self.collision_count += count
        if agent_id is not None and 0 <= agent_id < self.num_agents:
            self.agents[agent_id].collision_count += count

    def finalize(self, graph_env, flights, edge_len_map, delay_log=None):
        for i in range(self.num_agents):
            rec = self.agents[i]
            rec.crashed = graph_env.crashed[i]
            rec.priority = graph_env.priority[i]
            if delay_log:
                rec.delay_steps = delay_log[i]
                rec.wait_time_min = rec.delay_steps * self.time_step_sec / 60.0
            if rec.spawn_step is not None and rec.finish_step is not None:
                rec.taxi_steps = rec.finish_step - rec.spawn_step
                rec.taxi_sec = rec.taxi_steps * self.time_step_sec
                rec.moving_sec = rec.moving_steps * self.time_step_sec
                rec.onroute_wait_sec = max(0.0, rec.taxi_sec - rec.moving_sec)
            nodes = rec.path_nodes
            if nodes:
                deduped = [nodes[0]]
                for nd in nodes[1:]:
                    if nd != deduped[-1]:
                        deduped.append(nd)
                rec.path_dist = sum(
                    edge_len_map.get((deduped[k], deduped[k + 1]), 0.0)
                    for k in range(len(deduped) - 1)
                )

    def get_summary(self):
        eventual_success_agents = [
            r for r in self.agents
            if not r.crashed and r.finish_step is not None
        ]
        success_agents = [
            r for r in eventual_success_agents
            if r.collision_count == 0
        ]
        taxi_sec   = [r.taxi_sec        for r in success_agents if r.taxi_sec is not None]
        moving_sec = [r.moving_sec      for r in success_agents if r.moving_sec is not None]
        wait_sec   = [r.onroute_wait_sec for r in success_agents if r.onroute_wait_sec is not None]
        dists      = [r.path_dist       for r in success_agents if r.path_dist > 0]

        n = len(success_agents)
        eventual_n = len(eventual_success_agents)
        total = self.num_agents
        rate  = n / total * 100.0 if total > 0 else 0.0
        eventual_rate = eventual_n / total * 100.0 if total > 0 else 0.0

        def avg(lst): return sum(lst) / len(lst) if lst else 0.0

        avg_taxi_sec  = avg(taxi_sec)
        avg_mv_sec    = avg(moving_sec)
        avg_wait_sec  = avg(wait_sec)
        avg_dist      = avg(dists)
        wait_min_list = [r.wait_time_min for r in success_agents]
        avg_wait_min  = avg(wait_min_list)

        if self.snapshots:
            total_steps = self.snapshots[-1]["step"]
            total_hours = total_steps * self.time_step_sec / 3600.0
            tph = n / total_hours if total_hours > 0 else 0.0
        else:
            tph = 0.0

        inf = self.inference_latencies_ms
        rep = self.replan_latencies_ms

        wall = time.time() - self._wall_start

        # Priority group statistics (for Priority perturbation scenario)
        BASE_PRI = 1
        high = [r for r in success_agents if r.priority > BASE_PRI and r.taxi_sec is not None]
        norm = [r for r in success_agents if r.priority <= BASE_PRI and r.taxi_sec is not None]

        def avg_taxi_min(recs):
            vals = [r.taxi_sec for r in recs if r.taxi_sec is not None]
            return round(sum(vals) / len(vals) / 60.0, 2) if vals else None

        high_min = avg_taxi_min(high)
        norm_min = avg_taxi_min(norm)
        speedup  = None
        if high_min is not None and norm_min and norm_min > 0:
            speedup = round((norm_min - high_min) / norm_min * 100.0, 1)

        return {
            "algorithm": self.algorithm_name,
            "completion_rate_pct": round(rate, 2),
            "success_count": n,
            "eventual_completion_rate_pct": round(eventual_rate, 2),
            "eventual_success_count": eventual_n,
            "total_count": total,
            "collision_count": self.collision_count,
            "deadlock_count": self.deadlock_count,
            "avg_moving_min":       round(avg_mv_sec / 60.0, 2),
            "avg_onroute_wait_min": round(avg_wait_sec / 60.0, 2),
            "avg_taxi_min":         round(avg_taxi_sec / 60.0, 2),
            "avg_taxi_sec":         round(avg_taxi_sec, 2),
            "avg_delay_min":        round(avg_wait_min, 2),
            "avg_total_time_min":   round(avg_taxi_sec / 60.0 + avg_wait_min, 2),
            "avg_wait_min":         round(avg_wait_min, 2),
            "avg_path_dist":        round(avg_dist, 2),
            "throughput_per_hour":  round(tph, 2),
            "avg_inference_ms": round(avg(inf), 3) if inf else "N/A",
            "max_inference_ms": round(max(inf), 3) if inf else "N/A",
            "avg_replan_ms":    round(avg(rep), 2) if rep else "N/A",
            "max_replan_ms":    round(max(rep), 2) if rep else "N/A",
            "replan_count":     len(rep),
            # Priority group
            "high_pri_count":          len(high),
            "normal_pri_count":        len(norm),
            "high_pri_avg_taxi_min":   high_min,
            "normal_pri_avg_taxi_min": norm_min,
            "high_pri_speedup_pct":    speedup,
            "wall_time_sec": round(wall, 1),
        }

    def print_summary(self):
        s = self.get_summary()
        print("\n" + "=" * 62)
        print(f"  [{s['algorithm']}] Simulation Report")
        print("=" * 62)
        print(f"  Completion rate  : {s['completion_rate_pct']}%  ({s['success_count']}/{s['total_count']})")
        if s.get("eventual_completion_rate_pct") != s["completion_rate_pct"]:
            print(f"  Eventual complete: {s['eventual_completion_rate_pct']}%  ({s['eventual_success_count']}/{s['total_count']})")
        print(f"  Collisions       : {s['collision_count']}  |  Deadlocks: {s['deadlock_count']}")
        print(f"  Avg moving time  : {s['avg_moving_min']} min")
        print(f"  Avg onroute wait : {s['avg_onroute_wait_min']} min")
        print(f"  Avg total taxi   : {s['avg_taxi_min']} min")
        print(f"  Avg delay (rqueue): {s['avg_delay_min']} min")
        print(f"  Avg path dist    : {s['avg_path_dist']} units")
        print(f"  Throughput       : {s['throughput_per_hour']} ac/hr")
        if s["avg_inference_ms"] != "N/A":
            print(f"  Avg inference    : {s['avg_inference_ms']} ms  (max {s['max_inference_ms']} ms)")
        if s["avg_replan_ms"] != "N/A":
            print(f"  Avg replan       : {s['avg_replan_ms']} ms  (max {s['max_replan_ms']} ms, {s['replan_count']} times)")
        if s["high_pri_count"] > 0:
            print(f"  -- Priority Group Stats --")
            print(f"  High-pri agents  : {s['high_pri_count']}  avg taxi {s['high_pri_avg_taxi_min']} min")
            print(f"  Normal agents    : {s['normal_pri_count']}  avg taxi {s['normal_pri_avg_taxi_min']} min")
            if s["high_pri_speedup_pct"] is not None:
                print(f"  High-pri speedup : {s['high_pri_speedup_pct']}%  (Fig 4-X data)")
        print(f"  Wall time        : {s['wall_time_sec']} s")
        print("=" * 62)

    def export_csv(self, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        algo = self.algorithm_name.replace("+", "_").replace(" ", "_")

        summary_path = os.path.join(output_dir, f"{algo}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, ensure_ascii=False, indent=2)
        print(f"  Summary saved: {summary_path}")

        ts_path = os.path.join(output_dir, f"{algo}_timeseries.csv")
        if self.snapshots:
            with open(ts_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.snapshots[0].keys())
                writer.writeheader()
                writer.writerows(self.snapshots)
            print(f"  Timeseries saved: {ts_path}")

        agent_path = os.path.join(output_dir, f"{algo}_per_agent.csv")
        fieldnames = ["agent_id", "priority", "spawn_step", "finish_step", "taxi_steps",
                      "taxi_sec", "moving_steps", "moving_sec", "onroute_wait_sec",
                      "delay_steps", "wait_time_min", "path_dist", "collision_count",
                      "deadlock_count", "crashed", "path_nodes"]
        with open(agent_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.agents:
                writer.writerow({
                    "agent_id":        rec.agent_id,
                    "priority":        rec.priority,
                    "spawn_step":      rec.spawn_step,
                    "finish_step":     rec.finish_step,
                    "taxi_steps":      rec.taxi_steps,
                    "taxi_sec":        rec.taxi_sec,
                    "moving_steps":    rec.moving_steps,
                    "moving_sec":      round(rec.moving_sec, 2) if rec.moving_sec is not None else None,
                    "onroute_wait_sec": round(rec.onroute_wait_sec, 2) if rec.onroute_wait_sec is not None else None,
                    "delay_steps":     rec.delay_steps,
                    "wait_time_min":   round(rec.wait_time_min, 2),
                    "path_dist":       round(rec.path_dist, 3),
                    "collision_count": rec.collision_count,
                    "deadlock_count":  rec.deadlock_count,
                    "crashed":         rec.crashed,
                    "path_nodes":      str(rec.path_nodes),
                })
        print(f"  Per-agent data saved: {agent_path}")
