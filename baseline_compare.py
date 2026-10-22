"""基线算法对比脚本：将真实航班数据（data.xlsx）对接到 FCFS A* 与 CBS 基线。

运行方式:
    python baseline_compare.py --excel data.xlsx
    python baseline_compare.py --excel data.xlsx --cbs-window 08:00-09:00
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Optional

import pandas as pd

# 引入环境与基线工具
from env.taxi_graph_env import TaxiGraphEnv
from cbs_baseline import (
    build_adj, build_edge_length_map, bfs_heuristic,
    cbs_solve, path_to_rl_steps
)
from fcfs_astar_baseline import ReservationTable, bfs_heuristic as fcfs_bfs, st_astar_fcfs
from data_loader import STAND_TO_NODE_MAP


def _apply_scenario(env, scenario: str) -> None:
    """根据场景名称配置环境参数。"""
    g = env
    if scenario == "normal":
        g.delay_prob      = 0.00
        g.close_prob      = 0.00
        g.priority_prob   = 0.00
    elif scenario == "delay":
        g.delay_prob      = 0.02
        g.close_prob      = 0.00
        g.priority_prob   = 0.01
    elif scenario == "closure":
        g.delay_prob      = 0.00
        g.close_prob      = 0.05
        g.priority_prob   = 0.01
    elif scenario == "priority":
        g.delay_prob      = 0.01
        g.close_prob      = 0.01
        g.priority_prob   = 0.15


# ============================================================
# 1. 数据加载（复用 simulate_continuous 的逻辑）
# ============================================================

def load_flights(filepath: str, time_scale: float = 5.0) -> list[dict]:
    """从 Excel 加载并过滤航班计划，返回标准化任务列表。"""
    df = pd.read_excel(filepath)

    def parse_hm(t_val):
        if pd.isna(t_val): return None, None
        t_str = str(t_val).strip()
        if hasattr(t_val, 'hour'):
            h, m = t_val.hour, t_val.minute
        else:
            match = re.search(r'(\d+):(\d+)', t_str)
            if not match: return None, None
            h, m = int(match.group(1)), int(match.group(2))
        return h * 3600 + m * 60, f"{h:02d}:{m:02d}"

    flights = []
    for _, row in df.iterrows():
        gate_str = str(row.get("机位", "")).strip()
        m_gate = re.search(r'\d+', gate_str)
        if not m_gate or m_gate.group() not in STAND_TO_NODE_MAP:
            continue
        gate_num = m_gate.group()
        gate_node = STAND_TO_NODE_MAP[gate_num]
        airline = str(row.get("航空公司", "")).strip()
        f_type = str(row.get("进出", ""))

        if "进" in f_type:
            t_sec, t_str = parse_hm(row.get("实际到达")) or parse_hm(row.get("计划到达"))
            if t_sec:
                flights.append({"time_sec": t_sec, "time_str": t_str,
                                 "start": 100, "goal": gate_node,
                                 "type": "ARR", "airline": airline, "gate": gate_num})
        if "出" in f_type:
            t_sec, t_str = parse_hm(row.get("实际起飞")) or parse_hm(row.get("计划起飞"))
            if t_sec:
                flights.append({"time_sec": t_sec, "time_str": t_str,
                                 "start": gate_node, "goal": 105,
                                 "type": "DEP", "airline": airline, "gate": gate_num})

    if not flights: return []
    flights.sort(key=lambda x: x["time_sec"])

    # 剔除头部孤立航班（间隔 > 1 小时）
    GAP = 3600
    while len(flights) > 1 and (flights[1]["time_sec"] - flights[0]["time_sec"]) > GAP:
        flights.pop(0)

    base_t = flights[0]["time_sec"]
    for f in flights:
        f["t0"] = int(max(0, f["time_sec"] - base_t) / time_scale)
    return flights


def filter_window(flights: list[dict], window_str: str) -> list[dict]:
    """按时间窗口（格式 'HH:MM-HH:MM'）筛选航班子集。"""
    m = re.match(r'(\d+):(\d+)-(\d+):(\d+)', window_str)
    if not m:
        raise ValueError(f"时间窗口格式错误: {window_str}，应为 'HH:MM-HH:MM'")
    h1, m1, h2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    t_start = h1 * 3600 + m1 * 60
    t_end = h2 * 3600 + m2 * 60
    subset = [f for f in flights if t_start <= f["time_sec"] <= t_end]
    if not subset: return subset
    base_t = subset[0]["time_sec"]
    for f in subset:
        f["t0_local"] = int(max(0, f["time_sec"] - base_t) / 5.0)
    return subset


# ============================================================
# 2. FCFS A* 全量运行
# ============================================================

def run_fcfs(flights: list[dict], env: TaxiGraphEnv) -> dict:
    """对全量航班运行 FCFS A*，返回 KPI 字典。"""
    adj = build_adj(env)
    edge_len = build_edge_length_map(env)
    table = ReservationTable()

    n = len(flights)
    starts = [f["start"] for f in flights]
    goals  = [f["goal"]  for f in flights]
    t0s    = [f["t0"]    for f in flights]

    print(f"\n[FCFS A*] 正在规划 {n} 架次（按 t0 优先级顺序）...")
    t_begin = time.time()

    paths = []
    resolved = []
    taxi_times = []

    for i in range(n):
        # max_t 须能覆盖 t0 + 最长滑行（留 500 步余量）
        max_t_i = t0s[i] + 600
        h = fcfs_bfs(goals[i], adj)
        path = st_astar_fcfs(
            start=starts[i], goal=goals[i], t0=t0s[i],
            adj=adj, edge_len=edge_len, h=h,
            table=table, max_t=max_t_i, cruise_speed=2.0
        )
        resolved.append(path is not None)
        paths.append(path)
        if path is not None:
            table.reserve_path(path)
            entry_t = path[0][1] if path else t0s[i]
            finish_t = path[-1][1] if path else t0s[i]
            taxi_times.append(finish_t - entry_t)

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{n} | 已解决: {sum(resolved)}")

    elapsed = time.time() - t_begin

    success = sum(resolved)
    avg_taxi = sum(taxi_times) / len(taxi_times) if taxi_times else 0
    total_steps = max((p[-1][1] for p in paths if p), default=0)
    throughput = success / total_steps if total_steps > 0 else 0

    # 平均实际物理距离（相邻边权重累加）
    path_dists = []
    for p in paths:
        if p:
            nodes = [p[0][0]]
            for node, _ in p[1:]:
                if node != nodes[-1]:
                    nodes.append(node)
            dist = sum(edge_len.get((nodes[k], nodes[k+1]), 0) for k in range(len(nodes)-1))
            path_dists.append(dist)
    avg_path_dist = sum(path_dists) / len(path_dists) if path_dists else 0

    print(f"[FCFS A*] 规划完成，耗时 {elapsed:.1f}s")
    return {
        "算法": "FCFS A*",
        "总任务数": n,
        "成功数": success,
        "碰撞次数": 0,
        "完成率": f"{success/n:.2%}",
        "平均滑行时间(steps)": round(avg_taxi, 2),
        "平均滑行时间(秒)": round(avg_taxi * 5, 1),
        "平均滑行距离(边权和)": round(avg_path_dist, 1),
        "系统吞吐率": round(throughput, 4),
        "总仿真时长(steps)": total_steps,
    }


# ============================================================
# 3. CBS 子集运行
# ============================================================

def run_cbs(flights: list[dict], env: TaxiGraphEnv, window_str: str) -> Optional[dict]:
    """在指定时间窗口内的航班子集上运行 CBS。"""
    subset = filter_window(flights, window_str)
    if not subset:
        print(f"[CBS] 时间窗口 {window_str} 内无有效航班。")
        return None

    n = len(subset)
    print(f"\n[CBS] 时间窗口 {window_str}，共 {n} 架次...")

    if n > 35:
        print(f"[CBS] 子集规模 {n} > 35，CBS 求解可能超时。仅取前 30 架次。")
        subset = subset[:30]
        n = 30

    adj = build_adj(env)
    edge_len = build_edge_length_map(env)
    starts = [f["start"] for f in subset]
    goals  = [f["goal"]  for f in subset]
    t0s    = [f.get("t0_local", 0) for f in subset]

    h_all = [bfs_heuristic(g, adj) for g in goals]

    t_begin = time.time()
    paths = cbs_solve(starts, goals, adj, h_all, t0=t0s, max_t=600)
    elapsed = time.time() - t_begin
    print(f"[CBS] 求解耗时 {elapsed:.1f}s")

    if paths is None:
        print("[CBS] 求解失败（超时或无解）。")
        return {
            "算法": f"CBS ({window_str}子集)",
            "总任务数": n, "成功数": 0, "碰撞次数": "N/A",
            "完成率": "0.00%", "平均滑行时间(steps)": "N/A",
            "平均滑行时间(秒)": "N/A", "系统吞吐率": "N/A", "总仿真时长(steps)": "N/A"
        }

    taxi_times = []
    for i, p in enumerate(paths):
        active = p[t0s[i]:]
        rl_s = path_to_rl_steps(active, edge_len, cruise_speed=2.0)
        taxi_times.append(rl_s)

    avg_taxi = sum(taxi_times) / len(taxi_times) if taxi_times else 0
    total_steps = max(t0s[i] + taxi_times[i] for i in range(n))
    throughput = n / total_steps if total_steps > 0 else 0

    # CBS path 是 node ID 列表，计算边权重累加距离
    cbs_path_dists = []
    for p in paths:
        if p:
            nodes = [p[0]]
            for node in p[1:]:
                if node != nodes[-1]:
                    nodes.append(node)
            dist = sum(edge_len.get((nodes[k], nodes[k+1]), 0) for k in range(len(nodes)-1))
            cbs_path_dists.append(dist)
    avg_path_dist = sum(cbs_path_dists) / len(cbs_path_dists) if cbs_path_dists else 0

    return {
        "算法": f"CBS ({window_str}子集 {n}架次)",
        "总任务数": n,
        "成功数": n,
        "碰撞次数": 0,
        "完成率": "100.00%",
        "平均滑行时间(steps)": round(avg_taxi, 2),
        "平均滑行时间(秒)": round(avg_taxi * 5, 1),
        "平均滑行距离(边权和)": round(avg_path_dist, 1),
        "系统吞吐率": round(throughput, 4),
        "总仿真时长(steps)": total_steps,
    }


# ============================================================
# 4. 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="基线对比：FCFS A* + CBS vs MARL")
    parser.add_argument("--excel", type=str, default="data.xlsx")
    parser.add_argument("--cbs-window", type=str, default="12:00-13:00",
                        help="CBS 测试时间窗口，格式 'HH:MM-HH:MM'")
    parser.add_argument("--result-dir", type=str, default="results")
    # MARL 参考结果（直接从 simulate_continuous 的输出粘贴）
    parser.add_argument("--marl-success", type=int, default=259)
    parser.add_argument("--marl-total", type=int, default=276)
    parser.add_argument("--marl-conflicts", type=int, default=3)
    parser.add_argument("--marl-avg-taxi", type=float, default=36.03)
    parser.add_argument("--marl-avg-dist", type=float, default=64.1)
    parser.add_argument("--marl-throughput", type=float, default=0.0084)
    parser.add_argument("--marl-steps", type=int, default=30000)
    parser.add_argument("--scenario", type=str, default="normal",
                        choices=["normal", "delay", "closure", "priority"],
                        help="测试场景，目前主要用于同步 MARL 的测试口径")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # 加载数据
    print(f"正在加载: {args.excel}")
    flights = load_flights(args.excel)
    if not flights:
        print("未找到有效航班数据！")
        sys.exit(1)
    print(f"共加载 {len(flights)} 个任务")

    # 初始化环境图
    env = TaxiGraphEnv(layout="F", num_agents=4, max_steps=100)
    _apply_scenario(env, args.scenario)
    print(f"✓ 已同步基线对比场景口径: {args.scenario}")

    results = []

    # MARL 参考结果
    marl_n = args.marl_total
    results.append({
        "算法": "MARL (4.20.pt)",
        "总任务数": marl_n,
        "成功数": args.marl_success,
        "碰撞次数": args.marl_conflicts,
        "完成率": f"{args.marl_success/marl_n:.2%}",
        "平均滑行时间(steps)": args.marl_avg_taxi,
        "平均滑行距离(边权和)": args.marl_avg_dist,
        "平均滑行时间(秒)": round(args.marl_avg_taxi * 5, 1),
        "系统吞吐率": args.marl_throughput,
        "总仿真时长(steps)": args.marl_steps,
    })

    # FCFS A* 全量
    fcfs_result = run_fcfs(flights, env)
    results.append(fcfs_result)

    # CBS 子集
    cbs_result = run_cbs(flights, env, args.cbs_window)
    if cbs_result:
        results.append(cbs_result)

    # 打印对比表
    df = pd.DataFrame(results)
    print("\n\n" + "="*75)
    print("  三方算法性能对比报告".center(72))
    print("="*75)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.ambiguous_as_wide", True)
    print(df.to_string(index=False))
    print("="*75)

    # 保存
    save_path = os.path.join(args.result_dir, "baseline_comparison.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n对比结果已保存至: {save_path}")


if __name__ == "__main__":
    main()
