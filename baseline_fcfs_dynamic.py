"""
FCFS + 纯空间 A* 基线（Pure-Spatial 版）
==========================================
核心设计原则（修复自我锁死 & 时序漂移 Bug）：
  - A* 只规划节点序列 [100, 110, 101, ...]，不带时间戳
  - 彻底去除时空预留表（ReservationTable）
  - 物理引擎（TaxiGraphEnv）负责所有时序：加速、刹车、让行
  - FCFS 只负责路权顺序：优先级高的飞机先规划、先选路
  - 执行器：飞机到节点 → 查节点序列 → 发方向动作，简单可靠
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import time
from collections import deque

from env.marl_env import MARLEnv
from configs.config import Config
from perturbation_scenarios import PerturbationScenario, ScenarioInjector
from cbs_baseline import build_adj, build_edge_length_map
from simulate_continuous import load_full_flight_schedule


# ================================================================
# 纯空间 A*（无时间维度）
# ================================================================
def bfs_heuristic_lazy(goal: int, adj: dict) -> dict[int, float]:
    dist = {goal: 0.0}
    q = deque([goal])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1.0
                q.append(nb)
    return dist


def spatial_astar(
    start: int,
    goal: int,
    adj: dict,
    edge_len: dict,
    h: dict[int, float],
    soft_blocked: set[int] | None = None,
    soft_penalty: float = 50.0,
    hard_blocked_edges: set[tuple[int, int]] | None = None,
) -> list[int] | None:
    """纯空间 A*，返回节点序列。

    Args:
        soft_blocked:       软阻塞节点集合，增加绕行惩罚代价但不禁止通行。
        soft_penalty:       软阻塞节点的绕路惩罚代价（默认 50.0）。
        hard_blocked_edges: 硬约束边集合 {(u, v), ...}，这些边完全不可通行（用于滑行道关闭场景）。
    """
    soft_blocked = soft_blocked or set()
    hard_blocked_edges = hard_blocked_edges or set()
    tb = itertools.count()
    # (f, g, tie, node, path)
    heap = [(h.get(start, 0), 0.0, next(tb), start, [start])]
    best_g: dict[int, float] = {}

    while heap:
        f, g, _, cur, path = heapq.heappop(heap)
        if cur == goal:
            return path
        if best_g.get(cur, float("inf")) <= g:
            continue
        best_g[cur] = g
        for nb in adj.get(cur, []):
            if nb == cur:
                continue
            # 硬约束：关闭边完全不可通行，直接跳过
            if (cur, nb) in hard_blocked_edges:
                continue
            cost = edge_len.get((cur, nb), 1.0) + (soft_penalty if nb in soft_blocked else 0.0)
            ng = g + cost
            if ng < best_g.get(nb, float("inf")):
                heapq.heappush(heap, (ng + h.get(nb, 0), ng, next(tb), nb, path + [nb]))

    # 若硬约束导致无路，放宽硬约束重试（保证不完全卡死）
    if hard_blocked_edges:
        return spatial_astar(start, goal, adj, edge_len, h,
                             soft_blocked=soft_blocked, soft_penalty=soft_penalty)
    # 若软阻塞导致无路，放宽软阻塞重试
    if soft_blocked:
        return spatial_astar(start, goal, adj, edge_len, h,
                             hard_blocked_edges=hard_blocked_edges)
    return None



# ================================================================
# 主评估函数
# ================================================================
def run_dynamic_fcfs_baseline(args):
    config = Config.from_yaml(args.config)
    flights = load_full_flight_schedule(
        args.excel, time_scale=args.time_scale,
        transit_ratio=args.transit_ratio,
        start_hour=8.0, end_hour=24.0,
    )
    if not flights:
        print("未找到有效航班数据。")
        return

    num_total = len(flights)
    starts = [f["start"] for f in flights]
    goals  = [f["goal"]  for f in flights]
    t0s    = [f["t0"]    for f in flights]

    config.env.num_agents = num_total
    env = MARLEnv(config)
    adj      = build_adj(env.graph_env)
    edge_len = build_edge_length_map(env.graph_env)

    # 惰性启发值缓存（按目标节点）
    h_cache: dict[int, dict] = {}
    def get_h(g: int) -> dict:
        if g not in h_cache:
            h_cache[g] = bfs_heuristic_lazy(g, adj)
        return h_cache[g]

    env.reset(starts=starts, goals=goals, delays=t0s)

    scenario_enum = PerturbationScenario(args.scenario)
    injector = ScenarioInjector(scenario_enum, env.graph_env, num_total,
                                total_steps=args.max_steps, seed=42)

    # 追踪变量
    agent_spawn_step  = [None] * num_total
    agent_finish_step = [None] * num_total
    agent_paths: dict[int, list] = {i: [] for i in range(num_total)}

    # 纯节点序列路径 & 游标
    planned_routes: dict[int, list[int]] = {i: [] for i in range(num_total)}
    route_cursor:   dict[int, int]       = {i: 0  for i in range(num_total)}

    total_replan_ms = 0.0
    replan_count    = 0
    replan_needed   = True
    replan_cd       = 0

    agent_last_pos  = list(env.graph_env.pos)
    agent_stall_cnt = [0] * num_total
    deadlock_count  = 0
    # 延误追踪：每架飞机因拥塞被推迟的累计仿真步数
    delay_log       = [0] * num_total   # 单位：仿真步
    DELAY_STEPS     = max(1, int(60.0 / args.time_scale))  # 60 物理秒
    MAX_DELAYS      = 20   # 超过此次数仍无法出发才真正放弃
    delay_count     = [0] * num_total   # 每架飞机已被延误的次数
    total_steps_executed = args.max_steps

    print(f"\n[FCFS-Spatial] 场景={args.scenario}  共 {num_total} 架次")
    wall_start = time.time()

    step = 0
    while step < args.max_steps:
        # 扰动注入
        prev_log = len(injector.event_log)
        injector.step(step)
        if len(injector.event_log) > prev_log:
            replan_needed = True

        # 活跃列表
        active_agents = []
        for i in range(num_total):
            if agent_spawn_step[i] is None and env.graph_env.start_delays[i] == 0 and not env.graph_env.done[i]:
                agent_spawn_step[i] = step
            if agent_finish_step[i] is None and env.graph_env.done[i]:
                agent_finish_step[i] = step
            if not env.graph_env.done[i] and env.graph_env.start_delays[i] == 0:
                active_agents.append(i)
                agent_paths[i].append(env.graph_env.pos[i])

        if all(env.graph_env.done):
            total_steps_executed = step
            break

        # 快进
        # jump>0：手动跳步并跳过 env.step()（这些步里不需要物理更新）
        # jump==0（min_delay==1）：必须调用 env.step()，让环境把 start_delays 从 1 减到 0
        if not active_agents:
            pending = [env.graph_env.start_delays[i]
                       for i in range(num_total)
                       if not env.graph_env.done[i] and env.graph_env.start_delays[i] > 0]
            if pending:
                jump = min(pending) - 1
                if jump > 0:
                    for i in range(num_total):
                        if not env.graph_env.done[i]:
                            env.graph_env.start_delays[i] = max(0, env.graph_env.start_delays[i] - jump)
                    step += jump
                    total_steps_executed = step
                    step += 1
                    continue
            # jump==0 或无待激活飞机：放行到 env.step() 让环境自然递减 start_delays

        # 死锁检测
        if replan_cd > 0:
            replan_cd -= 1

        for i in list(active_agents):
            if env.graph_env.on_edge[i]:
                agent_stall_cnt[i] = 0
                continue
            cur = env.graph_env.pos[i]
            if cur == agent_last_pos[i]:
                agent_stall_cnt[i] += 1
            else:
                agent_stall_cnt[i] = 0
                agent_last_pos[i] = cur
                # 推进游标
                route = planned_routes[i]
                c = route_cursor[i]
                while c < len(route) - 1 and route[c] != cur:
                    c += 1
                route_cursor[i] = c

            if agent_stall_cnt[i] == 30 and replan_cd == 0:
                replan_needed = True
                replan_cd = 60

            if agent_stall_cnt[i] >= args.deadlock_timeout:
                # ── 不抛弃航班：推迟 60s 重入队列 ──
                if delay_count[i] < MAX_DELAYS:
                    env.graph_env.start_delays[i] = DELAY_STEPS
                    # 飞机留在原地（物理引擎会处理），重置路线等待重规划
                    planned_routes[i] = []
                    route_cursor[i]   = 0
                    delay_log[i]     += DELAY_STEPS
                    delay_count[i]   += 1
                    agent_stall_cnt[i] = 0
                    active_agents.remove(i)
                    replan_needed = True   # 触发一次重规划，绕开拥塞点
                else:
                    # 极端情况：延误超过 MAX_DELAYS 次，记录为死锁移除
                    env.graph_env.crashed[i] = True
                    env.graph_env.done[i]    = True
                    active_agents.remove(i)
                    deadlock_count += 1
                    agent_stall_cnt[i] = 0

        # ── 纯空间 A* 规划 ──
        if replan_needed:
            t_rp = time.perf_counter()

            # 关闭边对应的软阻塞节点
            closed_nodes: set[int] = set()
            for e in env.graph_env.edges:
                if e.closed:
                    closed_nodes.add(e.start)
                    closed_nodes.add(e.end)

            unfinished = sorted(
                [i for i in range(num_total) if not env.graph_env.done[i]],
                key=lambda i: env.graph_env.priority[i],
            )

            occupied: set[int] = set()
            for i in unfinished:
                s = starts[i] if env.graph_env.start_delays[i] > 0 else env.graph_env.pos[i]
                g = goals[i]
                if s == g:
                    planned_routes[i] = [s]
                    route_cursor[i]   = 0
                    continue

                route = spatial_astar(s, g, adj, edge_len, get_h(g),
                                      soft_blocked=closed_nodes | occupied)
                if route:
                    planned_routes[i] = route
                    route_cursor[i]   = 0
                    occupied.update(route[1:-1])

            rp_ms = (time.perf_counter() - t_rp) * 1000
            total_replan_ms += rp_ms
            replan_count    += 1
            replan_needed    = False
            if args.verbose:
                print(f"  [Replan] step={step} {rp_ms:.1f}ms")

        # ── 动作提取（纯节点序列）──
        actions = [7] * num_total
        for i in active_agents:
            if env.graph_env.on_edge[i]:
                continue  # 在边上由物理引擎接管
            if env.graph_env.delayed[i] and env.graph_env.delay_timers[i] > 0:
                continue  # 被扰动延迟，等待

            route = planned_routes[i]
            if not route:
                continue

            cur_pos = env.graph_env.pos[i]
            c = route_cursor[i]
            # 对齐游标
            while c < len(route) - 1 and route[c] != cur_pos:
                c += 1
            route_cursor[i] = c

            if c >= len(route) - 1:
                continue  # 已到终点

            next_node = route[c + 1]
            for a_idx, edge_info in enumerate(env.graph_env.adj[cur_pos]):
                nb = edge_info[0] if isinstance(edge_info, tuple) else edge_info
                if nb == next_node:
                    actions[i] = a_idx
                    break

        if step % 500 == 0:
            sc = sum(1 for i in range(num_total)
                     if env.graph_env.done[i] and not env.graph_env.crashed[i])
            print(f"  Step {step:<6} | 活跃:{len(active_agents):<3} | "
                  f"成功:{sc:<3}/{num_total} | 死锁:{deadlock_count}")

        env.step(actions)
        step += 1

    # ── 汇总 ──
    success_cnt = sum(1 for i in range(num_total)
                      if env.graph_env.done[i] and not env.graph_env.crashed[i])
    taxi_times, path_dists = [], []
    for i in range(num_total):
        if env.graph_env.done[i] and not env.graph_env.crashed[i]:
            sp, fi = agent_spawn_step[i], agent_finish_step[i]
            if sp is not None and fi is not None:
                taxi_times.append(fi - sp)
            path = agent_paths[i]
            if path:
                nd = [path[0]]
                for n in path[1:]:
                    if n != nd[-1]: nd.append(n)
                path_dists.append(sum(edge_len.get((nd[k], nd[k+1]), 0) for k in range(len(nd)-1)))

    avg_taxi  = sum(taxi_times) / len(taxi_times) if taxi_times else 0
    avg_dist  = sum(path_dists) / len(path_dists) if path_dists else 0
    avg_rp_ms = total_replan_ms / replan_count if replan_count else 0
    throughput = success_cnt / total_steps_executed if total_steps_executed else 0
    delayed_flights = sum(1 for d in delay_log if d > 0)
    avg_delay_sec   = (sum(delay_log) / delayed_flights * args.time_scale) if delayed_flights else 0

    print("\n" + "=" * 60)
    print("  FCFS+A*（纯空间版）仿真性能报告")
    print("=" * 60)
    print(f"  成功引导: {success_cnt}/{num_total} ({success_cnt/num_total*100:.1f}%)")
    print(f"  死锁移除: {deadlock_count}")
    print(f"  延误飞机: {delayed_flights} 架，平均额外延误 {avg_delay_sec:.0f}s")
    print(f"  平均滑行: {avg_taxi:.1f} steps = {avg_taxi*args.time_scale:.0f}s")
    print(f"  平均距离: {avg_dist:.1f}")
    print(f"  吞吐率:   {throughput:.4f} agents/step")
    print(f"  重规划:   {replan_count}次  均值{avg_rp_ms:.1f}ms")
    print(f"  总步长:   {total_steps_executed}")
    print(f"  实际耗时: {time.time()-wall_start:.1f}s")
    print("=" * 60)
    injector.summary()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FCFS+A* 纯空间路径基线")
    p.add_argument("--config",            default="configs/improved_v2.yaml")
    p.add_argument("--excel",             default="data.xlsx")
    p.add_argument("--max-steps",   type=int,   default=30000)
    p.add_argument("--time-scale",  type=float, default=3.0)
    p.add_argument("--transit-ratio",type=float, default=0.35)
    p.add_argument("--deadlock-timeout", type=int, default=300)
    p.add_argument("--scenario",    default="baseline",
                   choices=["baseline", "delay", "closure", "priority"])
    p.add_argument("--verbose",     action="store_true")
    run_dynamic_fcfs_baseline(p.parse_args())
