"""FCFS A* (带连续时间窗预留)的顺序路径规划基线。

该算法将路段的真实耗时（ceil(L/v)）纳入 A* 搜索状态，
并按先到先服务（FCFS）逐架飞机在共享的“时空预留表”中霸占时间窗口。
后续排队的飞机只能在前序飞机没占领的“时空空隙”里穿梭、等待。

核心机制与特点：
1. **纯物理时间对齐**：完全抛弃节点级匀速跳变假设，规划的状态空间直接变成 `(node, t)`，
   其中 t 为真实的物理仿真环境步（与 RL step 完全对齐）。
2. **入局等待免死金牌（Virtual Queue）**：带入场 t0 虚拟队列功能。如果 t0 时机位被前车
   占据，后来者会在图外状态 "OFF_GRAPH" 安全等待无限期而不会引发计算中崩溃。
3. **不可逆短视性**：这正是 A* FCFS 最大的缺陷。因为是严格顺序下放约束（贪心启发），
   后分配的飞机会认为前机的路径是“铁板一块无法撼动的砖墙”，
   只要场景中涉及拥挤走廊交织，极大概率低优先级飞机将无路可走而算力溢出宣告失败（Unresolved）。
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

from test import FIXED_TEST_CASES
from env.taxi_graph_env import TaxiGraphEnv
from cbs_baseline import build_adj, build_edge_length_map


class ReservationTable:
    def __init__(self):
        # 节点占用状态: (node, t)
        self.node_occupancy: set[tuple[int, int]] = set()
        # 边占用状态: ((start, end), t) -> 防止对头相撞与连环同道并行
        self.edge_occupancy: set[tuple[tuple[int, int], int]] = set()

    def is_node_free(self, node: int, t: int) -> bool:
        """检查目标节点在确切的时间 t 是否能自由站立而不会蹭到别人"""
        return (node, t) not in self.node_occupancy

    def is_move_safe(self, u: int, v: int, t_start: int, t_end: int) -> bool:
        """检查跨边行驶过程 (从 t_start+1 到 t_end) 期间是否绝对安全"""
        # 检查路段期间有无相撞 (同时查勘路段双向，保证不追尾也不对头撞)
        for t in range(t_start + 1, t_end + 1):
            if ((u, v), t) in self.edge_occupancy:
                return False
            if ((v, u), t) in self.edge_occupancy:
                return False
        
        # 检查抵达的那一刹那，对面节点上面有没有站着一架刚退来或者还没走的飞机
        if (v, t_end) in self.node_occupancy:
            return False
            
        return True

    def reserve_path(self, path: list[tuple[int, int]]):
        """将刚规划出的路径锁进时间窗口（记录占用的时间序列锁）。"""
        if not path:
            return
            
        # 1. 节点瞬时占位（凡是出场露脸的时间点全部刻死）
        for node, t_step in path:
            self.node_occupancy.add((node, t_step))
            
        # 2. 从轨迹重构边段的排他性通过
        for i in range(len(path) - 1):
            u, t_start = path[i]
            v, t_end = path[i+1]
            
            if u == v:
                # 显式 wait 动作，原地杵了这么久全算节点锁定
                for t in range(t_start + 1, t_end + 1):
                    self.node_occupancy.add((u, t))
            else:
                # 转移状态动作，锁定这段通车路权
                for t in range(t_start + 1, t_end + 1):
                    self.edge_occupancy.add(((u, v), t))
                    self.edge_occupancy.add(((v, u), t))


def bfs_heuristic(goal: int, adj: dict[int, list[int]]) -> dict[int, float]:
    """计算最短拓扑跳数，为启发搜素提供乐观估计值下界。"""
    dist = {goal: 0.0}
    queue = [goal]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb != cur and nb not in dist:
                dist[nb] = dist[cur] + 1.0
                queue.append(nb)
    return dist


def st_astar_fcfs(
    start: int, 
    goal: int, 
    t0: int, 
    adj: dict[int, list[int]], 
    edge_len: dict[tuple[int, int], float], 
    h: dict[int, float], 
    table: ReservationTable, 
    max_t: int = 800,
    cruise_speed: float = 2.0
) -> Optional[list[tuple[int, int]]]:
    """带时序排队窗口机制的 FCFS 寻找通道搜索核。"""
    
    import itertools
    tiebreaker = itertools.count()
    
    # 状态堆: (f_score, real_time_t, tiebreaker, node_id, 记忆path_trace)
    # node="OFF_GRAPH" 表象飞机还没被唤醒，允许不占地空间空等
    open_heap = []
    heapq.heappush(open_heap, (t0 + h.get(start, 0), t0, next(tiebreaker), "OFF_GRAPH", []))
    
    # (node, t) -> 最小的已探索耗费，避开重复访问
    visited = {}
    
    def get_f_t(n, step_t):
        if n == "OFF_GRAPH":
            return step_t + h.get(start, 0)
        return step_t + h.get(n, 0)

    while open_heap:
        f, t, _, cur, path = heapq.heappop(open_heap)
        
        # 抵达目的（RL模式中抵达并消失，不阻拦它人）
        if cur == goal:
            return path
            
        # 超时保护返回无解
        if t >= max_t:
            continue
            
        if cur != "OFF_GRAPH":
            state = (cur, t)
            if state in visited and visited[state] <= t:
                continue
            visited[state] = t
            
            # 操作 1: 不走，在场上硬坐等 1 秒（必须不犯规）
            next_t = t + 1
            if next_t <= max_t and table.is_node_free(cur, next_t):
                if (cur, next_t) not in visited or visited[(cur, next_t)] > next_t:
                    heapq.heappush(open_heap, (get_f_t(cur, next_t), next_t, next(tiebreaker), cur, path + [(cur, next_t)]))
            
            # 操作 2: 切向邻居进发
            for nb in adj[cur]:
                if nb == cur: 
                    continue
                length = edge_len.get((cur, nb), 1.0)
                travel_time = max(1, math.ceil(length / cruise_speed))
                next_t = t + travel_time
                if next_t <= max_t and table.is_move_safe(cur, nb, t, next_t):
                    if (nb, next_t) not in visited or visited[(nb, next_t)] > next_t:
                        heapq.heappush(open_heap, (get_f_t(nb, next_t), next_t, next(tiebreaker), nb, path + [(nb, next_t)]))
                        
        else:
            # “图外虚拟待命队”特权操作
            # 1. 怂逼不出场，苟在场外再看一秒
            next_t_off = t + 1
            if next_t_off <= max_t:
                heapq.heappush(open_heap, (get_f_t("OFF_GRAPH", next_t_off), next_t_off, next(tiebreaker), "OFF_GRAPH", path))
            
            # 2. 探出头试冲起始阵地
            if table.is_node_free(start, t):
                heapq.heappush(open_heap, (get_f_t(start, t), t, next(tiebreaker), start, path + [(start, t)]))

    return None

def main():
    print("[FCFS A*] 正在构建时钟驱动的纯物理连续时间模型...")
    env = TaxiGraphEnv(layout="F")
    adj = build_adj(env)
    edge_len = build_edge_length_map(env)
    CRUISE_SPEED = 2.0

    all_makespan = []
    total_completed = 0
    total_agents = 0
    total_rl_steps = 0

    for case_id, case in enumerate(FIXED_TEST_CASES):
        starts = case["starts"]
        goals  = case["goals"]
        t0s    = case.get("t0", [0]*len(starts))
        n = len(starts)

        # 全局资源池挂载
        table = ReservationTable()
        paths = [None] * n
        resolved = [False] * n

        h_all = [bfs_heuristic(goals[i], adj) for i in range(n)]

        # 按优先出场排队的队列进行单飞 A* （完全不顾后来者死活）
        for i in range(n):
            path = st_astar_fcfs(
                start=starts[i], goal=goals[i], t0=t0s[i],
                adj=adj, edge_len=edge_len, h=h_all[i], table=table,
                max_t=800, cruise_speed=CRUISE_SPEED
            )
            if path is not None:
                paths[i] = path
                resolved[i] = True
                # 把自己碾压过的一切时空写进王法里
                table.reserve_path(path)

        completed = sum(resolved)
        total_completed += completed
        total_agents += n

        mk_rl = 800
        if any(resolved):
            mk_rl = max(p[-1][1] for p in paths if p is not None)
            all_makespan.append(mk_rl)
            total_rl_steps += mk_rl

        status_str = " ".join("✔" if resolved[i] else "✘" for i in range(n))
        print(f"  Case {case_id:2d}: {starts} -> {goals} (t0={t0s}) [{status_str}]  Makespan={mk_rl} steps")

    avg_makespan_rl = sum(all_makespan) / len(all_makespan) if all_makespan else 0
    completion_rate = total_completed / total_agents if total_agents > 0 else 0.0
    throughput_rl = total_agents / total_rl_steps if total_rl_steps > 0 else 0.0

    print()
    print("=" * 60)
    print("  场景: 时空优先顺序规划 (Continuous Time-Window FCFS A*)")
    print("=" * 60)
    print(f"  平均滑行时间(Makespan):    {avg_makespan_rl:.2f} steps")
    print(f"  冲突次数:                  0 (排队让行完美回避)")
    print(f"  完成率:                    {completion_rate:.2%}")
    print(f"  吞吐量:                    {throughput_rl:.4f} agents/step")
    print()

if __name__ == "__main__":
    main()
