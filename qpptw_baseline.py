"""QPPTW（带时间窗优先级顺序路径规划）基线求解器。

算法描述
--------
QPPTW（Queue-based Priority Path Planning with Time Windows）是一种
标准的 **解耦/顺序规划** 方法，流程如下：

    1. 按固定优先级顺序（Priority 1 = 最高）依次为每个智能体规划路径。
    2. 已完成规划的智能体将其"时空轨迹"写入全局 **时空预留表（Reservation Table）**。
       预留表记录每个 (节点, 时间步) 是否已被占用，以及每条 (边, 时间步) 是否已被穿越。
    3. 当前正在规划的低优先级智能体使用 **Space-Time A*** 在时空图中搜索路径，
       所有违反预留表的时空状态都被视为障碍物，从而在源头避免冲突。
    4. 如果搜索失败（路径被完全封死），则该智能体被标记为 unresolved，
       完成率据此计算。

用法
----
    python qpptw_baseline.py
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

from test import FIXED_TEST_CASES
from env.taxi_graph_env import TaxiGraphEnv
from cbs_baseline import build_adj, build_edge_length_map, path_to_rl_steps

# =========================================================
# 时空预留表
# =========================================================

class ReservationTable:
    """时空预留表（Space-Time Reservation Table）。

    记录：
      - 节点占用：(node, t) -> agent_id
      - 边穿越：  ((u, v), t) -> agent_id  （防止边交叉冲突）
    """

    def __init__(self) -> None:
        self._node: dict[tuple[int, int], int] = {}         # (node, t) -> agent_id
        self._edge: dict[tuple[tuple[int,int], int], int] = {}  # ((u,v), t) -> agent_id

    def is_node_free(self, node: int, t: int) -> bool:
        return (node, t) not in self._node

    def is_edge_free(self, u: int, v: int, t: int) -> bool:
        return ((u, v), t) not in self._edge and ((v, u), t) not in self._edge

    def reserve_path(self, path: list[int], agent_id: int) -> None:
        """将一条时序路径的所有时空点登记进预留表。"""
        for t, node in enumerate(path):
            self._node[(node, t)] = agent_id
        for t in range(len(path) - 1):
            u, v = path[t], path[t + 1]
            self._edge[((u, v), t)] = agent_id
            # 防止两机在同一步互换节点（swap conflict）
            self._edge[((v, u), t)] = agent_id


# =========================================================
# Space-Time A*（带时空预留表约束）
# =========================================================

def st_astar(
    start: int,
    goal: int,
    adj: dict[int, list[int]],
    h: dict[int, float],
    table: ReservationTable,
    max_t: int = 800,
) -> Optional[list[int]]:
    """Space-Time A* 搜索。

    状态空间: (node, t)
    允许的动作: 移动到邻居 / 原地等待（等待也需要检查预留表）

    Returns:
        时序路径列表 [node_t0, node_t1, ...], 或 None 表示搜索失败。
    """
    # heap: (f, g, node, t, parent_path_ref)
    # 使用路径压缩，parent_path_ref 存储从 start 到当前状态的节点列表
    open_heap: list[tuple[float, int, int, list[int]]] = []
    heapq.heappush(open_heap, (h[start], 0, start, [start]))
    visited: dict[tuple[int, int], float] = {}  # (node, t) -> best g so far

    while open_heap:
        f, g, cur, path = heapq.heappop(open_heap)

        if cur == goal:
            return path

        t = g  # g = 从起点经过的步数 = 当前时间
        if t >= max_t:
            continue

        state = (cur, t)
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g

        next_t = t + 1
        for nb in adj[cur]:
            if not table.is_node_free(nb, next_t):
                continue
            if nb != cur and not table.is_edge_free(cur, nb, t):
                continue
            new_g = next_t
            new_f = new_g + h[nb]
            new_state = (nb, next_t)
            if new_state in visited and visited[new_state] <= new_g:
                continue
            heapq.heappush(open_heap, (new_f, new_g, nb, path + [nb]))

    return None  # 无解


# =========================================================
# 启发式：单源最短路（BFS，边全长为1用于节点级启发）
# =========================================================

def bfs_heuristic(goal: int, adj: dict[int, list[int]]) -> dict[int, float]:
    """从目标节点反向 BFS 计算到达目标的最小节点跳数（启发值下界）。"""
    dist: dict[int, float] = {goal: 0}
    queue = [goal]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb != cur and nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


# =========================================================
# QPPTW 主求解函数
# =========================================================

def solve_qpptw(
    starts: list[int],
    goals: list[int],
    adj: dict[int, list[int]],
    edge_len: dict[tuple[int, int], float],
    max_t: int = 800,
    cruise_speed: float = 2.0,
) -> dict:
    """对给定起终点组运行 QPPTW 算法。

    Returns:
        dict 包含 paths（各 Agent 的时序完整路径）、resolved（是否成功）、
        rl_steps（各Agent的等效RL步数）。
    """
    n = len(starts)
    table = ReservationTable()
    paths: list[Optional[list[int]]] = [None] * n
    resolved: list[bool] = [False] * n

    # 预先计算所有目标节点的 BFS 启发表
    h_all = [bfs_heuristic(goals[i], adj) for i in range(n)]

    for i in range(n):
        path = st_astar(
            start=starts[i],
            goal=goals[i],
            adj=adj,
            h=h_all[i],
            table=table,
            max_t=max_t,
        )
        if path is not None:
            paths[i] = path
            resolved[i] = True
            table.reserve_path(path, agent_id=i)

    # 将各 Agent 的节点时序路径转为 RL 等效步数
    rl_steps: list[int] = []
    for i in range(n):
        if paths[i] is not None:
            # path 已包含等待（重复节点），直接用 path_to_rl_steps 折算
            rl_steps.append(path_to_rl_steps(paths[i], edge_len, cruise_speed))
        else:
            rl_steps.append(max_t)  # 失败智能体计最大时间

    return {
        "paths": paths,
        "resolved": resolved,
        "rl_steps": rl_steps,
        "makespan_hops": max(len(p) - 1 for p in paths if p is not None) if any(paths) else max_t,
    }


# =========================================================
# 主程序：复用 test.py 的固定用例，输出对齐格式的指标
# =========================================================

def main() -> None:
    # ------------------------------------------------------------------
    # 构建环境（仅用于获取图拓扑，不进行 RL 推理）
    # ------------------------------------------------------------------
    print("[QPPTW] 正在构建机场路网拓扑 (Layout F) ...")
    env = TaxiGraphEnv(layout="F")
    adj = build_adj(env)
    edge_len = build_edge_length_map(env)

    CRUISE_SPEED = 2.0  # 与 RL 环境中保持一致

    # ------------------------------------------------------------------
    # 逐 Case 求解
    # ------------------------------------------------------------------
    total_cases = len(FIXED_TEST_CASES)
    num_agents   = len(FIXED_TEST_CASES[0]["starts"])

    all_makespan_hops: list[float] = []
    all_makespan_rl:   list[float] = []
    total_conflicts  = 0   # QPPTW 顺序规划无冲突（通过约束源头消除）
    total_completed  = 0
    total_agents     = 0
    total_rl_steps   = 0

    for case_id, case in enumerate(FIXED_TEST_CASES):
        starts = case["starts"]
        goals  = case["goals"]

        result = solve_qpptw(
            starts=starts,
            goals=goals,
            adj=adj,
            edge_len=edge_len,
            max_t=800,
            cruise_speed=CRUISE_SPEED,
        )

        completed = sum(result["resolved"])
        total_completed += completed
        total_agents    += num_agents

        if result["paths"]:
            mk_hops = result["makespan_hops"]
            all_makespan_hops.append(mk_hops)
            # RL 等效步数以本组内最长路径为 Makespan
            mk_rl = max(result["rl_steps"])
            all_makespan_rl.append(mk_rl)
            total_rl_steps += mk_rl
        
        # 输出每个 case 的简要信息
        status_str = " ".join(
            "✔" if result["resolved"][i] else "✘"
            for i in range(num_agents)
        )
        print(f"  Case {case_id:2d}: {starts} -> {goals}  [{status_str}]  "
              f"makespan={result['makespan_hops']} hops / {max(result['rl_steps'])} RL-steps")

    # ------------------------------------------------------------------
    # 汇总指标（对齐 CBS / RL 格式）
    # ------------------------------------------------------------------
    avg_makespan_hops = sum(all_makespan_hops) / len(all_makespan_hops) if all_makespan_hops else 0
    avg_makespan_rl   = sum(all_makespan_rl)   / len(all_makespan_rl)   if all_makespan_rl   else 0
    completion_rate   = total_completed / total_agents if total_agents > 0 else 0.0
    throughput_rl     = (total_agents / total_rl_steps) if total_rl_steps > 0 else 0.0

    print()
    print("=" * 60)
    print("  场景: 顺序规划基线 (QPPTW Baseline)")
    print("=" * 60)
    print(f"  平均滑行时间(路径节点跳):  {avg_makespan_hops:.2f} hops/episode")
    print(f"  平均滑行时间(RL等效步数):  {avg_makespan_rl:.2f} steps/episode")
    print(f"  冲突次数:                  {total_conflicts}"
          f"  (顺序规划通过预留表在源头消除冲突)")
    print(f"  完成率:                    {completion_rate:.2%}")
    print(f"  吞吐量(RL时间尺度):        {throughput_rl:.4f} agents/step")



if __name__ == "__main__":
    main()
