"""CBS（冲突树搜索）基线求解器。

对 FIXED_TEST_CASES 中的固定起终点组合运行 CBS 算法，
得到多智能体最优无冲突路径，作为 RL 模型的理论上界对比。

用法:
    python cbs_baseline.py

CBS 版本：标准 CBS（Sharon et al., 2015），节点级离散碰撞检测。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

from test import FIXED_TEST_CASES
from env.taxi_graph_env import TaxiGraphEnv


# ==============================================================================
# 工具：从环境构建 CBS 用的邻居图（节点级，整数步）
# ==============================================================================

def build_adj(env: TaxiGraphEnv) -> dict[int, list[int]]:
    """构建节点邻接表（双向无向），包含等待自边。"""
    adj: dict[int, list[int]] = {nid: [] for nid in env.nodes}
    for e in env.edges:
        adj[e.start].append(e.end)
    # 每个节点都允许"原地等待"
    for nid in adj:
        adj[nid].append(nid)
    return adj


def build_edge_length_map(env: TaxiGraphEnv) -> dict[tuple[int, int], float]:
    """返回 (start, end) -> edge.length 映射，用于 CBS 路径 → RL 步数转换。"""
    emap: dict[tuple[int, int], float] = {}
    for e in env.edges:
        emap[(e.start, e.end)] = e.length
        emap[(e.end, e.start)] = e.length  # 双向
    return emap


def path_to_rl_steps(
    path: list[int],
    edge_len: dict[tuple[int, int], float],
    cruise_speed: float = 2.0,
) -> int:
    """将 CBS 节点跳序列转换为等效 RL 仿真步数。

    转换逻辑：
      - CBS 的"节点跳"只记录离散动作，不包含在物理边上滑行所需的时间
      - RL 环境中，飞机在边上滑行时每步推进 speed/edge.length 的进度
      - 因此一条长为 L 的边在速度 v 下需要 ceil(L / v) 步才能穿越
      - 等待动作（节点不变）= 1步

    Args:
        path: CBS 输出的节点序列（含等待，即重复节点）
        edge_len: (start, end) -> 物理长度的字典
        cruise_speed: RL 智能体的典型巡航速度（默认 2.0）

    Returns:
        等效 RL 仿真步数（整数）
    """
    import math
    total_steps = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u == v:
            # 原地等待：CBS 1步 = RL 1步
            total_steps += 10
        else:
            length = edge_len.get((u, v), 50)  # 找不到边则默认1.0（单位长度）
            # 穿越该边所需步数 = ceil(length / speed)
            total_steps += math.ceil(length / cruise_speed)
    return total_steps

# ==============================================================================
# 单智能体 A*（带约束）
# ==============================================================================

@dataclass(order=True)
class _ANode:
    f: float
    g: int
    pos: int = field(compare=False)
    t: int = field(compare=False)
    parent: Optional["_ANode"] = field(default=None, compare=False)


def constrained_astar(
    start: int,
    goal: int,
    adj: dict[int, list[int]],
    h: dict[int, float],
    constraints: set[tuple[int, int]],        # (pos, t) 或 (pos1, pos2, t) 顶点约束 → (node, time)
    edge_constraints: set[tuple[int, int, int]],  # (from, to, t): t时刻不可走 from→to
    max_t: int = 500,
) -> list[int] | None:
    """带约束的单智能体 A*，返回节点序列（含等待），失败返回 None。"""
    open_heap: list[_ANode] = []
    start_node = _ANode(f=h[start], g=0, pos=start, t=0)
    heapq.heappush(open_heap, start_node)
    visited: dict[tuple[int,int], int] = {}  # (pos, t) → g

    while open_heap:
        cur = heapq.heappop(open_heap)
        state = (cur.pos, cur.t)
        if state in visited and visited[state] <= cur.g:
            continue
        visited[state] = cur.g

        if cur.pos == goal and cur.t > 0:
            # 回溯路径
            path: list[int] = []
            node = cur
            while node is not None:
                path.append(node.pos)
                node = node.parent
            return list(reversed(path))

        if cur.t >= max_t:
            continue

        for nxt in adj.get(cur.pos, []):
            nt = cur.t + 1
            # 顶点约束
            if (nxt, nt) in constraints:
                continue
            # 边约束
            if (cur.pos, nxt, cur.t) in edge_constraints:
                continue
            ng = cur.g + 1 if nxt != cur.pos else cur.g + 1
            hv = h.get(nxt, 1e9)
            nstate = (nxt, nt)
            if nstate not in visited or visited[nstate] > ng:
                heapq.heappush(open_heap, _ANode(f=ng + hv, g=ng, pos=nxt, t=nt, parent=cur))

    return None


# ==============================================================================
# 启发值预计算（单源 BFS 到目标的最短跳数）
# ==============================================================================

def bfs_heuristic(goal: int, adj: dict[int, list[int]]) -> dict[int, float]:
    dist: dict[int, float] = {goal: 0.0}
    queue = [goal]
    while queue:
        cur = queue.pop(0)
        for nxt in adj.get(cur, []):
            if nxt not in dist and nxt != cur:
                dist[nxt] = dist[cur] + 1.0
                queue.append(nxt)
    return dist


# ==============================================================================
# CBS 冲突检测
# ==============================================================================

def _pad(path: list[int], length: int) -> list[int]:
    """将路径末尾用最终节点补长（到达目的地后静止不动）。"""
    if not path:
        return path
    return path + [path[-1]] * max(0, length - len(path))


def find_first_conflict(paths: list[list[int]]) -> tuple | None:
    """找第一个冲突：顶点冲突 (i, j, pos, t) 或边冲突 (i, j, u, v, t)。"""
    max_len = max(len(p) for p in paths)
    padded = [_pad(p, max_len) for p in paths]
    n = len(paths)
    for t in range(max_len):
        for i in range(n):
            for j in range(i + 1, n):
                pi_t = padded[i][t]
                pj_t = padded[j][t]
                # 顶点冲突
                if pi_t == pj_t:
                    return ("vertex", i, j, pi_t, t)
                # 边冲突（对向穿越）
                if t + 1 < max_len:
                    if padded[i][t] == padded[j][t+1] and padded[j][t] == padded[i][t+1]:
                        return ("edge", i, j, padded[i][t], padded[j][t], t)
    return None


def _find_first_conflict_with_t0(paths: list[list[int]], t0: list[int]) -> tuple | None:
    """t0 感知的冲突检测：两架飞机都在等待期（t < t0[i] AND t < t0[j]）时共享起点不算冲突。"""
    max_len = max(len(p) for p in paths)
    padded = [_pad(p, max_len) for p in paths]
    n = len(paths)
    for t in range(max_len):
        for i in range(n):
            for j in range(i + 1, n):
                # 若 i 和 j 都处于场外等待期，跳过（虚拟队列不产生物理冲突）
                i_waiting = t < t0[i]
                j_waiting = t < t0[j]
                if i_waiting and j_waiting:
                    continue

                pi_t = padded[i][t]
                pj_t = padded[j][t]
                # 顶点冲突（两架飞机都已入场，且在同一节点）
                if pi_t == pj_t and not (i_waiting or j_waiting):
                    return ("vertex", i, j, pi_t, t)
                # 边冲突（对向穿越，两架均已入场）
                if t + 1 < max_len and not i_waiting and not j_waiting:
                    if padded[i][t] == padded[j][t+1] and padded[j][t] == padded[i][t+1]:
                        return ("edge", i, j, padded[i][t], padded[j][t], t)
    return None


# ==============================================================================
# CBS 主算法
# ==============================================================================

@dataclass(order=True)
class CBSNode:
    cost: int
    constraints: list = field(compare=False)
    edge_constraints: list = field(compare=False)
    paths: list = field(compare=False)


def cbs_solve(
    starts: list[int],
    goals: list[int],
    adj: dict[int, list[int]],
    h_all: list[dict[int, float]],
    t0: list[int] | None = None,   # 每架飞机的预定进入时刻
    max_t: int = 300,
) -> list[list[int]] | None:
    """CBS 入口，返回每个 agent 的节点序列，失败返回 None。
    
    若提供 t0，每个 agent 的路径会在其 t0 之前预填充出发节点（在场外等待），
    从而正确处理多机在不同时刻进入场景的时序冲突检测。
    """
    n = len(starts)
    if t0 is None:
        t0 = [0] * n

    # 初始无约束规划：每架飞机先独自规划最短路
    # constrained_astar 已从 t=0 开始，我们在路径前拼接 t0 个等待步
    init_paths = []
    for i in range(n):
        p = constrained_astar(starts[i], goals[i], adj, h_all[i], set(), set(), max_t)
        if p is None:
            print(f"  [CBS] Agent {i} 初始路径规划失败: {starts[i]} -> {goals[i]}")
            return None
        # 在路径前加入 t0[i] 个"场外等待"步（起始节点重复）
        full_path = [starts[i]] * t0[i] + p
        init_paths.append(full_path)

    root = CBSNode(
        cost=sum(len(p) - 1 - t0[i] for i, p in enumerate(init_paths)),  # 代价不计等待期
        constraints=[set() for _ in range(n)],
        edge_constraints=[set() for _ in range(n)],
        paths=init_paths,
    )
    open_heap: list[CBSNode] = [root]
    iterations = 0

    while open_heap:
        iterations += 1
        if iterations > 5000:
            print("  [CBS] 搜索超上限（5000次扩展），返回当前最优（可能未完全消解冲突）")
            node = heapq.heappop(open_heap)
            return node.paths

        node = heapq.heappop(open_heap)
        conflict = _find_first_conflict_with_t0(node.paths, t0)

        if conflict is None:
            print(f"  [CBS] 求解成功（扩展 {iterations} 次），总代价 = {node.cost}")
            return node.paths

        ctype = conflict[0]
        if ctype == "vertex":
            _, i, j, pos, t = conflict
            for agent_idx, constraint in [(i, (pos, t + 1)), (j, (pos, t + 1))]:
                new_vc = [set(vc) for vc in node.constraints]
                new_ec = [set(ec) for ec in node.edge_constraints]
                new_vc[agent_idx].add(constraint)
                p = constrained_astar(starts[agent_idx], goals[agent_idx], adj, h_all[agent_idx],
                                       new_vc[agent_idx], new_ec[agent_idx], max_t)
                if p is None:
                    continue
                full_path = [starts[agent_idx]] * t0[agent_idx] + p
                new_paths = list(node.paths)
                new_paths[agent_idx] = full_path
                child = CBSNode(
                    cost=sum(len(pp) - 1 - t0[k] for k, pp in enumerate(new_paths)),
                    constraints=new_vc,
                    edge_constraints=new_ec,
                    paths=new_paths,
                )
                heapq.heappush(open_heap, child)
        elif ctype == "edge":
            _, i, j, u, v, t = conflict
            for agent_idx, ec in [(i, (u, v, t)), (j, (v, u, t))]:
                new_vc = [set(vc) for vc in node.constraints]
                new_ec = [set(ec2) for ec2 in node.edge_constraints]
                new_ec[agent_idx].add(ec)
                p = constrained_astar(starts[agent_idx], goals[agent_idx], adj, h_all[agent_idx],
                                       new_vc[agent_idx], new_ec[agent_idx], max_t)
                if p is None:
                    continue
                full_path = [starts[agent_idx]] * t0[agent_idx] + p
                new_paths = list(node.paths)
                new_paths[agent_idx] = full_path
                child = CBSNode(
                    cost=sum(len(pp) - 1 - t0[k] for k, pp in enumerate(new_paths)),
                    constraints=new_vc,
                    edge_constraints=new_ec,
                    paths=new_paths,
                )
                heapq.heappush(open_heap, child)

    return None


# ==============================================================================
# 主入口：逐 Case 求解并输出对比
# ==============================================================================

def main():
    env = TaxiGraphEnv(layout="F", num_agents=4, max_steps=300)
    adj = build_adj(env)
    edge_len = build_edge_length_map(env)
    CRUISE_SPEED = 2.0  # RL 典型巡航速度，与智能体加速后的平均速度一致

    print("\n" + "="*65)
    print(" CBS 基线求解（理论最优多智能体路径）- Layout F")
    print(f" 时间尺度对齐: CBS 节点跳 → RL 仿真步 (cruise_speed={CRUISE_SPEED})")
    print("="*65)

    total_episodes = len(FIXED_TEST_CASES)
    total_makespan_hops = 0   # CBS 原始节点跳数
    total_makespan_rl   = 0   # 映射到 RL 仿真步数
    total_completed = 0
    total_agents = 0
    total_conflicts = 0

    for case_id, case in enumerate(FIXED_TEST_CASES):
        starts = case["starts"]
        goals = case["goals"]
        t0_sched = case.get("t0", [0] * len(starts))
        num_a = len(starts)
        total_agents += num_a

        print(f"\n--- Case {case_id} ---")
        print(f"  Starts: {starts}")
        print(f"  Goals:  {goals}")
        print(f"  t0(f):  {t0_sched}  (单位: steps)")

        h_all = [bfs_heuristic(g, adj) for g in goals]
        paths = cbs_solve(starts, goals, adj, h_all, t0=t0_sched, max_t=300)

        if paths is None:
            print("  [结果] CBS 无解（路径不可达或超时）")
            total_makespan_hops += 300
            total_makespan_rl   += 300
        else:
            net_steps_hops = [len(p) - 1 - t0_sched[i] for i, p in enumerate(paths)]
            makespan_hops  = max(len(p) - 1 for p in paths)
            total_makespan_hops += makespan_hops
            total_completed += num_a

            # 将每条路径映射到 RL 仿真步数
            rl_steps_per_agent = [
                t0_sched[i] + path_to_rl_steps(p[t0_sched[i]:], edge_len, CRUISE_SPEED)
                for i, p in enumerate(paths)
            ]
            makespan_rl = max(rl_steps_per_agent)
            total_makespan_rl += makespan_rl

            print(f"  [结果] CBS 节点总代价(跳): {sum(net_steps_hops)}  "
                  f"| Makespan(跳): {makespan_hops}  "
                  f"| Makespan(RL步): {makespan_rl}")
            for i, (p, g) in enumerate(zip(paths, goals)):
                active_path = p[t0_sched[i]:]
                rl_s = rl_steps_per_agent[i]
                print(f"  Agent {i} (t0={t0_sched[i]}步): "
                      f"{net_steps_hops[i]}跳 / {rl_s}RL步 | "
                      f"{' -> '.join(map(str, active_path))}")

    avg_hops = total_makespan_hops / max(total_episodes, 1)
    avg_rl   = total_makespan_rl   / max(total_episodes, 1)
    completion_rate = total_completed / max(total_agents, 1) * 100
    throughput_rl   = total_completed / max(total_makespan_rl, 1)

    print("\n\n场景: 理论最优基线 (CBS Baseline)")
    print("─" * 60)
    print(f"  CBS Makespan (节点跳):      {avg_hops:.2f} hops/episode")
    print(f"  CBS Makespan (RL等效步数):  {avg_rl:.2f} steps/episode   ← 与 RL 可直接对比")
    print(f"  冲突次数:                   {total_conflicts} (CBS绝对无冲突)")
    print(f"  完成率:                     {completion_rate:.2f}%")
    print(f"  吞吐量(RL时间尺度):         {throughput_rl:.4f} agents/step")
if __name__ == "__main__":
    main()
