"""飞机智能体模块。

管理单个智能体的状态、观测构造和动作掩码逻辑。

观测向量设计（三段拼接）：
  Part1: 邻居槽位 max_neighbors × 6
         [path_cost, on_planned_path, node_occupied, edge_occupied, head_on_risk, downstream_capacity_risk]
  Part2: 自身状态 8
         [speed, on_edge, edge_progress, dist_to_goal,
          steps_remaining_ratio, path_progress_ratio, priority_norm, is_at_gate]
  Part3: 他者感知 (num_agents-1) × 5
         [dist_to_me, their_speed, threatens_my_next, is_done, intent_overlap]

  obs_dim = max_neighbors*6 + 8 + (num_agents-1)*5
          = 5*6 + 8 + 3*5 = 53 （默认参数下）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env.taxi_graph_env import TaxiGraphEnv


class AircraftAgent:
    """飞机智能体。

    核心设计：路径规划负责导航，强化学习负责冲突协调（调速/等待/滑入滑出决策）。
    """

    ACTION_DIM = 8  # 0-4 选邻接节点, 5 减速, 6 加速, 7 保持

    def __init__(
        self,
        agent_id: int,
        planned_path: list[int],
        num_agents: int = 4,
        max_speed: float = 3.0,
        min_speed: float = 0.0,
        speed_step: float = 0.5,
        max_neighbors: int = 5,
        auto_speed: bool = True,
        use_astar_guide: bool = True,
        prevent_occupied_target_entry: bool = False,
        enforce_reference_progress: bool = False,
        clearance_aware_reference_progress: bool = False,
        pre_entry_clearance_guard: bool = False,
    ) -> None:
        """初始化智能体。

        Args:
            agent_id: 智能体 ID。
            planned_path: 由 A* 生成的先验路径。
            num_agents: 环境中智能体总数（影响 obs_dim）。
            max_speed: 最大速度。
            min_speed: 最小速度。
            speed_step: 速度调节步长。
            max_neighbors: 最大邻居数量（观测维度）。
            auto_speed: 是否开启环境自动接管巡航。
            use_astar_guide: 是否开启 A* 先验路径引导。
                设为 False 时，观测向量中的 path_cost_norm 和
                on_planned_path 两个特征将被置零（尺寸不变）。
                用于 [Ablation Study] MAPPO-NoGuide 消融实验。
        """
        self.agent_id = agent_id
        self.planned_path = planned_path
        self.planned_path_set: set[int] = set(planned_path)  # O(1) 归属查询
        self.num_agents = num_agents
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_step = speed_step
        self.max_neighbors = max_neighbors
        self.auto_speed = auto_speed
        self.use_astar_guide = use_astar_guide
        self.prevent_occupied_target_entry = prevent_occupied_target_entry
        self.enforce_reference_progress = enforce_reference_progress
        self.clearance_aware_reference_progress = clearance_aware_reference_progress
        self.pre_entry_clearance_guard = pre_entry_clearance_guard
        self.path_index = 0
        self.reference_speed = 1.0

    @property
    def obs_dim(self) -> int:
        """精简后的观测向量维度。
        Part1: 邻居槽位 max_neighbors * 7
        Part2: 自身状态 6
        总计: 5*7 + 6 = 41
        """
        return self.max_neighbors * 7 + 6

    def get_observation(self, env: "TaxiGraphEnv", all_agents: list["AircraftAgent"] | None = None) -> np.ndarray:
        """精简的高效观测向量设计（Egocentric）。
        
        Part 1 每邻居槽位 (max_neighbors × 7):
          [0] is_valid           槽位有效标志 (1/0)
          [1] path_cost_norm     经由此邻接节点到达目标的归一化最短距离
          [2] on_planned_path    该邻居是否为 A* 先验路径的下一步 (1/0)
          [3] immediate_risk     物理占用碰撞风险 (由连线占用、节点占用、迎面驶来构成) 1/0
          [4] corridor_locked    走廊死锁状态感知 (走廊处于对方占用方向，进入必锁死) 1/0
          [5] downstream_occ     下游 2-hop 节点是否有飞机 (1/0)
          [6] corridor_load      目标走廊的拥挤度 (归一化)

        Part 2 自身状态 (6):
          [0] speed              当前速度归一化 [0, 1]
          [1] on_edge            是否正在连线上行驶 (物理边) 1/0
          [2] dist_to_goal_norm  当前到目标的归一化最短距离
          [3] steps_ratio        剩余可用步数比例 (紧迫度) [0, 1]
          [4] priority_norm      当前优先级得分分布
          [5] is_at_gate         是否身处机位（死胡同尽头保护区）1/0

        Args:
            env: TaxiGraphEnv 实例
            all_agents: 保留供接口兼容，此精简版本不再迭代全局他者以解决排序重排和多余维度问题

        Returns:
            观测向量 shape=(obs_dim,)
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        offset = 0

        i = self.agent_id
        pos = env.agent_positions[i]
        goal = env.agent_goals[i]
        neighbors = env.get_neighbors(pos)

        # 预计算下一个规划节点
        next_planned_idx = self.path_index + 1
        next_planned_node = (
            self.planned_path[next_planned_idx]
            if next_planned_idx < len(self.planned_path)
            else -1
        )
        
        # 归一化常量 (Layout F的曼哈顿最大视距约在 100~150 左右)
        MAX_DIST = 100.0

        # ============================================================
        # Part 1：邻居槽位状况 (max_neighbors × 7)
        # ============================================================
        for k in range(self.max_neighbors):
            if k < len(neighbors):
                nid = neighbors[k]
                
                # [0] is_valid
                obs[offset + 0] = 1.0
                
                # [1] path_cost_norm
                edge_idx = env.get_edge_between(pos, nid)
                edge_len = env.edges[edge_idx].length if edge_idx is not None else 0.0
                sp_dist = env.get_shortest_path_distance(nid, goal)
                actual_cost = (edge_len + max(sp_dist, 0.0)) if sp_dist >= 0 else 0.0
                # [Ablation] MAPPO-NoGuide: 屏蔽 A* 路径代价先验
                obs[offset + 1] = min(actual_cost / MAX_DIST, 1.0) if self.use_astar_guide else 0.0

                # [2] on_planned_path
                # [Ablation] MAPPO-NoGuide: 屏蔽 A* 路径方向引导
                obs[offset + 2] = (1.0 if nid == next_planned_node else 0.0) if self.use_astar_guide else 0.0
                
                # [3] immediate_risk (物理占用风险：节点停靠、连线行驶、对向相接)
                node_occ = any(
                    j != i and not env.agent_done[j] and getattr(env, 'start_delays', [0]*env.num_agents)[j] == 0
                    and env.agent_positions[j] == nid and not env.agent_on_edge[j]
                    for j in range(env.num_agents)
                )
                edge_occ = env.edges[edge_idx].occupied if edge_idx is not None else False
                head_on = any(
                    j != i and env.agent_on_edge[j] and getattr(env, 'edge_idx', [None]*env.num_agents)[j] is not None
                    and env.edges[env.edge_idx[j]].start == nid and env.edges[env.edge_idx[j]].end == pos
                    for j in range(env.num_agents)
                )
                clearance_blocked = (
                    getattr(env, "clearance_risk_observation", False)
                    and env.is_node_clearance_blocked(nid, i)
                )
                obs[offset + 3] = 1.0 if (node_occ or edge_occ or head_on or clearance_blocked) else 0.0
                
                # [4] corridor_locked
                c_locked = 0.0
                corridor = env.corridor_mgr.get_corridor(pos, nid)
                if corridor is not None:
                    entry_dir = corridor.edge_direction(pos, nid)
                    if not env.corridor_mgr.can_enter(corridor, entry_dir, agent_id=self.agent_id):
                        c_locked = 1.0
                # --- [新增] 2-hop 观测与全局 Markov 状态 ---
                # [5] 下游状态探测 (2-hop)
                downstream_neighbors = env.get_neighbors(nid)
                downstream_occupied = 0.0
                for dn in downstream_neighbors:
                    if dn == pos:
                        continue
                    for j in range(env.num_agents):
                        if (j != i and not env.agent_done[j]
                                and getattr(env, 'start_delays', [0] * env.num_agents)[j] == 0
                                and env.agent_positions[j] == dn and not env.agent_on_edge[j]):
                            downstream_occupied = 1.0
                            break
                obs[offset + 5] = downstream_occupied

                # [6] 走廊拥挤度 (Corridor Load)
                c_load = 0.0
                if corridor is not None:
                    c_load = len(corridor.occupied_by) / max(env.num_agents, 1)
                obs[offset + 6] = c_load

            else:
                # 填充无效槽位
                obs[offset + 0] = 0.0
                obs[offset + 1] = 1.0 # 满代价作为惩罚填充
                obs[offset + 2] = 0.0
                obs[offset + 3] = 0.0
                obs[offset + 4] = 0.0
                obs[offset + 5] = 0.0
                obs[offset + 6] = 0.0
                
            offset += 7

        # ============================================================
        # Part 2：自身状态（6）
        # ============================================================
        # [0] 当前速度
        obs[offset + 0] = env.agent_speeds[i] / max(self.max_speed, 1.0)
        
        # [1] 是否在边上行进
        on_edge = env.agent_on_edge[i]
        obs[offset + 1] = 1.0 if on_edge else 0.0
        
        # [2] 当前到目标绝对归一化距离
        my_dist_goal = env.get_shortest_path_distance(pos, goal)
        obs[offset + 2] = min(max(my_dist_goal, 0.0) / MAX_DIST, 1.0)
        
        # [3] 紧迫度倒计时比例
        obs[offset + 3] = max(env.max_steps - env.step_count, 0) / env.max_steps
        
        # [4] 相对优先级
        # Scenario-level priority boosts may use a large raw value (e.g.
        # 1001) for deterministic right-of-way.  Keep the policy feature in
        # the training support [0, 1] while the environment still uses the
        # unbounded raw value for its ordering rules.
        obs[offset + 4] = min(max(env.priority[i], 0), max(env.num_agents - 1, 1)) / max(env.num_agents - 1, 1)
        
        # [5] 是否身处尽头机位
        obs[offset + 5] = 1.0 if len(env.adj[pos]) == 1 else 0.0
        
        offset += 6

        return obs

    def get_action_mask(self, env: "TaxiGraphEnv", all_agents: list["AircraftAgent"] | None = None) -> np.ndarray:
        """生成动作掩码。

        规则：
        - 在边上：仅允许速度控制动作
        - 在走廊内部节点上：允许所有方向+速度控制，但禁止 WAIT 和导致 V=0 的减速
          （Min-Speed Clamp：确保走廊内持续移动）
        - 在普通节点上：允许选邻接节点 + 速度控制
          - 额外检查廊道预留状态：若廊道已被对向占用，屏蔽该动作
        - 速度边界：v=0 禁止减速，v=max 禁止加速

        Returns:
            掩码向量 shape=(8,)，1 表示可执行。
        """
        mask = np.zeros(self.ACTION_DIM, dtype=np.float32)

        if env.agent_on_edge[self.agent_id]:
            if self.auto_speed:
                mask[5] = 0.0
                mask[6] = 0.0
                mask[7] = 1.0  # 环境自动控制巡航
            else:
                mask[5] = 1.0
                mask[6] = 1.0
                mask[7] = 1.0
        else:
            pos = env.agent_positions[self.agent_id]
            neighbors = env.get_neighbors(pos)
            inside_corridor = env.corridor_mgr.is_inside_corridor(pos)
            came_from = getattr(env, 'came_from', [None]*env.num_agents)[self.agent_id]
            current_is_gate = getattr(env.nodes[pos].type, "name", "") == "GATE"

            for idx in range(min(len(neighbors), 5)):
                nxt = neighbors[idx]
                
                # 用户要求：禁止航空器掉头来回摇摆
                # 除非是在度数为 1 的死胡同（机位）必须掉头，否则绝不能回退到刚才所在的节点
                # A U-turn is normally undesirable, but it is the only
                # recovery action after entering a degree-one terminal node.
                # Masking it there creates a permanent dead-end.
                if nxt == came_from and len(env.adj[pos]) != 1:
                    # Preserve the anti-oscillation rule during ordinary
                    # motion, but never mask a recovery move explicitly
                    # requested by the current A* reference. This occurs
                    # after a replan from a corridor end or stand branch.
                    reference_node = self.get_next_planned_node(env)
                    is_reference_recovery = (
                        self.use_astar_guide and nxt == reference_node
                    )
                    if not is_reference_recovery:
                        mask[idx] = 0.0
                        continue

                # 【修复】检查通向该邻居的边是否已被物理关闭（Closure 扰动）
                # adj 格式为 [(neighbor_node, edge_idx), ...]
                _eidx_to_nxt = next(
                    (eidx for nb, eidx in env.adj.get(pos, []) if nb == nxt), None
                )
                if _eidx_to_nxt is not None and env.edges[_eidx_to_nxt].closed:
                    mask[idx] = 0.0
                    continue

                # 用户要求：机位安全出站（强制延误逻辑）

                # 检查通往干道的支线节点（Stub Node）是否已被占据
                if current_is_gate:
                    nxt_occupied = [j for j in range(env.num_agents) 
                                   if j != self.agent_id and not env.agent_done[j] 
                                   and getattr(env, 'start_delays', [0]*env.num_agents)[j] <= 0
                                   and env.agent_positions[j] == nxt and not env.agent_on_edge[j]]
                    if len(nxt_occupied) > 0:
                        # 支线出口被占，必须在 Gate 里 WAIT
                        mask[idx] = 0.0
                        continue
                
                target_is_gate = getattr(env.nodes[nxt].type, "name", "") == "GATE"
                is_my_goal = (nxt == env.agent_goals[self.agent_id])
                # Gate nodes and their connection stubs are task-specific.
                # Entering a foreign stand branch cannot advance any mission
                # and previously could leave an aircraft without a legal exit.
                target_stub_gate = nxt - 2000
                is_stand_stub = (
                    nxt >= 2000
                    and target_stub_gate in env.nodes
                    and getattr(env.nodes[target_stub_gate].type, "name", "") == "GATE"
                )
                leaving_own_gate = current_is_gate and is_stand_stub and target_stub_gate == pos
                if target_is_gate and not is_my_goal:
                    mask[idx] = 0.0
                    continue
                if is_stand_stub and not (target_stub_gate == env.agent_goals[self.agent_id] or leaving_own_gate):
                    mask[idx] = 0.0
                    continue
                # Boundary leaves are valid only as an aircraft's own entry
                # or exit endpoint; entering another one is a nonrecoverable
                # detour in a directed task graph.
                target_is_boundary_leaf = (
                    getattr(env.nodes[nxt].type, "name", "") == "INTERSECTION"
                    and len(env.adj[nxt]) == 1
                )
                if target_is_boundary_leaf and not is_my_goal:
                    mask[idx] = 0.0
                    continue

                target_gate_privilege = (target_is_gate and is_my_goal)

                # Do not admit an aircraft into an edge whose terminal node is
                # already occupied. Otherwise TaxiGraphEnv stops it near the
                # destination (progress=0.99), turning a node queue into an
                # edge blockage that the policy cannot correct under
                # automatic speed control.
                nxt_occupied_by = [
                    j for j in range(env.num_agents)
                    if j != self.agent_id and not env.agent_done[j]
                    and getattr(env, 'start_delays', [0] * env.num_agents)[j] <= 0
                    and env.agent_positions[j] == nxt and not env.agent_on_edge[j]
                ]
                if self.prevent_occupied_target_entry and nxt_occupied_by:
                    mask[idx] = 0.0
                    continue

                # The low-level clearance shield otherwise allows entry and
                # brakes the aircraft near the downstream node. Masking that
                # transition earlier preserves the edge for through traffic.
                if (
                    self.pre_entry_clearance_guard
                    and env.is_node_entry_blocked(nxt, self.agent_id)
                ):
                    mask[idx] = 0.0
                    continue

                corridor = env.corridor_mgr.get_corridor(pos, nxt)
                if corridor is not None:
                    entry_dir = corridor.edge_direction(pos, nxt)
                    if not env.corridor_mgr.can_enter(corridor, entry_dir, agent_id=self.agent_id):
                        if not target_gate_privilege:
                            mask[idx] = 0.0
                            continue
                
                # 防堆积出口封锁 (Intersection Output Blocking) - Swap 防止
                if target_gate_privilege:
                    # 去机位点不考虑干道 swap
                    mask[idx] = 1.0
                    continue

                # 新增核心：物理边沿绝对锁定 (Edge Occupancy Lock)
                # 防止“幽灵飞机”漏洞（对方上路后在节点处变为隐身，导致自己盲目冲入发生正面对撞）
                edge_occupied = False
                for j in range(env.num_agents):
                    if j != self.agent_id and env.agent_on_edge[j] and not env.agent_done[j]:
                        ej_idx = getattr(env, 'edge_idx', [None]*env.num_agents)[j]
                        if ej_idx is not None:
                            ej = env.edges[ej_idx]
                            # 仅阻挡对向开来的飞机 (Head-on Risk)，允许同向尾随 (Same direction)
                            # 因为 TaxiGraphEnv._move 里有完善的 ahead_prog 防追尾机制
                            if ej.start == nxt and ej.end == pos:
                                edge_occupied = True
                                break
                
                if edge_occupied:
                    mask[idx] = 0.0
                    continue

                if self.use_astar_guide and len(nxt_occupied_by) > 0 and all_agents is not None:
                    swap_risk = False
                    for j in nxt_occupied_by:
                        j_next = all_agents[j].get_next_planned_node(env)
                        if j_next == pos: 
                            # 优先级破局机制
                            # 只有我优先级低于对方，或者优先级相等但我 ID 更大时，才执行让行锁死动作
                            my_pri = env.priority[self.agent_id]
                            their_pri = env.priority[j]
                            if my_pri < their_pri or (my_pri == their_pri and self.agent_id > j):
                                swap_risk = True
                                break
                    
                    if swap_risk:
                        mask[idx] = 0.0
                        continue

                # ── 翼尖净空阴影检查（Wingtip Clearance Gate Shadow Zone）─────
                # 物理依据：ICAO Annex 14 Table 3-1
                # 当飞机位于机位节点（待推出状态），且翼尖阴影组内有其他飞机
                # 正处于推出阶段（在 gate_node 或 stub_node），则屏蔽出站动作，
                # 强制等待相邻飞机推出完成、翼尖退出干涉区域后方可启动推出。
                # ─────────────────────────────────────────────────────────────
                i_am_at_gate = current_is_gate
                if i_am_at_gate:
                    shadow_peers = getattr(env, 'gate_shadow_groups', {}).get(pos, set())
                    if shadow_peers:
                        shadow_blocked = False
                        for j in range(env.num_agents):
                            if (j == self.agent_id or env.agent_done[j]
                                    or getattr(env, 'start_delays', [0] * env.num_agents)[j] > 0):
                                continue
                            j_pos = env.agent_positions[j]
                            j_at_gate = (j_pos < 2000 and len(env.adj[j_pos]) == 1)
                            j_at_stub  = (j_pos >= 2000)
                            if j_at_gate or j_at_stub:
                                j_gate_id = j_pos if j_at_gate else (j_pos - 2000)
                                if j_gate_id in shadow_peers:
                                    shadow_blocked = True
                                    break
                        if shadow_blocked:
                            mask[idx] = 0.0
                            continue
                # ─────────────────────────────────────────────────────────────

                mask[idx] = 1.0

            if self.auto_speed:
                # 屏蔽人工速度控制，交由底层发车：
                mask[5] = 0.0
                mask[6] = 0.0
            else:
                mask[5] = 1.0
                mask[6] = 1.0

            if inside_corridor:
                mask[7] = 0.0  # 走廊内禁止 WAIT
            else:
                mask[7] = 1.0

            # A waiting action is useful when the reference direction is
            # blocked. It is not useful when the current A* successor has
            # already passed all feasibility checks above. Mask it only in
            # that clear-reference state, leaving alternative route choices
            # available to the policy.
            if self.enforce_reference_progress and mask[7] > 0:
                reference_node = self.get_next_planned_node(env)
                for action_id, (next_node, _) in enumerate(env.adj[pos][:5]):
                    clearance_blocked = (
                        self.clearance_aware_reference_progress
                        and env.is_node_clearance_blocked(next_node, self.agent_id)
                    )
                    if (
                        next_node == reference_node
                        and mask[action_id] > 0
                        and not clearance_blocked
                    ):
                        mask[7] = 0.0
                        break

        if not self.auto_speed:
            # 速度边界约束
            speed = env.agent_speeds[self.agent_id]
            if speed <= self.min_speed:
                mask[5] = 0.0
            if speed >= self.max_speed:
                mask[6] = 0.0

            # 走廊内防止静止排队时间过长导致死锁，额外检查
            if not env.agent_on_edge[self.agent_id] and env.corridor_mgr.is_inside_corridor(env.agent_positions[self.agent_id]):
                if speed <= 0.5:
                    mask[5] = 0.0

        if mask.sum() == 0:
            mask[7] = 1.0
        return mask


    def replan_path(
        self,
        env: "TaxiGraphEnv",
        planner: "Any",
        graph: dict[int, list[tuple[int, float]]],
        blocked_nodes: set[int] | None = None
    ) -> None:
        """动态 A* 路径重规划。
        
        用于在遇到持久死锁时，绕开前方拥堵节点。
        """
        start = env.agent_positions[self.agent_id]
        goal = env.agent_goals[self.agent_id]
        new_path = planner.plan_path(start, goal, graph, blocked_nodes=blocked_nodes)
        
        if new_path:
            self.planned_path = new_path
            self.planned_path_set = set(new_path)
            self.path_index = 0

    def get_next_planned_node(self, env: "TaxiGraphEnv") -> int | None:
        """获取先验路径中的下一个节点。

        Args:
            env: 图环境实例。

        Returns:
            下一个节点 ID，如果已到终点返回 None。
        """
        pos = env.agent_positions[self.agent_id]
        # 同步 path_index
        if self.path_index < len(self.planned_path):
            if self.planned_path[self.path_index] != pos:
                for idx, nid in enumerate(self.planned_path):
                    if nid == pos:
                        self.path_index = idx
                        break

        next_idx = self.path_index + 1
        if next_idx < len(self.planned_path):
            return self.planned_path[next_idx]
        return None

    def sync_path_index(self, cur_pos: int) -> None:
        """根据当前实际位置同步路径索引（替代盲目递增的 advance_path）。

        策略：
        1. 优先检查 path_index+1 是否匹配（最常见情况，O(1)）。
        2. 若不匹配，向后扫描以处理跳步情况。
        3. 若当前位置不在规划路径上，path_index 保持不变，
           不再盲目漂移，路径导引信号维持有效。

        Args:
            cur_pos: 智能体当前所在节点 ID。
        """
        # 1. 检查紧接的下一节点（最常见）
        next_idx = self.path_index + 1
        if next_idx < len(self.planned_path) and self.planned_path[next_idx] == cur_pos:
            self.path_index = next_idx
            return
        # 2. 向后扫描（跳步/fast-forward）：不允许回退，从 next_idx+1 开始
        for idx in range(next_idx + 1, len(self.planned_path)):
            if self.planned_path[idx] == cur_pos:
                self.path_index = idx
                return
        # 3. 当前位置不在规划路径上 → 保持 path_index 不变，避免漂移
