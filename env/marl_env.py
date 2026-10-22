"""多智能体环境接口（MARLEnv Wrapper）。

将 TaxiGraphEnv 包装为标准的多智能体强化学习接口，
负责观测构造、奖励计算、动作掩码等。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.aircraft_agent import AircraftAgent
from configs.config import Config, EnvConfig, RewardConfig
from env.taxi_graph_env import TaxiGraphEnv
from planner.astar import AStarPlanner


# Critic-only state schema. Actor observations and action dimensions remain
# unchanged, so an existing actor checkpoint can be used for fine-tuning.
CRITIC_STATE_SCHEMA = "topology_semantic_v2"
CRITIC_AGENT_FEATURES = 15
CRITIC_GLOBAL_FEATURES = 4


class MARLEnv:
    """多智能体强化学习环境包装器。

    整合 TaxiGraphEnv、A* 规划器和 AircraftAgent。
    提供标准的 reset / step 接口。
    """

    def __init__(self, config: Config | None = None) -> None:
        """初始化多智能体环境。

        Args:
            config: 项目配置。为 None 时使用默认配置。
        """
        if config is None:
            config = Config()
        self.config = config
        self.env_cfg: EnvConfig = config.env
        self.reward_cfg: RewardConfig = config.reward
        # Training-only mechanism; corridor protection remains in TaxiGraphEnv.
        self.enable_deadlock_recovery = bool(
            getattr(self.env_cfg, "enable_deadlock_recovery", True)
        )

        self.graph_env = TaxiGraphEnv(
            num_agents=self.env_cfg.num_agents,
            max_steps=self.env_cfg.max_steps,
            close_prob=self.env_cfg.taxiway_close_prob,
            priority_prob=self.env_cfg.priority_change_prob,
            delay_prob=self.env_cfg.delay_prob,
            layout=self.env_cfg.layout,
            auto_speed=getattr(self.env_cfg, "auto_speed", True),
            disable_corridor=getattr(self.env_cfg, "disable_corridor", False),
            max_speed=self.env_cfg.max_speed,
            min_speed=self.env_cfg.min_speed,
            speed_step=self.env_cfg.speed_step,
            gate_speed_limit=getattr(self.env_cfg, "gate_speed_limit", 0.15),
            turn_speed_limit=getattr(self.env_cfg, "turn_speed_limit", 1.0),
            turn_brake_buffer=getattr(self.env_cfg, "turn_brake_buffer", 0.5),
            enable_anticipatory_turn_braking=getattr(self.env_cfg, "enable_anticipatory_turn_braking", True),
            midroute_start_prob=getattr(self.env_cfg, "midroute_start_prob", 0.0),
            random_flow_mix=getattr(self.env_cfg, "random_flow_mix", True),
            entry_stagger_steps=getattr(self.env_cfg, "entry_stagger_steps", 120),
            departure_stagger_steps=getattr(self.env_cfg, "departure_stagger_steps", 60),
            gate_shadow_mode=getattr(self.env_cfg, "gate_shadow_mode", "extended"),
            shadow_release_headway_steps=getattr(
                self.env_cfg, "shadow_release_headway_steps", 0
            ),
            physical_clearance_shield=getattr(
                self.env_cfg, "physical_clearance_shield", False
            ),
            physical_clearance_min_distance_sq=getattr(
                self.env_cfg, "physical_clearance_min_distance_sq", 3.0
            ),
            clearance_risk_observation=getattr(
                self.env_cfg, "clearance_risk_observation", False
            ),
            pre_entry_clearance_guard=getattr(
                self.env_cfg, "pre_entry_clearance_guard", False
            ),
            approach_reservation_progress=getattr(
                self.env_cfg, "approach_reservation_progress", 0.60
            ),
        )

        # [Ablation Study] 是否启用 A* 先验路径引导
        # 设为 False 时触发 MAPPO-NoGuide 消融实验，屏蔽观测中的 A* 特征
        self.use_astar_guide: bool = getattr(self.env_cfg, "use_astar_guide", True)

        # A* 规划器
        node_positions = {
            nid: n.position for nid, n in self.graph_env.nodes.items()
        }
        self.planner = AStarPlanner(node_positions)
        coordinates = np.asarray(list(node_positions.values()), dtype=np.float32)
        self._coord_min = coordinates.min(axis=0)
        self._coord_span = np.maximum(coordinates.max(axis=0) - self._coord_min, 1.0)
        all_distances = [
            value
            for row in getattr(self.graph_env, "dist", {}).values()
            for value in row.values()
            if value >= 0
        ]
        self._max_path_distance = max(all_distances, default=100.0)

        self.agents: list[AircraftAgent] = []
        self.num_agents = self.env_cfg.num_agents
        
        # 精简后的维度 = max_neighbors * 7 + 6
        self.obs_dim = self.env_cfg.max_neighbors * 7 + 6
        
        self.action_dim = AircraftAgent.ACTION_DIM
        self.prev_positions: list[int] = []

    def reset(self, starts=None, goals=None, delays=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """重置环境并返回初始观测和动作掩码。

        Args:
            starts:  可选，各智能体初始节点列表（传 None 则由底层环境随机生成）。
            goals:   可选，各智能体目标节点列表（传 None 则由底层环境随机生成）。
            delays:  可选，各智能体的 t0 发车延时列表（测试阶段传入时刻表）。

        Returns:
            (observations, action_masks) — 每个智能体的观测和掩码列表。
        """
        info = self.graph_env.reset(starts=starts, goals=goals, delays=delays)

        graph = self.graph_env.get_graph_for_planner()

        # 为每个智能体规划先验路径
        self.agents = []
        for i in range(self.num_agents):
            start = self.graph_env.agent_positions[i]
            goal = self.graph_env.agent_goals[i]
            path = self.planner.plan_path(start, goal, graph)
            if not path:
                path = [start]  # fallback
            self.graph_env.set_reference_path(i, path)
            agent = AircraftAgent(
                agent_id=i,
                planned_path=path,
                num_agents=self.num_agents,
                max_speed=self.env_cfg.max_speed,
                min_speed=self.env_cfg.min_speed,
                speed_step=self.env_cfg.speed_step,
                max_neighbors=self.env_cfg.max_neighbors,
                prevent_occupied_target_entry=getattr(
                    self.env_cfg, "prevent_occupied_target_entry", False
                ),
                enforce_reference_progress=getattr(
                    self.env_cfg, "enforce_reference_progress", False
                ),
                clearance_aware_reference_progress=getattr(
                    self.env_cfg, "clearance_aware_reference_progress", False
                ),
                pre_entry_clearance_guard=getattr(
                    self.env_cfg, "pre_entry_clearance_guard", False
                ),
                auto_speed=getattr(self.env_cfg, "auto_speed", True),
                use_astar_guide=self.use_astar_guide,   # [Ablation] 透传消融开关
            )
            self.agents.append(agent)

        self.prev_positions = list(self.graph_env.agent_positions)
        # 初始化距离记录
        self._prev_dist = [
            self.graph_env.get_shortest_path_distance(
                self.graph_env.agent_positions[i], self.graph_env.agent_goals[i]
            )
            for i in range(self.num_agents)
        ]
        self._prev_progress = [0.0] * self.num_agents
        self._stuck_counters = [0] * self.num_agents
        # 追踪已发放 completion 奖励的智能体，避免重复计算
        self._agent_done_awarded: set[int] = set()
        # 团队奖励跨 episode 清零
        self._team_bonus_awarded: set[int] = set()
        # 死锁位置统计（每个 episode 重置，记录每次死锁事件的详细位置）
        self._deadlock_events: list[dict] = []
        self._safe_completed = [False] * self.num_agents
        self._last_preemptive_replan = [-10_000] * self.num_agents

        return self._build_observations_and_masks()

    def activate_slot(
        self,
        slot: int,
        start: int,
        goal: int,
        *,
        priority: int = 1,
    ) -> None:
        """Reuse one fixed policy slot for a newly released scheduled flight.

        Paper-scale simulations must retain the trained population size.  This
        method resets only one physical slot; flights that have not yet been
        admitted therefore never become virtual occupants in the environment.
        """
        if not 0 <= slot < self.num_agents:
            raise IndexError(f"slot {slot} outside [0, {self.num_agents})")

        graph_env = self.graph_env
        graph_env.corridor_mgr.leave_all(slot)
        graph_env.pos[slot] = start
        graph_env.goal[slot] = goal
        graph_env.on_edge[slot] = False
        graph_env.edge_idx[slot] = None
        graph_env.progress[slot] = 0.0
        graph_env.speed[slot] = 1.0
        if hasattr(graph_env, "edge_speed_cap"):
            graph_env.edge_speed_cap[slot] = None
            graph_env.turn_brake_active[slot] = False
        graph_env.done[slot] = False
        graph_env.crashed[slot] = False
        if hasattr(graph_env, "clearance_blocked"):
            graph_env.clearance_blocked[slot] = False
        if hasattr(graph_env, "clearance_blocker_by_agent"):
            graph_env.clearance_blocker_by_agent[slot] = None
            graph_env.clearance_min_distance_sq_by_agent[slot] = float("inf")
        if hasattr(graph_env, "edge_hold_reason"):
            graph_env.edge_hold_reason[slot] = None
        graph_env.delayed[slot] = False
        graph_env.delay_timers[slot] = 0
        graph_env.start_delays[slot] = 0
        graph_env.priority[slot] = priority
        graph_env.prev_edge_dir[slot] = None
        graph_env.came_from[slot] = None
        graph_env.prev_speed_delta[slot] = 0.0

        graph = graph_env.get_graph_for_planner()
        path = self.planner.plan_path(start, goal, graph) or [start]
        graph_env.set_reference_path(slot, path)
        self.agents[slot] = AircraftAgent(
            agent_id=slot,
            planned_path=path,
            num_agents=self.num_agents,
            max_speed=self.env_cfg.max_speed,
            min_speed=self.env_cfg.min_speed,
            speed_step=self.env_cfg.speed_step,
            max_neighbors=self.env_cfg.max_neighbors,
            prevent_occupied_target_entry=getattr(
                self.env_cfg, "prevent_occupied_target_entry", False
            ),
            enforce_reference_progress=getattr(
                self.env_cfg, "enforce_reference_progress", False
            ),
            clearance_aware_reference_progress=getattr(
                self.env_cfg, "clearance_aware_reference_progress", False
            ),
            pre_entry_clearance_guard=getattr(
                self.env_cfg, "pre_entry_clearance_guard", False
            ),
            auto_speed=getattr(self.env_cfg, "auto_speed", True),
            use_astar_guide=self.use_astar_guide,
        )
        self.prev_positions[slot] = start
        self._prev_dist[slot] = graph_env.get_shortest_path_distance(start, goal)
        self._prev_progress[slot] = 0.0
        self._stuck_counters[slot] = 0
        self._safe_completed[slot] = False
        if hasattr(self, "_last_preemptive_replan"):
            self._last_preemptive_replan[slot] = -10_000
        self._agent_done_awarded.discard(slot)
        self._team_bonus_awarded.discard(slot)

    def deactivate_slot(self, slot: int) -> None:
        """Remove a slot from physical occupancy until another flight is due."""
        if not 0 <= slot < self.num_agents:
            raise IndexError(f"slot {slot} outside [0, {self.num_agents})")
        self.graph_env.corridor_mgr.leave_all(slot)
        self.graph_env.done[slot] = True
        self.graph_env.crashed[slot] = False
        self.graph_env.on_edge[slot] = False
        self.graph_env.edge_idx[slot] = None
        self.graph_env.progress[slot] = 0.0
        self.graph_env.speed[slot] = 0.0
        if hasattr(self.graph_env, "edge_speed_cap"):
            self.graph_env.edge_speed_cap[slot] = None
            self.graph_env.turn_brake_active[slot] = False
        self.graph_env.start_delays[slot] = 0
        if hasattr(self.graph_env, "clearance_blocked"):
            self.graph_env.clearance_blocked[slot] = False
        if hasattr(self.graph_env, "clearance_blocker_by_agent"):
            self.graph_env.clearance_blocker_by_agent[slot] = None
            self.graph_env.clearance_min_distance_sq_by_agent[slot] = float("inf")
        if hasattr(self.graph_env, "edge_hold_reason"):
            self.graph_env.edge_hold_reason[slot] = None

    def step(
            self, actions: list[int]
        ) -> tuple[list[np.ndarray], list[float], list[bool], list[np.ndarray], dict[str, Any]]:
            """执行一步。"""
            self.prev_positions = list(self.graph_env.agent_positions)
            prev_done = list(self.graph_env.done)
            prev_crashed = list(self.graph_env.crashed)
            # Preserve decision-time feasibility. Reward computation happens
            # after movement, when an aircraft may already be on an edge and
            # the node-level mask is no longer available.
            pre_observations, pre_masks = self._build_observations_and_masks()
            pre_on_edge = list(self.graph_env.agent_on_edge)
            
            # 1. 执行底层环境步进
            info, all_completed, crashed = self.graph_env.step(actions)
            
            # 2. 【修复点】定义 dones。通常底层环境会维护一个 done 数组。
            # 如果底层环境没有，则通过位置和 crashed 状态构造：
            dones = [self.graph_env.done[i] for i in range(self.num_agents)]

            # 3. 同步 path_index，确保构造观测时使用的是最新的路径进度
            for i, agent in enumerate(self.agents):
                if not dones[i]: # 现在 dones 已经定义，不会报错了
                    cur_pos = self.graph_env.agent_positions[i]
                    if cur_pos != self.prev_positions[i]:
                        # 修正：这里应该是同步方法名，确认你的 AircraftAgent 里是 sync_path_index 还是 advance_path
                        agent.sync_path_index(cur_pos)

            # 4. 构造返回数据：未放行/已结束的 agent 只暴露确定性 no-op。
            observations, masks = self._build_observations_and_masks()
            crashed_this_step = [
                self.graph_env.crashed[i] and not prev_crashed[i]
                for i in range(self.num_agents)
            ]
            completed_this_step = [
                (not prev_done[i])
                and self.graph_env.done[i]
                and (not self.graph_env.crashed[i])
                and self.graph_env.agent_positions[i] == self.graph_env.agent_goals[i]
                for i in range(self.num_agents)
            ]
            for i, completed in enumerate(completed_this_step):
                if completed:
                    self._safe_completed[i] = True
            
            observations, masks = self._build_observations_and_masks()

            # 5. 计算奖励（传入刚才得到的 crashed 信息）
            rewards, reward_breakdown = self._compute_rewards(
                info,
                crashed_this_step,
                completed_this_step,
                actions=actions,
                pre_observations=pre_observations,
                pre_masks=pre_masks,
                pre_on_edge=pre_on_edge,
            )

            # 真正到达目标的智能体判定
            goal_reached = list(self._safe_completed)

            # 6. 填充 Info 字典
            info["reward_breakdown"] = reward_breakdown
            info["goal_reached"] = goal_reached   
            info["crashed"] = crashed             
            info["deadlock_events"] = (self._deadlock_events if self.enable_deadlock_recovery else [])  
            
            self._prev_progress = list(self.graph_env.progress)

            # 7. 死锁检测后触发动态重规划（与奖励计算解耦）
            if self.enable_deadlock_recovery:
                self._check_and_replan()
            return observations, rewards, dones, masks, info

    '''
    def get_global_state(self) -> np.ndarray:
        """获取全局状态（用于 centralized critic）。

        Returns:
            全局状态向量。
        """
        states = []
        for i in range(self.num_agents):
            states.extend([
                float(self.graph_env.agent_positions[i]),
                float(self.graph_env.agent_goals[i]),
                float(self.graph_env.agent_speeds[i]),
                1.0 if self.graph_env.agent_on_edge[i] else 0.0,
                1.0 if self.graph_env.agent_done[i] else 0.0,
            ])
        return np.array(states, dtype=np.float32)'''
    def get_global_state(self) -> np.ndarray:
        return self._build_topology_critic_state()

    def _legacy_global_state(self) -> np.ndarray:
        states = []
        for i in range(self.num_agents):
            states.extend([
                float(self.graph_env.agent_positions[i]),
                float(self.graph_env.agent_goals[i]),
                float(self.graph_env.agent_speeds[i]),
                1.0 if self.graph_env.agent_on_edge[i] else 0.0,
                1.0 if self.graph_env.agent_done[i] else 0.0,
                min(max(self.graph_env.priority[i], 0), self.num_agents) / self.num_agents, # 优先级
                self._prev_dist[i] / 100.0,                   # 距离目标的紧迫感
                (
                    self.agents[i].path_index / max(len(self.agents[i].planned_path), 1)
                    if self.use_astar_guide else 0.0
                )
            ])
        # 加入全局时间压力和冲突计数
        states.append(self.graph_env.step_count / self.graph_env.max_steps)
        states.append(self.graph_env.conflict_count / 10.0) # 归一化冲突
        
        return np.array(states, dtype=np.float32)

    def _build_topology_critic_state(self) -> np.ndarray:
        """Build a normalized topology-aware centralized-critic state."""
        states: list[float] = []
        max_steps = max(self.graph_env.max_steps, 1)

        for i in range(self.num_agents):
            pos = self.graph_env.agent_positions[i]
            goal = self.graph_env.agent_goals[i]
            pos_xy = np.asarray(self.graph_env._get_agent_coords(i), dtype=np.float32)
            goal_xy = np.asarray(self.graph_env.nodes[goal].position, dtype=np.float32)
            pos_norm = (pos_xy - self._coord_min) / self._coord_span
            goal_norm = (goal_xy - self._coord_min) / self._coord_span

            delay = max(getattr(self.graph_env, "start_delays", [0] * self.num_agents)[i], 0)
            done = bool(self.graph_env.done[i])
            crashed = bool(self.graph_env.crashed[i])
            distance = max(self._prev_dist[i], 0.0)

            corridor_load = 0.0
            corridor_locked = 0.0
            next_node = None
            if self.graph_env.agent_on_edge[i]:
                edge_index = self.graph_env.edge_idx[i]
                if edge_index is not None:
                    edge = self.graph_env.edges[edge_index]
                    next_node = edge.end
                    corridor = self.graph_env.corridor_mgr.get_corridor(edge.start, edge.end)
                else:
                    corridor = None
            else:
                next_node = self.agents[i].get_next_planned_node(self.graph_env)
                corridor = (
                    self.graph_env.corridor_mgr.get_corridor(pos, next_node)
                    if next_node is not None and next_node != pos
                    else None
                )
            if corridor is not None:
                corridor_load = len(corridor.occupied_by) / max(self.num_agents, 1)
                if next_node is not None and next_node != pos:
                    direction = corridor.edge_direction(pos, next_node)
                    corridor_locked = float(
                        not self.graph_env.corridor_mgr.can_enter(
                            corridor, direction, agent_id=i
                        )
                    )

            states.extend([
                float(pos_norm[0]), float(pos_norm[1]),
                float(goal_norm[0]), float(goal_norm[1]),
                self.graph_env.agent_speeds[i] / max(self.env_cfg.max_speed, 1.0),
                float(self.graph_env.agent_on_edge[i]),
                float(self.graph_env.progress[i]),
                min(delay / max_steps, 1.0),
                float(done), float(crashed),
                min(max(self.graph_env.priority[i], 0), max(self.num_agents - 1, 1)) / max(self.num_agents - 1, 1),
                min(distance / max(self._max_path_distance, 1.0), 1.0),
                corridor_load, corridor_locked,
                float(delay == 0 and not done),
            ])

        active_count = sum(
            not self.graph_env.done[i] and self.graph_env.start_delays[i] == 0
            for i in range(self.num_agents)
        )
        states.extend([
            self.graph_env.step_count / max_steps,
            min(self.graph_env.conflict_count / max(self.num_agents, 1), 1.0),
            active_count / max(self.num_agents, 1),
            sum(self._safe_completed) / max(self.num_agents, 1),
        ])
        return np.asarray(states, dtype=np.float32)

    @property
    def global_state_dim(self) -> int:
        """扩展后的全局状态维度。
        每个智能体 8 个特征: 
        [pos, goal, speed, on_edge, done, priority, dist_to_goal, path_progress]
        + 全局特征: [step_count_ratio, total_conflicts]
        """
        return self.num_agents * CRITIC_AGENT_FEATURES + CRITIC_GLOBAL_FEATURES

    @property
    def critic_state_schema(self) -> str:
        return CRITIC_STATE_SCHEMA
    
    def _compute_rewards(
        self,
        info: dict[str, Any],
        crashed_this_step: list[bool],
        completed_this_step: list[bool],
        actions: list[int] | None = None,
        pre_observations: list[np.ndarray] | None = None,
        pre_masks: list[np.ndarray] | None = None,
        pre_on_edge: list[bool] | None = None,
    ) -> tuple[list[float], dict[str, float]]:
        """计算每个智能体的奖励，并返回各子项统计。

        设计要点：
        - completion + efficiency_bonus 只在智能体**首次完成**时发放一次
        - 发生 crashed 立刻给出巨大惩罚并视作 done（无 completion）
        - 所有人安全完成时给出团队奖励 team_bonus
        - done/crashed 后的步骤直接跳过
        - 效率奖励 = efficiency_rate × 剩余步数
        - 分项统计通过 reward_breakdown 返回，供训练日志使用

        Args:
            info: 环境信息字典。
            crashed: 各个智能体在此 step 是否发生车祸

        Returns:
            (rewards, reward_breakdown)
        """
        rewards = []
        rb: dict[str, float] = {
            "step": 0.0,
            "completion": 0.0,
            "efficiency": 0.0,
            "proximity": 0.0,
            "conflict": 0.0,
            "wait": 0.0,
            "yield": 0.0,
            "path_dev": 0.0,
            "deadlock": 0.0,
            "team_bonus": 0.0,
            "team_prog": 0.0,
            "block": 0.0,
            "danger": 0.0,
            "timeout": 0.0,
            "avoidable_wait": 0.0,
        }

        max_steps = self.graph_env.max_steps
        current_step = self.graph_env.step_count
        remaining_steps = max(max_steps - current_step, 0)
        
        # 判定是否触发了全员完美过关
        # 所有人都是 graph_env.done 为 True 且 谁都没 crash
        team_success = all(self._safe_completed[i] for i in range(self.num_agents))

        # 1. 团队进度奖励：本步新完成的智能体数量
        new_completions = sum(
            1 for i in range(self.num_agents) 
            if i not in self._agent_done_awarded and completed_this_step[i]
        )
        team_prog_reward = 0.0
        if new_completions > 0:
            team_prog_reward = (getattr(self.reward_cfg, 'team_progress', 10.0) * new_completions) / self.num_agents

        for i in range(self.num_agents):
            r = 0.0

            # 已发放过奖励的智能体：只结算团队进度奖并跳过
            if i in self._agent_done_awarded:
                if team_prog_reward != 0:
                    rb["team_prog"] += team_prog_reward
                    rewards.append(team_prog_reward)
                else:
                    rewards.append(0.0)
                continue

            # 加上团队进度奖
            if team_prog_reward != 0:
                r += team_prog_reward
                rb["team_prog"] += team_prog_reward

            # 用位置判断是否真正到达目标（done[i] 在 crash 时也为 True，不可信赖）
            reached_goal = completed_this_step[i]

            # 完成奖励：只要位置到达目标，无论是否同时碰撞，均发放
            safe_reached_goal = reached_goal and not crashed_this_step[i]
            if safe_reached_goal:
                r_comp = getattr(self.reward_cfg, 'completion', 10.0)
                r_eff = getattr(self.reward_cfg, 'efficiency_rate', 0.1) * remaining_steps
                r += r_comp + r_eff
                rb["completion"] += r_comp
                rb["efficiency"] += r_eff
                self._prev_dist[i] = 0.0

            # Crash 惩罚：独立计算，不因 reached_goal 而跳过
            if crashed_this_step[i]:
                r_conf = getattr(self.reward_cfg, 'conflict', -10.0)
                r += r_conf
                rb["conflict"] += r_conf

            # 完成或崩溃后标记，不再处理后续步骤
            if safe_reached_goal or crashed_this_step[i]:
                # Team Bonus （只能拿一次，且必须全员无碰撞才触发）
                if team_success and i not in self._team_bonus_awarded:
                    tb = getattr(self.reward_cfg, 'team_bonus', 20.0)
                    r += tb
                    rb["team_bonus"] += tb
                    self._team_bonus_awarded.add(i)
                self._agent_done_awarded.add(i)
                rewards.append(r)
                continue


            # 如果智能体还在虚拟队列排队（未达到 t0），不计入任何惩罚，跳过
            if getattr(self.graph_env, 'start_delays', None) and self.graph_env.start_delays[i] > 0:
                rewards.append(r)
                continue

            # A generic post-step waiting penalty cannot tell whether the
            # policy chose to idle or whether every route was blocked. Use the
            # pre-action mask and local risk flags to penalize only avoidable
            # idling. This remains zero for existing configurations.
            if (
                actions is not None
                and pre_observations is not None
                and pre_masks is not None
                and pre_on_edge is not None
                and not pre_on_edge[i]
                and int(actions[i]) == 7
            ):
                clear_routes = [
                    action_id for action_id in range(min(5, len(pre_masks[i])))
                    if pre_masks[i][action_id] > 0
                    and pre_observations[i][action_id * 7 + 3] < 0.5
                    and pre_observations[i][action_id * 7 + 4] < 0.5
                ]
                if clear_routes:
                    r_avoidable_wait = getattr(self.reward_cfg, 'avoidable_wait', 0.0)
                    if r_avoidable_wait != 0.0:
                        r += r_avoidable_wait
                        rb["avoidable_wait"] += r_avoidable_wait

            # --- 渐进式碰撞威慑 (Graduated Collision Reward) ---
            danger_penalty = self._proximity_danger_reward(i)
            if danger_penalty < 0:
                r += danger_penalty
                rb["danger"] += danger_penalty

            # 未完成智能体：收取 step_penalty
            r_step = getattr(self.reward_cfg, 'step_penalty', -0.01)
            r += r_step
            rb["step"] += r_step

            cur_pos = self.graph_env.agent_positions[i]
            prev_pos = self.prev_positions[i]
            on_edge = self.graph_env.agent_on_edge[i]

            # --- 1. 计算当前包含物理边上进度的连续最短距离 ---
            goal = self.graph_env.agent_goals[i]
            if on_edge and getattr(self.graph_env, 'edge_idx', [None]*self.num_agents)[i] is not None:
                e = self.graph_env.edges[self.graph_env.edge_idx[i]]
                dist_from_end = self.graph_env.get_shortest_path_distance(e.end, goal)
                if dist_from_end >= 0:
                    prog = self.graph_env.progress[i]
                    remaining_edge_len = (1.0 - prog) * e.length
                    cur_continuous_dist = dist_from_end + remaining_edge_len
                else:
                    cur_continuous_dist = -1.0
            else:
                cur_continuous_dist = self.graph_env.get_shortest_path_distance(cur_pos, goal)

            # --- 2. 连续平滑的 Proximity 奖励 (解决边上乱动问题) ---
            if cur_continuous_dist >= 0 and self._prev_dist[i] >= 0:
                delta = self._prev_dist[i] - cur_continuous_dist
                # delta > 0 表示距离真实地缩短了，无论是在节点跳跃还是在边上滑行
                if delta != 0:
                    r_prox = getattr(self.reward_cfg, 'proximity', 0.8) * delta
                    r += r_prox
                    rb["proximity"] += r_prox
            self._prev_dist[i] = cur_continuous_dist

            # --- 3. 节点跳跃式逻辑：仅在抵达新节点时结算先验路径偏离惩罚 ---
            if self.use_astar_guide and not on_edge and cur_pos != prev_pos:
                # 先验路径偏离惩罚（使用 set 做 O(1) 查找）
                if cur_pos not in self.agents[i].planned_path_set:
                    r_pdev = getattr(self.reward_cfg, 'path_deviation', -2.0)
                    r += r_pdev
                    rb["path_dev"] += r_pdev
            
            # 等待与阻塞逻辑判断
            is_waiting_at_node = (cur_pos == prev_pos and not on_edge)
            if is_waiting_at_node:
                my_next = (
                    self.agents[i].get_next_planned_node(self.graph_env)
                    if self.use_astar_guide else None
                )
                
                # --- 增强等待区分度：识别 forced_wait ---
                forced_wait = False
                if my_next is not None and my_next != cur_pos:
                    for j in range(self.num_agents):
                        if j != i and not self.graph_env.done[j]:
                            j_pos = self.graph_env.agent_positions[j]
                            # 如果目标节点被占，或者目标节点相关边上有其他飞机
                            if j_pos == my_next or (self.graph_env.agent_on_edge[j] and j_pos in (cur_pos, my_next)):
                                forced_wait = True
                                break
                
                # 仅对非被迫等待的情况施加惩罚
                if not forced_wait:
                    r_wait = getattr(self.reward_cfg, 'wait', -0.05)
                    r += r_wait
                    rb["wait"] += r_wait

                # --- 阻塞别人惩罚 ---
                is_at_gate_tmp = (len(self.graph_env.adj[cur_pos]) == 1)
                if self.use_astar_guide and not is_at_gate_tmp:
                    blocking_others = False
                    for j in range(self.num_agents):
                        if j != i and not self.graph_env.done[j]:
                            j_next = self.agents[j].get_next_planned_node(self.graph_env)
                            if j_next == cur_pos:
                                blocking_others = True
                                break
                    if blocking_others:
                        r_block = getattr(self.reward_cfg, 'blocking', -0.2)
                        r += r_block
                        rb["block"] += r_block

                # ─── 让行成功奖励 ───────────────────────────────────────────
                # 条件：我主动停在节点上等待（非边上，非死锁触发），
                #       且在我等待的这一步内，有其他agent确实推进了路径（prev→cur不同）
                # 意义：学到「在拥挤路口让别人先走」是有价值的协同行为，为防止刷分漏洞，必须是身边（邻居内）的避让
                r_yield_cfg = getattr(self.reward_cfg, 'yield_bonus', 0.3)
                my_neighbors = self.graph_env.get_neighbors(cur_pos)
                for j in range(self.num_agents):
                    if (j != i
                            and not self.graph_env.done[j]
                            and not self.graph_env.agent_on_edge[j]
                            and self.graph_env.agent_positions[j] != self.prev_positions[j]):
                        
                        j_prev = self.prev_positions[j]
                        j_cur = self.graph_env.agent_positions[j]
                        # 仅当对方是从我的邻居节点，或者移向我的邻居节点时，才算真正产生了近距离的物理“让路”
                        if j_prev in my_neighbors or j_cur in my_neighbors:
                            r += r_yield_cfg
                            rb["yield"] += r_yield_cfg
                            break  # 每步最多只拿一次，杜绝多人移动产生的倍增刷分

            # (已删除 redundant 的速度偏差惩罚 speed_dev)

            # 死锁趋势惩罚：梯度化提前介入，不再等到 >10 才惩罚
            prog_diff = abs(self.graph_env.progress[i] - self._prev_progress[i])
            if cur_pos == prev_pos and (not on_edge or prog_diff < 0.01):
                self._stuck_counters[i] += 1
            else:
                self._stuck_counters[i] = 0
            is_at_gate = (len(self.graph_env.adj[cur_pos]) == 1)
            sc = self._stuck_counters[i]
            if self.enable_deadlock_recovery and not is_at_gate and sc > 0:
                r_dl_cfg = getattr(self.reward_cfg, 'deadlock', -0.2)
                # Once an aircraft is already stopped near the end of an
                # occupied edge, the current policy has no edge action with
                # which to correct the situation. Applying the full penalty
                # every following step turns one blocking event into thousands
                # of negative return units and obscures the causal route
                # choice. Use sparse escalation signals instead.
                if sc == 4:
                    r_dl = r_dl_cfg * 0.5
                elif sc == 7:
                    r_dl = r_dl_cfg * 1.5
                elif sc == 11:
                    r_dl = r_dl_cfg * 3.0
                elif sc > 11 and (sc - 11) % 12 == 0:
                    r_dl = r_dl_cfg
                else:
                    r_dl = 0.0
                if r_dl != 0.0:
                    r += r_dl
                    rb["deadlock"] += r_dl

                # 记录死锁事件（仅在 sc==11 时记录一次，避免重复）
                if sc == 11:
                    edge_info = None
                    if on_edge and self.graph_env.edge_idx[i] is not None:
                        e_obj = self.graph_env.edges[self.graph_env.edge_idx[i]]
                        edge_info = (e_obj.start, e_obj.end)
                    self._deadlock_events.append({
                        "agent": i,
                        "node": cur_pos,
                        "on_edge": on_edge,
                        "edge": edge_info,
                        "goal": self.graph_env.agent_goals[i],
                        "step": self.graph_env.step_count,
                    })

            # A timeout is not a neutral outcome. Penalize only aircraft that
            # were actually released, and scale the terminal cost by remaining
            # route distance so near-goal aircraft are not treated as equally
            # poor as aircraft that never made operational progress.
            if current_step >= max_steps:
                timeout_base = getattr(self.reward_cfg, "timeout", 0.0)
                if timeout_base != 0.0:
                    remaining_ratio = min(
                        max(cur_continuous_dist, 0.0) / max(self._max_path_distance, 1.0),
                        1.0,
                    )
                    r_timeout = timeout_base * (0.25 + 0.75 * remaining_ratio)
                    r += r_timeout
                    rb["timeout"] += r_timeout

            rewards.append(r)

        return rewards, rb

    def _proximity_danger_reward(self, agent_i: int) -> float:
        """
        渐进碰撞威慑奖励（Graduated Collision Avoidance Reward）
        已弃用 (Returns 0.0)：为了保证快速收敛，我们切回了 auto_speed=True。
        在自动巡航下智能体无法自主刹车，因此取消距离惩罚。
        """
        return 0.0
        
        if self.graph_env.done[agent_i] or self.graph_env.crashed[agent_i]:
            return 0.0
            
        import math
        xi, yi = self.graph_env._get_agent_coords(agent_i)
        danger_penalty = 0.0
        
        for j in range(self.num_agents):
            if j == agent_i or self.graph_env.done[j] or self.graph_env.crashed[j]:
                continue
            
            # 排队中的虚拟飞机不计算物理危险
            if getattr(self.graph_env, 'start_delays', None) and self.graph_env.start_delays[j] > 0:
                continue
            if getattr(self.graph_env, 'start_delays', None) and self.graph_env.start_delays[agent_i] > 0:
                continue

            xj, yj = self.graph_env._get_agent_coords(j)
            dist_units = math.hypot(xi - xj, yj - yi)
            dist_m = dist_units * 5.833  # L_SCALE
            
            if dist_m < 2.0:
                danger_penalty -= 2.0
            elif dist_m < 5.0:
                danger_penalty -= 0.5
            elif dist_m < 10.0:
                danger_penalty -= 0.1
                
        return danger_penalty

    def _check_and_replan(self) -> None:
        """死锁触发的动态 A* 重规划（与奖励计算解耦，在 step() 末尾调用）。

        对每个卡死超过阀値且不在机位的智能体，每隔 5 步尝试绕行重规划。
        """
        if not self.use_astar_guide or not self.enable_deadlock_recovery:
            return

        graph = self.graph_env.get_graph_for_planner()
        for i, agent in enumerate(self.agents):
            if self.graph_env.done[i]:
                continue
            if self._stuck_counters[i] <= 10:
                continue
            cur_pos = self.graph_env.agent_positions[i]
            if len(self.graph_env.adj[cur_pos]) == 1:  # 机位不重规划
                continue
            if self._stuck_counters[i] % 5 == 1:
                blocked_nodes = {
                    self.graph_env.agent_positions[j]
                    for j in range(self.num_agents)
                    if j != i
                    and not self.graph_env.done[j]
                    and self._stuck_counters[j] >= 3
                }
                next_node = agent.get_next_planned_node(self.graph_env)
                if next_node is not None and next_node != cur_pos:
                    agent.replan_path(
                        self.graph_env,
                        self.planner,
                        graph,
                        blocked_nodes=blocked_nodes,
                    )

    def _inactive_obs_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic no-op view for agents that cannot act."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        mask = np.zeros(self.action_dim, dtype=np.float32)
        mask[7] = 1.0
        return obs, mask

    def _maybe_preemptively_replan(
        self,
        agent_id: int,
        agent: AircraftAgent,
        mask: np.ndarray,
    ) -> bool:
        """Refresh an A* reference before an avoidable blocked wait.

        The legacy replan trigger starts only after a prolonged stall. Here the
        reference is refreshed when its first edge is already infeasible and
        a different topological route may be available. This affects guidance
        features only; the policy still selects the final feasible action.
        """
        if (
            not self.use_astar_guide
            or not self.enable_deadlock_recovery
            or not getattr(self.env_cfg, "preemptive_replan", False)
            or self.graph_env.agent_on_edge[agent_id]
            or self.graph_env.done[agent_id]
        ):
            return False

        current_step = self.graph_env.step_count
        cooldown = max(
            int(getattr(self.env_cfg, "preemptive_replan_cooldown_steps", 3)), 1
        )
        if current_step - self._last_preemptive_replan[agent_id] < cooldown:
            return False

        current_node = self.graph_env.agent_positions[agent_id]
        reference_node = agent.get_next_planned_node(self.graph_env)
        if reference_node is None or reference_node == current_node:
            return False
        reference_action = next(
            (
                action_id
                for action_id, (node, _) in enumerate(
                    self.graph_env.adj[current_node][:5]
                )
                if node == reference_node
            ),
            None,
        )
        reference_clearance_blocked = (
            getattr(self.env_cfg, "clearance_aware_reference_progress", False)
            and self.graph_env.is_node_clearance_blocked(reference_node, agent_id)
        )
        # A reference successor that is no longer adjacent means the cached
        # A* path is stale (typically after an earlier diversion). It must be
        # replanned rather than silently treated as a clear route.
        if (
            reference_action is not None
            and mask[reference_action] > 0
            and not reference_clearance_blocked
        ):
            return False

        # Static A* accepts blocked nodes, so encode current physical
        # occupancy and the unavailable reference successor as temporary
        # exclusions. Do not block the final goal: an occupied stand is a
        # queueing condition, not a reason to declare the mission unreachable.
        blocked_nodes: set[int] = {reference_node}
        for other_id in range(self.num_agents):
            if (
                other_id == agent_id
                or self.graph_env.done[other_id]
                or self.graph_env.start_delays[other_id] > 0
            ):
                continue
            if self.graph_env.agent_on_edge[other_id]:
                edge_index = self.graph_env.edge_idx[other_id]
                if edge_index is not None:
                    blocked_nodes.add(self.graph_env.edges[edge_index].end)
            else:
                blocked_nodes.add(self.graph_env.agent_positions[other_id])
        blocked_nodes.discard(current_node)
        blocked_nodes.discard(self.graph_env.agent_goals[agent_id])

        old_path = list(agent.planned_path)
        agent.replan_path(
            self.graph_env,
            self.planner,
            self.graph_env.get_graph_for_planner(),
            blocked_nodes=blocked_nodes,
        )
        # Keep the physical turn preview synchronized with the latest route.
        self.graph_env.set_reference_path(agent_id, agent.planned_path)
        self._last_preemptive_replan[agent_id] = current_step
        return agent.planned_path != old_path

    def _build_observations_and_masks(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        observations = []
        masks = []
        for i, agent in enumerate(self.agents):
            if not self.graph_env.done[i] and self.graph_env.start_delays[i] == 0:
                obs = agent.get_observation(self.graph_env, self.agents)
                mask = agent.get_action_mask(self.graph_env, self.agents)
                if self._maybe_preemptively_replan(i, agent, mask):
                    obs = agent.get_observation(self.graph_env, self.agents)
                    mask = agent.get_action_mask(self.graph_env, self.agents)
                observations.append(obs)
                masks.append(mask)
            else:
                obs, mask = self._inactive_obs_mask()
                observations.append(obs)
                masks.append(mask)
        return observations, masks

    def close(self) -> None:
        """关闭环境（预留接口）。"""
        pass
