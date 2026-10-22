"""预设离散扰动场景模块。

测试阶段专用：将三类扰动从训练时的「随机概率触发」改为「确定性时刻触发」，
使每类扰动场景可重复、可控制、可独立评估。

使用方式:
    from perturbation_scenarios import PerturbationScenario, apply_perturbation_step

    scenario = PerturbationScenario.DELAY   # 选择场景
    injector = ScenarioInjector(scenario, env.graph_env)
    for step in range(max_steps):
        injector.step(step)                  # 每步调用，按计划注入扰动
        obs, rew, done, mask, info = env.step(actions)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.taxi_graph_env import TaxiGraphEnv


# =========================================================
# 场景枚举
# =========================================================

class PerturbationScenario(Enum):
    BASELINE = "baseline"       # 无扰动基准场景
    DELAY    = "delay"          # 场景S2: 航班延误
    PRIORITY = "priority"       # 场景S3: 优先级变化
    CLOSURE  = "closure"        # 场景S4: 滑行道临时关闭


# =========================================================
# 预设扰动计划表（确定性时间序列）
# =========================================================

@dataclass
class DelayEvent:
    """单次延误事件：在 trigger_step 步时，对尚未推出的飞机增加 duration 步的推出等待。"""
    trigger_step: int
    agent_ids: list[int]   # 受影响飞机索引（-1 表示随机选取1架等待推出的飞机）
    duration: int          # 额外等待步数（追加到 start_delays）


@dataclass
class PriorityEvent:
    """单次优先级突变事件：在 trigger_step 步将 n_boost 架飞机提升为高优先级。
    或者在 target_flights 飞机出场时将其提升为高优先级。

    冷却期结束后自动恢复原始优先级，模拟急救/VIP任务结束。
    """
    trigger_step: int
    cooldown: int = 200    # 高优先级维持步数（约10分钟 @3s/step），之后恢复
    n_boost: int  = 2      # 每次突变提升的飞机数量
    target_flights: list[int] | None = None # 固定指定的要提升为VIP的航班索引


@dataclass
class ClosureEvent:
    """单次滑行道关闭事件：在 trigger_step 步关闭 edge_nodes 指定的边，持续 duration 步后自动恢复。

    edge_nodes: (u, v) 节点对，关闭 u->v 及其反向边 v->u。
    若为 None，则随机选取一条空闲边关闭。
    """
    trigger_step: int
    duration: int
    edge_nodes: tuple[int, int] | None = None   # None = 随机


# =========================================================
# 场景注入器
# =========================================================

class ScenarioInjector:
    """在仿真循环中按预设计划注入扰动事件，完全替代训练阶段的概率触发机制。

    特性：
    - 测试前将底层环境三类概率归零（不影响已加载模型的策略权重）
    - 按确定性时刻表逐步触发，保证跨实验可复现性
    - 提供事件日志，供实验后指标分析使用

    Priority 场景说明：
    - 初始化时将所有飞机优先级归一化为 BASE_PRIORITY（= 1）
    - 扰动时将 n_boost 架活跃飞机提升为 BOOST_PRIORITY（= num_agents + 100）
    - 冷却期结束后自动恢复为 BASE_PRIORITY
    - 可通过 event_log 追踪哪些飞机被提升，用于轨迹对比分析
    """

    BASE_PRIORITY  = 1          # 所有飞机的基础优先级（相同等级）
    BOOST_PRIORITY_DELTA = 1000 # 高优先级 = BASE_PRIORITY + BOOST_PRIORITY_DELTA

    def __init__(
        self,
        scenario: PerturbationScenario,
        graph_env: "TaxiGraphEnv",
        num_agents: int,
        total_steps: int = 30000,
        seed: int = 42,
        flights: list | None = None,
        time_scale: float = 3.0,
    ) -> None:
        """
        Args:
            scenario:    扭动场景枚举値。
            graph_env:   底层 TaxiGraphEnv 实例。
            num_agents:  本次仿真的飞机总数（用于动态生成计划）。
            total_steps: 仿真总步数（用于生成均匀分布的触发时刻）。
            seed:        随机种子（保证可复现性）。
            flights:     航班数据列表（Delay场景气象延误模式需要）。
            time_scale:  每仿真步对应的物理秒数（用于将时刻转换为步数）。
        """
        self.scenario = scenario
        self.env = graph_env
        self.num_agents = num_agents
        self.total_steps = total_steps
        self.rng = random.Random(seed)
        self._flights = flights       # 保存航班列表引用（气象延误模式需要）
        self._time_scale = time_scale # 保存时间比例

        # 事件执行日志
        self.event_log: list[dict] = []

        # 将底层环境所有概率清零，防止训练期随机事件污染测试评估
        self._disable_random_perturbations()

        # 生成确定性事件计划
        self.delay_plan: list[DelayEvent] = []
        self.priority_plan: list[PriorityEvent] = []
        self.closure_plan: list[ClosureEvent] = []

        if scenario == PerturbationScenario.DELAY:
            self.delay_plan = self._build_delay_plan(total_steps, num_agents)
        elif scenario == PerturbationScenario.PRIORITY:
            self.priority_plan = self._build_priority_plan(total_steps)
            # Priority 场景：将所有飞机优先级归一化为同一基准値
            self._normalize_priorities()
        elif scenario == PerturbationScenario.CLOSURE:
            self.closure_plan = self._build_closure_plan(total_steps)
        # BASELINE: 所有计划均为空列表，不注入任何扭动

        # 内部状态
        self._triggered_delays: set[int] = set()    # 已触发的 delay 事件索引
        self._triggered_priority: set[int] = set()  # 已触发的 priority 事件索引
        self._triggered_closure: set[int] = set()   # 已触发的 closure 事件索引
        self._priority_cooldown: int = 0

        # 优先级恢复状态（Priority场景专用）
        self._boosted_agents: list[int] = []               # 当前被提升优先级的飞机列表
        self._original_priorities: dict[int, int] = {}     # 提升前的原始优先级値

    # ---- 计划生成 --------------------------------------------------------

    def _build_delay_plan(self, total_steps: int, num_agents: int) -> list[DelayEvent]:
        """场景S2：气象洪峰延误，将12:00-13:00间的所有进港航班集中延误至13:30释放。

        设计原则：
        - 代表着猜测天气当天导致地坦完全封闭1小时的极端场景
        - 所有进港航班延误至13:30同时释放，在单一时刻形成流量加贷0（traffic surge）
        - 该场景是对MAPPO和FCFS两者都极具挑战性的闰值测试
        """
        plan = []
        # 气象封地导致延误的起始时刻：13:30对应的事件在 t=0开始同一部解就激活
        # 要在运行时动态计算，所以这里使用一个占位符，实际延误在 _process_delay 中计算
        plan.append(DelayEvent(
            trigger_step=1,  # 与请求正好一起在第1步激活，对所有满足条件的航班一次性延误
            agent_ids=[-2],  # -2 表示“气象延误模式：带12:00-13:00内进港航班延误至13:30”
            duration=0,      # 持续时长测師如下，实际值对应 agent 机计算
        ))
        return plan

    def _build_priority_plan(self, total_steps: int) -> list[PriorityEvent]:
        """场景S3：预先指定特定航班，在它们出场滑行时将其提升为急救/VIP优先级。

        设计原则：
        - 初始所有飞机同一优先级（BASE_PRIORITY=1），不存在预设优先差异
        - 为了保证 FCFS 和 MAPPO 对比时的公平性与微观一致性，必须在两边都提升完全相同的航班
        - 指定 4 个特定航班（分布在不同时段），在它们实际进入滑行道时（start_delays==0）触发提权
        """
        plan = []
        # 固定指定 4 个目标航班，确保它们在复杂流量中出现
        targets_list = [[120], [180], [240], [300]]
        for targets in targets_list:
            plan.append(PriorityEvent(
                trigger_step=-1, # 不再依赖绝对时间步，依靠实际航班出场状态触发
                cooldown=300,    # 维持 15 分钟 VIP 状态
                n_boost=1,
                target_flights=targets,
            ))
        return plan

    def _build_closure_plan(self, total_steps: int) -> list[ClosureEvent]:
        """场景S4：依据热力图分析，关闭全场流量最大的两段主干道，制造显著拓扑扰动。

        热力图分析结果（Baseline场景统计）：
        - 102→103：底层主滑行道中央横段（MAPPO 113次、FCFS 145次），几乎所有横穿机场的飞机必经。
          封路后，底层飞机必须绕行至 y=5 层 202→203，使用路径增加约 2 个拓扑节点。
        - 201→300：左侧主纵向动脉（FCFS 225次），所有从左侧（100/101系列机位）出发要进入
          中间主滑行道区域的飞机唯一通路。封路后必须绕行 101→200→201→... 大弯路。

        注意：两次封路分别在仿真前段（10%）和后段（55%），保证全程均有扰动效果可见。
        """
        plan = [
            ClosureEvent(
                trigger_step=int(total_steps * 0.10),
                duration=800,               # 约 40 分钟（@3s/step），产生实质拓扑冲击
                edge_nodes=(102, 103),      # 底层主滑行道中央横段（热力图 Top-2 高流量边）
            ),
            ClosureEvent(
                trigger_step=int(total_steps * 0.55),
                duration=800,               # 同等强度第二次关闭
                edge_nodes=(304, 305),      # 左侧纵向主动脉（热力图 Top-1 高流量边）
            ),
        ]

        return plan


    # ---- 每步调用 --------------------------------------------------------

    def step(self, current_step: int) -> None:
        """在仿真主循环每步开始前调用，按计划注入本步应触发的事件。

        Args:
            current_step: 当前仿真步数（从 0 开始）。
        """
        if self.scenario == PerturbationScenario.DELAY:
            self._process_delay(current_step)
        elif self.scenario == PerturbationScenario.PRIORITY:
            self._process_priority(current_step)
        elif self.scenario == PerturbationScenario.CLOSURE:
            self._process_closure(current_step)

        # 优先级冷却倒计时 + 到期恢复
        if self._priority_cooldown > 0:
            self._priority_cooldown -= 1
            if self._priority_cooldown == 0:
                self._restore_priorities()

    # ---- 延误处理 --------------------------------------------------------

    def _process_delay(self, step: int) -> None:
        """延误扰动：追加 start_delays，推迟停机坪推出时刻。"""
        for idx, event in enumerate(self.delay_plan):
            if idx in self._triggered_delays:
                continue
            if step >= event.trigger_step:
                self._triggered_delays.add(idx)
                pairs = self._resolve_agent_ids_for_delay(
                    event.agent_ids,
                    flights=getattr(self, "_flights", None),
                    time_scale=getattr(self, "_time_scale", 3.0),
                )
                for agent_id, duration in pairs:
                    if agent_id < self.num_agents and not self.env.done[agent_id]:
                        self.env.start_delays[agent_id] += duration
                        self.event_log.append({
                            "step": step,
                            "type": "delay",
                            "agent_id": agent_id,
                            "duration": duration,
                            "new_start_delay": self.env.start_delays[agent_id],
                        })
                        print(f"  [扭动S2] Step {step}: 飞机#{agent_id} 推出延误 +{duration} 步"
                              f"（当前 start_delay={self.env.start_delays[agent_id]}）")


    def _resolve_agent_ids_for_delay(self, agent_ids: list[int], flights=None, time_scale: float = 3.0) -> list[tuple[int, int]]:
        """将 agent_ids 解析为 (agent_id, delay_duration) 列表。

        -1 = 随机选卂1架
        -2 = 气象延误模式：对12:00-13:00的所有进港航班延误至13:30
        """
        result = []
        for aid in agent_ids:
            if aid == -2:
                # 气象延误模式：仅对 11:00-13:00 内【进港(ARR)】航班延误，
                # 在 13:30-14:30 的 60 分钟窗口内按原始 t0 顺序均匀错峰释放。
                # 只延误 ARR 的理由：气象关闭终端区，进港航班被迫空中等待；
                # DEP 航班通常提前 push-back 或改时刻，不属于同一批次冲击。
                # 60 分钟错峰 = 峰值并发约 3-5 架，在 MAPPO 训练分布内。
                if flights is None:
                    return result
                DELAY_START_SEC = 11 * 3600    # 11:00
                DELAY_END_SEC   = 13 * 3600    # 13:00
                RELEASE_SEC     = 13.5 * 3600  # 13:30 开始释放
                STAGGER_SEC     = 60 * 60      # 60 分钟内均匀散开（降低峰值密度）
                base_t = flights[0]["time_sec"]

                # 仅筛选进港航班，按原始 t0 排序
                candidates = []
                for i, f in enumerate(flights):
                    if f["type"] != "ARR":
                        continue  # 只影响进港航班（气象延误的现实语义）
                    t_sec = f["time_sec"]
                    if DELAY_START_SEC <= t_sec <= DELAY_END_SEC:
                        candidates.append((i, f["t0"]))
                candidates.sort(key=lambda x: x[1])  # 按原始入场顺序排

                # 均匀错峰：将 N 架飞机分散到 [RELEASE_SEC, RELEASE_SEC+STAGGER_SEC]
                n_cands = len(candidates)
                for rank, (i, orig_step) in enumerate(candidates):
                    if n_cands > 1:
                        offset_sec = STAGGER_SEC * rank / (n_cands - 1)
                    else:
                        offset_sec = 0
                    staggered_release_step = int((RELEASE_SEC + offset_sec - base_t) / time_scale)
                    extra = max(0, staggered_release_step - orig_step)
                    if extra > 0:
                        result.append((i, extra))

                if result:
                    agents_str = [str(r[0]) for r in result[:5]]
                    print(f"  [扭动S2] 气象延误：{len(result)} 架 ARR 航班延误，"
                          f"在 13:30-14:30 分批错峰释放 (样例: {','.join(agents_str)}...)")
                else:
                    print(f"  [扭动S2] 警告：未找到 11:00-13:00 内的 ARR 航班")
                return result

            elif aid == -1:
                candidates = [
                    i for i in range(self.num_agents)
                    if not self.env.done[i]
                    and self.env.start_delays[i] > 0
                ]
                if candidates:
                    chosen = self.rng.choice(candidates)
                    result.append((chosen, 120))  # 默认延误120步
                else:
                    print(f"  [扭动S2] 警告：当前无等待推出的飞机，跳过本次延误注入")
            else:
                if not self.env.done[aid] and self.env.start_delays[aid] > 0:
                    result.append((aid, 120))
        return result

    # ---- 优先级处理 -------------------------------------------------------

    def _normalize_priorities(self) -> None:
        """将所有飞机优先级归一化为相同基准值（BASE_PRIORITY）。

        在 Priority 场景初始化时调用，确保实验开始时所有飞机处于同等优先级，
        避免初始优先级差异污染扰动效果。
        """
        for i in range(self.num_agents):
            self.env.priority[i] = self.BASE_PRIORITY
        print(f"[ScenarioInjector] Priority 场景：所有飞机优先级归一化为 {self.BASE_PRIORITY}")

    def _process_priority(self, step: int) -> None:
        """优先级突变扰动：阶跃提升特定飞机的优先级。

        论文 3.4.3 节语义：ρ_i^(T_p+) = ρ_i^(T_p-) + Δρ_i
        - 每次提升 n_boost 架活跃飞机，提升幅度 BOOST_PRIORITY_DELTA
        - 冷却期结束后自动恢复原始优先级（BASE_PRIORITY）
        - 通过 event_log 记录被提升的飞机列表，供轨迹对比分析
        """
        if self._priority_cooldown > 0:
            return  # 冷却中，不触发新事件

        for idx, event in enumerate(self.priority_plan):
            if idx in self._triggered_priority:
                continue

            boosted = []
            if event.target_flights is not None:
                # 检查目标航班是否已经出场（start_delays == 0 且未完成）
                active_targets = [
                    f for f in event.target_flights
                    if f < self.num_agents and not self.env.done[f] and self.env.start_delays[f] == 0
                ]
                if not active_targets:
                    continue # 还没出场，等下一步
                self._triggered_priority.add(idx)
                boosted = [f for f in event.target_flights if f < self.num_agents]
            else:
                if step >= event.trigger_step:
                    self._triggered_priority.add(idx)
                    active = [
                        i for i in range(self.num_agents)
                        if not self.env.done[i] and self.env.start_delays[i] == 0
                    ]
                    if not active:
                        print(f"  [扰动S3] Step {step}: 无活跃飞机，跳过优先级突变")
                        continue

                    n_boost = min(event.n_boost, len(active))
                    boosted = self.rng.sample(active, n_boost)
                else:
                    continue

            if boosted:
                # 记录原始优先级并执行阶跃提升
                self._boosted_agents = boosted
                self._original_priorities = {i: self.env.priority[i] for i in boosted}
                high_priority = self.BASE_PRIORITY + self.BOOST_PRIORITY_DELTA
                for i in boosted:
                    self.env.priority[i] = high_priority

                self._priority_cooldown = event.cooldown
                self.event_log.append({
                    "step": step,
                    "type": "priority_boost",
                    "boosted_agents": list(boosted),
                    "boost_value": high_priority,
                    "original_value": self.BASE_PRIORITY,
                    "cooldown": event.cooldown,
                })
                print(f"  [扰动S3] Step {step}: 飞机 {boosted} 优先级提升至 {high_priority}"
                      f"（持续 {event.cooldown} 步）")
                break   # 每步只触发一次

    def _restore_priorities(self) -> None:
        """冷却期结束时恢复被提升飞机的原始优先级。"""
        if not self._boosted_agents:
            return
        for i in self._boosted_agents:
            if i < self.num_agents:
                self.env.priority[i] = self._original_priorities.get(i, self.BASE_PRIORITY)
        print(f"  [扰动S3] 优先级恢复：飞机 {self._boosted_agents} → {self.BASE_PRIORITY}")
        self.event_log.append({
            "step": -1,  # 恢复时 step 未知，标记为 -1
            "type": "priority_restore",
            "restored_agents": list(self._boosted_agents),
        })
        self._boosted_agents = []
        self._original_priorities = {}

    # ---- 关闭处理 --------------------------------------------------------

    def _process_closure(self, step: int) -> None:
        for idx, event in enumerate(self.closure_plan):
            if idx in self._triggered_closure:
                continue
            if step >= event.trigger_step:
                self._triggered_closure.add(idx)
                closed_edges = self._close_edge(event.edge_nodes, event.duration)
                self.event_log.append({
                    "step": step,
                    "type": "closure",
                    "edge_nodes": event.edge_nodes,
                    "closed_edge_indices": closed_edges,
                    "duration": event.duration,
                })
                print(f"  [扰动S4] Step {step}: 关闭边 {event.edge_nodes}，持续 {event.duration} 步")

    def _close_edge(
        self,
        edge_nodes: tuple[int, int] | None,
        duration: int,
    ) -> list[int]:
        """关闭指定节点对对应的边（双向），并注册到 closure_timers。"""
        closed = []
        for eidx, e in enumerate(self.env.edges):
            if edge_nodes is not None:
                u, v = edge_nodes
                if (e.start == u and e.end == v) or (e.start == v and e.end == u):
                    if not e.closed:
                        e.closed = True
                        self.env.closure_timers[eidx] = duration
                        closed.append(eidx)
            else:
                # 随机选取：找到第一条未关闭且未被占用的边
                open_edges = [
                    i for i, ed in enumerate(self.env.edges)
                    if not ed.closed and not getattr(ed, "occupied", False)
                ]
                if open_edges:
                    chosen = self.rng.choice(open_edges)
                    self.env.edges[chosen].closed = True
                    self.env.closure_timers[chosen] = duration
                    closed.append(chosen)
                break   # 随机模式只选一条
        return closed

    # ---- 禁用概率扰动 ----------------------------------------------------

    def _disable_random_perturbations(self) -> None:
        """将底层环境的三类随机扰动概率全部设为 0，避免与预设场景叠加干扰。"""
        self.env.close_prob = 0.0
        self.env.delay_prob = 0.0
        self.env.priority_prob = 0.0
        print(f"[ScenarioInjector] 场景={self.scenario.value}：已禁用所有随机扰动概率。")

    # ---- 查询接口（供外部追踪使用）---------------------------------------

    def get_boosted_agents(self) -> list[int]:
        """返回当前处于高优先级状态的飞机列表（Priority 场景专用）。"""
        return list(self._boosted_agents)

    def is_high_priority(self, agent_id: int) -> bool:
        """判断指定飞机当前是否处于高优先级状态（Priority 场景专用）。"""
        return agent_id in self._boosted_agents

    # ---- 统计报告 --------------------------------------------------------

    def summary(self) -> None:
        """打印本次仿真的扰动事件统计。"""
        print(f"\n[扰动事件统计] 场景: {self.scenario.value}")
        if not self.event_log:
            print("  (无扰动事件触发)")
            return
        for ev in self.event_log:
            if ev["type"] == "delay":
                print(f"  Step {ev['step']:5d} | 延误  | 飞机#{ev['agent_id']} +{ev['duration']} 步"
                      f" → start_delay={ev['new_start_delay']}")
            elif ev["type"] == "priority_boost":
                print(f"  Step {ev['step']:5d} | 优先级| 飞机{ev['boosted_agents']} 提升至"
                      f" {ev['boost_value']}，持续 {ev['cooldown']} 步")
            elif ev["type"] == "priority_restore":
                print(f"  Step {'(恢复)':>5} | 优先级| 飞机{ev['restored_agents']} 恢复基础优先级")
            elif ev["type"] == "closure":
                print(f"  Step {ev['step']:5d} | 关闭  | 边 {ev['edge_nodes']} 关闭 {ev['duration']} 步")
