from __future__ import annotations
import random, math, heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


# =========================================================
# 基础结构
# =========================================================

class NodeType(Enum):
    GATE="gate"
    INTERSECTION="intersection"


@dataclass
class Node:
    id:int
    position:tuple[float,float]
    type:NodeType


@dataclass
class Edge:
    start:int
    end:int
    length:float
    direction:float
    occupied:bool=False
    closed:bool=False


# =========================================================
# 廊道预留机制（Corridor Reservation Mechanism）
# 防止对向飞机同时进入单行道/走廊造成永久死锁
# =========================================================

class Corridor:
    """表示一条单行走廊（一段节点序列，不允许对向同时占用）。"""

    FORWARD  = "forward"
    REVERSE  = "reverse"

    def __init__(self, cid: int, nodes_seq: list[int]) -> None:
        """
        Args:
            cid: 走廊 ID
            nodes_seq: 走廊节点序列（按 Forward 方向排列）
        """
        self.id = cid
        self.nodes_seq = nodes_seq  # e.g. [5, 4, 3, 2]

        # forward edges: 5→4→3→2  (即 nodes_seq[i]→nodes_seq[i+1])
        self.edges_forward: set[tuple[int,int]] = {
            (nodes_seq[i], nodes_seq[i+1])
            for i in range(len(nodes_seq)-1)
        }
        # reverse edges: 2→3→4→5
        self.edges_reverse: set[tuple[int,int]] = {
            (nodes_seq[i+1], nodes_seq[i])
            for i in range(len(nodes_seq)-1)
        }

        self.direction: str | None = None   # None / FORWARD / REVERSE
        self.occupied_by: set[int] = set()  # agent ids

    def edge_direction(self, u: int, v: int) -> str | None:
        """返回边 (u,v) 对应的走廊方向，不属于此走廊返回 None。"""
        if (u, v) in self.edges_forward:
            return self.FORWARD
        if (u, v) in self.edges_reverse:
            return self.REVERSE
        return None


class CorridorManager:
    """管理所有走廊的占用状态，提供进入/离开协议。"""

    def __init__(self, corridors: list[Corridor]) -> None:
        self.corridors = corridors
        # 快速查找：(start,end) -> Corridor
        self._edge_to_corridor: dict[tuple[int,int], Corridor] = {}
        for c in corridors:
            for e in c.edges_forward | c.edges_reverse:
                self._edge_to_corridor[e] = c

        # 预计算：走廊内部节点 -> {corridor, forward_next, reverse_next}
        self._interior_nodes: dict[int, dict] = {}
        for c in corridors:
            seq = c.nodes_seq
            for idx in range(1, len(seq) - 1):
                nid = seq[idx]
                self._interior_nodes[nid] = {
                    "corridor": c,
                    "forward_next": seq[idx + 1],
                    "reverse_next": seq[idx - 1],
                }

    def is_inside_corridor(self, node_id: int) -> bool:
        """节点是否在走廊内部（非端点）。"""
        return node_id in self._interior_nodes

    # ---- 查询 ------------------------------------------------

    def get_corridor(self, u: int, v: int) -> Corridor | None:
        """返回边 (u,v) 所属的走廊，若不属于任何走廊返回 None。"""
        return self._edge_to_corridor.get((u, v))

    def can_enter(self, corridor: Corridor, agent_dir: str, agent_id: int | None = None) -> bool:
        """检查飞机是否可以进入(或续行)走廊。

        已在走廊内的飞机无条件允许继续前进，避免卡死在走廊中间。
        新进入的飞机需方向匹配。
        """
        # 已在走廊内 → 无条件放行（不能把人堵在中间）
        if agent_id is not None and agent_id in corridor.occupied_by:
            return True
        # 走廊空闲 → 放行
        if corridor.direction is None:
            return True
        # 方向匹配 → 放行
        return corridor.direction == agent_dir


    # ---- 进入 ------------------------------------------------

    def enter(self, corridor: Corridor, agent_id: int, agent_dir: str) -> None:
        """登记飞机进入走廊。"""
        if corridor.direction is None:
            corridor.direction = agent_dir
        corridor.occupied_by.add(agent_id)

    # ---- 离开 ------------------------------------------------

    def leave(self, corridor: Corridor, agent_id: int) -> None:
        """登记飞机离开走廊，若走廊清空则重置方向。"""
        corridor.occupied_by.discard(agent_id)
        if not corridor.occupied_by:
            corridor.direction = None

    def leave_all(self, agent_id: int) -> None:
        """飞机完成/崩溃时，从所有走廊中移除其占用记录。"""
        for c in self.corridors:
            if agent_id in c.occupied_by:
                self.leave(c, agent_id)

    # ---- 按布局手动定义走廊 ------------------------------------

    @staticmethod
    def build_for_layout(layout: str) -> "CorridorManager":
        """按布局手动定义走廊。

        自动检测在度数≥3的稠密图中失效，因此对每种布局
        显式列出需要保护的窄通道节点序列。

        Args:
            layout: 布局名, 如 "D", "A", "B" 等。

        Returns:
            CorridorManager 实例。
        """
        corridor_defs: list[list[int]] = []

        if layout == "D":
            corridor_defs = [
                # 走廊1: 北侧东西横廊
                [900, 111, 112, 113, 114, 115, 116, 901],
                # 走廊2: 东侧纵廊
                [100, 101, 102, 103, 104, 105],
                # 走廊3: 西侧纵廊
                [120, 121, 122, 123, 124, 125],
            ]
        elif layout == "E":
            corridor_defs = [
                # 上排横廊
                [100, 101, 102, 103, 104, 105],
                # 中排横廊
                [111, 112, 113, 114, 115, 116],
                # 下排横廊
                [120, 121, 122, 123, 124, 125],
            ]
        elif layout == "F":
            corridor_defs = [
                # 南部外环1
                [101, 102, 103, 104],
                # 南部外环2
                [201, 202, 203, 204],
                # 纵向连络线
                [101, 201, 300],
                [102, 202, 304],
                [103, 203, 306],
                [104, 204, 309],
                # Y=10主横路 (横向连通支脉)
                [300, 301, 302, 303, 304],
                [304, 305, 306],
                [306, 307, 308, 309],
                # 远西纵向干道及避散斜线 (合并为单条以防止分段死锁)
                [300, 401, 402, 403, 404, 405, 406, 407, 408, 409],
                # 远东纵向大干道
                [309, 501, 502, 503, 504, 505, 506, 507],
                # 中央港湾区左纵线
                [304, 601, 602, 603, 604, 605],
                # 中央港湾区中轴线
                [305, 611, 612],
                # 中央港湾区右纵线
                [306, 621, 622, 623, 624, 625],
                # 中央港湾区北部连通顶廊
                [605, 631, 632, 612, 633, 634, 625]
            ]

        corridors = [
            Corridor(cid, nodes_seq)
            for cid, nodes_seq in enumerate(corridor_defs)
        ]
        return CorridorManager(corridors)



class TaxiGraphEnv:

    def __init__(
        self,
        nodes:list[Node]|None=None,
        edges:list[Edge]|None=None,
        num_agents:int=4,
        max_steps:int=200,
        close_prob:float=0.01,
        delay_prob:float=0.02,
        priority_prob:float=0.01,
        layout:str="default",
        auto_speed:bool=True,
        disable_corridor:bool=False,
        max_speed:float=2.0,
        min_speed:float=0.0,
        speed_step:float=0.21,
        gate_speed_limit:float=0.15,
        turn_speed_limit:float=1.0,
        turn_brake_buffer:float=0.5,
        enable_anticipatory_turn_braking:bool=True,
        midroute_start_prob:float=0.0,
        random_flow_mix:bool=True,
        entry_stagger_steps:int=120,
        departure_stagger_steps:int=60,
        gate_shadow_mode:str="extended",
        shadow_release_headway_steps:int=0,
        physical_clearance_shield:bool=False,
        physical_clearance_min_distance_sq:float=3.0,
        clearance_risk_observation:bool=False,
        pre_entry_clearance_guard:bool=False,
        approach_reservation_progress:float=0.60,
    ):

        if nodes is None or edges is None:
            nodes,edges=self._build_graph(layout)

        self.nodes={n.id:n for n in nodes}
        self.edges=edges
        self.num_agents=num_agents
        self.max_steps=max_steps
        self.layout=layout  # 保存布局名，供 corridor 定义使用

        self.close_prob=close_prob
        self.delay_prob=delay_prob
        self.priority_prob=priority_prob
        self.auto_speed=auto_speed
        self.midroute_start_prob = midroute_start_prob
        self.random_flow_mix = random_flow_mix
        self.entry_stagger_steps = entry_stagger_steps
        self.departure_stagger_steps = departure_stagger_steps
        self.gate_shadow_mode = str(gate_shadow_mode).lower()
        if self.gate_shadow_mode not in {"extended", "direct", "off"}:
            raise ValueError("gate_shadow_mode must be extended, direct, or off")
        self.shadow_release_headway_steps = max(
            int(shadow_release_headway_steps), 0
        )
        self.physical_clearance_shield = bool(physical_clearance_shield)
        self.physical_clearance_min_distance_sq = max(
            float(physical_clearance_min_distance_sq), 0.01
        )
        self.clearance_risk_observation = bool(clearance_risk_observation)
        self.pre_entry_clearance_guard = bool(pre_entry_clearance_guard)
        self.approach_reservation_progress = min(
            max(float(approach_reservation_progress), 0.0), 0.95
        )
        # ================================================================
        # 物理标定常数（基准边 305→612 = 175 m, 坐标距离 = 30 units）
        # L_scale = 5.833 m/unit; scaled 305-612 distance is 200 m
        # T_step  = V_max_code * L_scale / V_max_real
        #         = 3.0 * 5.833 / 15.52 = 1.127 s/step
        # 形成完整映射关系：
        #   v [m/s] = v_code [unit/step] * L_scale / T_step
        #   a [m/s²] = Δv_code * L_scale / T_step²
        # ================================================================
        self.L_SCALE  = 5.833   # m/unit
        self.T_STEP   = 1.127   # s/step
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_step = speed_step

        # Persistent operational speed controls.
        self.gate_speed_limit = max(float(gate_speed_limit), 0.0)
        self.turn_speed_limit = min(max(float(turn_speed_limit), 0.0), max_speed)
        self.turn_brake_decel = max(float(speed_step), 0.21)
        self.turn_brake_buffer = max(float(turn_brake_buffer), 0.0)
        self.enable_anticipatory_turn_braking = bool(enable_anticipatory_turn_braking)
        self.forced_turn_brake_count = 0

        # 状态记录
        self.pos:list[int]=[]
        self.goal:list[int]=[]
        self.speed:list[float]=[]
        self.edge_speed_cap:list[float|None]=[]
        self.turn_brake_active: list[bool] = []
        # A* routes are retained by the physical environment only for
        # turn-ahead preview; they do not force the policy to select actions.
        self.reference_paths: list[list[int]] = []
        self.on_edge:list[bool]=[]
        self.edge_idx:list[int|None]=[]
        self.progress:list[float]=[]
        self.done:list[bool]=[]
        self.crashed:list[bool]=[]
        self.delayed:list[bool]=[]
        self.priority:list[float]=[]
        self.prev_edge_dir:list[float|None]=[]
        self.came_from:list[int|None]=[]

        self._build_adj()
        self._precompute_all_pairs()

        # 按布局手动定义走廊，构建廊道预留管理器
        self.corridor_mgr: CorridorManager = CorridorManager.build_for_layout(layout)
        
        # [Ablation Study] 是否禁用廊道方向锁定机制
        self.disable_corridor = disable_corridor
        if self.disable_corridor:
            self.corridor_mgr.corridors.clear()
            self.corridor_mgr._edge_to_corridor.clear()
            self.corridor_mgr._interior_nodes.clear()
            print(f"\n[TaxiGraphEnv] 🚨 警告: 廊道保护机制被禁用! 智能体必须依靠纯RL避免死锁。")

        # ── 翼尖净空阴影组预计算（Wingtip Clearance Gate Shadow Zone）──────────
        # 物理依据：ICAO Annex 14 Table 3-1
        #   Code E 机型（A330/B787，翼展 52-65m）相邻机位净空要求 ≥ 7.5m
        #   Code F 机型（A380，翼展 65-80m）净空要求 ≥ 10m
        # 实现：机位节点通过 stub_node(≥2000) 连接到父干道节点
        #   同父节点下的机位为直接阴影组（翼尖扫掠必然干涉）
        #   父节点相邻节点下的机位为扩展阴影组（宽体机翼展延伸可能干涉）
        # 触发条件：飞机 i 在机位节点（推出待命），阴影组内有飞机在推出中
        # ─────────────────────────────────────────────────────────────────────
        self.gate_shadow_groups: dict[int, set[int]] = self._build_gate_shadow_groups()
        self.gate_shadow_components = self._build_gate_shadow_components()

        self.reset()



# =========================================================
# 拓扑构建
# =========================================================

    def _build_graph(self, layout:str):
        if layout=="A":
            print("build Layout A")
            return self._build_layout_a()
        elif layout=="B":
            print("build Layout B")
            return self._build_layout_b()
        elif layout=="C":
            print("build Layout c")
            return self._build_layout_c()
        elif layout=="D":
            print("build Layout D")
            return self._build_layout_d()
        elif layout=="E":
            print("build Layout E")
            return self._build_layout_e()
        elif layout=="F":
            print("build Layout F")
            return self._build_layout_f()
        elif layout=="G":
            print("build Layout G")
            return self._build_layout_g()
        else:
            return self._build_default_graph()

    @staticmethod
    def _make_edges(pos, edge_data):
        """通用的边构建工具方法。"""
        edges=[]
        for u,v in edge_data:
            x1,y1=pos[u]; x2,y2=pos[v]
            d=math.hypot(x1-x2,y1-y2)
            ang=math.atan2(y2-y1,x2-x1)
            edges.append(Edge(u,v,d,ang))
            edges.append(Edge(v,u,d,ang+math.pi))
        return edges

    def _build_default_graph(self):
        """原始 29 节点布局。"""
        pos={
            0:(0,0),1:(6,2),2:(6,4),3:(6,6),4:(6,8),
            13:(4,2),14:(4,4),15:(4,6),16:(4,8),
            12:(-6,2),11:(-6,4),10:(-6,6),9:(-6,8),
            24:(-4,2),23:(-4,4),22:(-4,6),21:(-4,8),
            5:(5,12),6:(2,12),7:(-2,12),8:(-5,12),
            17:(2,8),18:(-2,8),19:(0,8),20:(0,2),
            25:(-2,0),26:(2,0),27:(6,0),28:(-6,0)
        }
        gates={1,2,3,4,9,10,11,12,5,27,28}
        nodes=[Node(i,p,NodeType.GATE if i in gates else NodeType.INTERSECTION) for i,p in pos.items()]
        edge_data=[
            (1,13),(2,14),(3,15),(4,16),(9,21),(10,22),(11,23),(12,24),
            (16,5),(5,17),(17,6),(6,19),(19,7),(7,18),(8,18),(21,8),
            (16,17),(17,19),(19,18),(18,21),
            (13,14),(14,15),(15,16),(16,17),
            (24,23),(23,22),(22,21),(21,19),
            (19,20),
            (28,24),(25,24),(25,20),(26,20),(26,13),(27,13),
            (25,26),(26,27),(27,28)
        ]
        return nodes, self._make_edges(pos, edge_data)

    @staticmethod
    def _build_layout_a():
        """布局A：严格独立分支结构。
        
        核心改进：
        1. 每一个机位（Gate）都通过唯一的横向支路连接到主干。
        2. 错开所有机位的 Y 坐标，确保 9 和 37、10 和 36 等机位不再处于同一条线上。
        3. 所有 Gate 节点的度数严格为 1。
        """
        pos = {
            # ---- 中央主干道 (Backbone) ----
            0:(0,0), 20:(0,2), 19:(0,8),
            17:(2,8), 18:(-2,8),
            25:(-2,0), 26:(2,0),

            # ---- 东侧主干连接点 (Intersections) ----
            113:(4,1), 13:(4,2), 114:(4,3), 14:(4,4), 
            115:(4,5), 15:(4,6), 116:(4,7), 16:(4,8),

            # ---- 西侧主干连接点 (Intersections) ----
            124:(-4,1), 24:(-4,2), 123:(-4,3), 23:(-4,4), 
            122:(-4,5), 22:(-4,6), 121:(-4,7), 21:(-4,8),

            # ---- 东侧独立机位 (Gates) ----
            30:(8,1), 1:(6,2),  31:(8,3), 2:(6,4),
            32:(8,5), 3:(6,6),  33:(8,7), 4:(6,8),

            # ---- 西侧独立机位 (Gates) ----
            34:(-8,1), 12:(-6,2), 35:(-8,3), 11:(-6,4),
            36:(-8,5), 10:(-6,6), 37:(-8,7), 9:(-6,8),

            # ---- 北侧与南侧 (同样进行解耦处理) ----
            38:(4,12), 39:(-4,12), 44:(6,12), 45:(-6,12),
            5:(4,10), 8:(-4,10),  # 北侧近端
            40:(6,14), 41:(2,14), 42:(-2,14), 43:(-6,14), # 北侧远端
            46:(4,0), 47:(-4,0), 50:(0,-2),
            48:(6,-2), 49:(-6,-2) # 南侧机位
        }

        # 所有的机位集合
        gates={1,2,3,4,9,10,11,12,30,31,32,33,34,35,36,37,5,8,40,41,42,43,48,49}

        nodes=[
            Node(i,p,NodeType.GATE if i in gates else NodeType.INTERSECTION)
            for i,p in pos.items()
        ]

        edge_data=[
            # ---- 东侧：每一行只接一个机位 ----
            (113,13),(13,114),(114,14),(14,115),(115,15),(15,116),(116,16), # 主干连线
            (30,113), (1,13), (31,114), (2,14), (32,115), (3,15), (33,116), (4,16),

            # ---- 西侧：每一行只接一个机位 ----
            (124,24),(24,123),(23,122),(22,121),(21,121), # 主干连线 (部分简写)
            (124,123),(123,23),(23,122),(122,22),(121,21),
            (34,124), (12,24), (35,123), (11,23), (36,122), (10,22), (37,121), (9,21),

            # ---- 北侧与南侧主干 ----
            (16,17),(17,19),(19,18),(18,21),
            (16,38),(38,44),(21,39),(39,45),
            (5,16),(8,21), # 近端机位直连主干端点
            (40,44),(41,44),(42,45),(43,45), # 北侧远端
            (26,46),(25,47),(46,50),(47,50),(48,50),(49,50),
            
            (47,124),(113,46),(39,42),(38,41)

        ]

        edges=[]
        for u,v in edge_data:
            x1,y1=pos[u]; x2,y2=pos[v]
            d=math.hypot(x1-x2,y1-y2)
            ang=math.atan2(y2-y1,x2-x1)
            edges.append(Edge(u,v,d,ang))
            edges.append(Edge(v,u,d,ang+math.pi))

        return nodes, edges
    @staticmethod
    def _build_layout_b():
        """布局B：环形+放射状（60节点，约26个机位）。

        以中央枢纽为核心，4条放射通道连接东南西北四个机位区，
        外围有环形滑行道可绕行。
        """
        pos={
            # ---- 中央枢纽 ----
            0:(0,0),                                   # 核心中心
            100:(0,2),101:(2,0),102:(0,-2),103:(-2,0), # 内环4向

            # ---- 北区 (放射向上) ----
            # 主通道
            110:(0,4),111:(0,6),112:(0,8),113:(0,10),
            # 东侧机位
            120:(2,5),121:(2,7),122:(2,9),123:(2,11),
            # 西侧机位
            130:(-2,5),131:(-2,7),132:(-2,9),133:(-2,11),

            # ---- 东区 (放射向右) ----
            140:(4,0),141:(6,0),142:(8,0),143:(10,0),
            150:(5,2),151:(7,2),152:(9,2),
            160:(5,-2),161:(7,-2),162:(9,-2),

            # ---- 南区 (放射向下) ----
            170:(0,-4),171:(0,-6),172:(0,-8),173:(0,-10),
            180:(2,-5),181:(2,-7),182:(2,-9),
            190:(-2,-5),191:(-2,-7),192:(-2,-9),

            # ---- 西区 (放射向左) ----
            200:(-4,0),201:(-6,0),202:(-8,0),203:(-10,0),
            210:(-5,2),211:(-7,2),212:(-9,2),
            220:(-5,-2),221:(-7,-2),222:(-9,-2),

            # ---- 外围环形连接 ----
            250:(8,8),251:(-8,8),252:(-8,-8),253:(8,-8),
        }

        gates={
            # 各区远端机位
            122,123,132,133,   # 北区 4
            151,152,161,162,   # 东区 4
            181,182,191,192,   # 南区 4
            211,212,221,222,   # 西区 4
            # 角落机位
            250,251,252,253,   # 外围 4
            # 末端机位
            120,121,130,131,   # 北区近端 4
            143,173,203,       # 远端终点 3
        }

        nodes=[
            Node(i,p,NodeType.GATE if i in gates else NodeType.INTERSECTION)
            for i,p in pos.items()
        ]

        edge_data=[
            # ---- 内环 ----
            (0,100),(0,101),(0,102),(0,103),
            (100,101),(101,102),(102,103),(103,100),

            # ---- 北区放射 ----
            (100,110),(110,111),(111,112),(112,113),
            (110,120),(111,121),(112,122),(113,123),
            (110,130),(111,131),(112,132),(113,133),
            (120,121),(121,122),(122,123),
            (130,131),(131,132),(132,133),

            # ---- 东区放射 ----
            (101,140),(140,141),(141,142),(142,143),
            (140,150),(141,151),(142,152),
            (140,160),(141,161),(142,162),
            (150,151),(151,152),
            (160,161),(161,162),

            # ---- 南区放射 ----
            (102,170),(170,171),(171,172),(172,173),
            (170,180),(171,181),(172,182),
            (170,190),(171,191),(172,192),
            (180,181),(181,182),
            (190,191),(191,192),

            # ---- 西区放射 ----
            (103,200),(200,201),(201,202),(202,203),
            (200,210),(201,211),(202,212),
            (200,220),(201,221),(202,222),
            (210,211),(211,212),
            (220,221),(221,222),

            # ---- 外围环形 ----
            (113,250),(250,143),
            (113,251),(251,203),
            (173,252),(252,203),
            (173,253),(253,143),
            (250,251),(251,252),(252,253),(253,250),
        ]

        edges=[]
        for u,v in edge_data:
            x1,y1=pos[u]; x2,y2=pos[v]
            d=math.hypot(x1-x2,y1-y2)
            ang=math.atan2(y2-y1,x2-x1)
            edges.append(Edge(u,v,d,ang))
            edges.append(Edge(v,u,d,ang+math.pi))

        return nodes, edges
    @staticmethod
    def _build_layout_c():
        import math
        pos = {}
        edge_data = []
        
        # ---- 1. 定义外围框架节点 (Backbone) ----
        # 左斜线与入口
        pos[121] = (-15, 20) # 跑道入口点
        pos[120] = (-12, 16)
        # 左侧垂线
        pos[100] = (-10, 12); pos[101] = (-10, 8); pos[102] = (-10, 4); pos[103] = (-10, 0)
        # 底部横线与凹型转角
        pos[130] = (-5, 0); pos[140] = (-5, 6); pos[141] = (0, 6); pos[142] = (5, 6); pos[143] = (5, 0)
        # 右侧横线与垂线出口
        pos[150] = (10, 0); pos[160] = (10, 5); pos[161] = (10, 10); pos[162] = (10, 15); pos[163] = (10, 20)
        pos[190] = (10, 24) # 跑道出口点

        # --- 连接框架主干 (确保无断点链条) ---
        main_chain = [121, 120, 100, 101, 102, 103, 130, 140, 141, 142, 143, 150, 160, 161, 162, 163, 190]
        for i in range(len(main_chain) - 1):
            edge_data.append((main_chain[i], main_chain[i+1]))

        # ---- 2. 生成30个机位 (ID: 0-29) ----
        # 均匀分配在各个主干节点附近
        gate_count = 0
        def attach_gates(parent_id, offsets):
            nonlocal gate_count
            for dx, dy in offsets:
                if gate_count >= 30: break
                g_id = gate_count
                e_id = 2000 + gate_count # 接入点
                pos[g_id] = (pos[parent_id][0] + dx, pos[parent_id][1] + dy)
                # 接入点位于主干和机位中间，防止视觉断点
                pos[e_id] = (pos[parent_id][0] + dx*0.5, pos[parent_id][1] + dy*0.5)
                edge_data.append((parent_id, e_id)) # 主干连接入点
                edge_data.append((e_id, g_id))       # 接入点连机位
                gate_count += 1

        # 根据示意图分配机位偏移 (dx, dy)
        attach_gates(121, [(1, 1)])          # 左斜线
        attach_gates(120, [(1, 1)])
        attach_gates(100, [(2, 0)]); attach_gates(101, [(2, 0)]); attach_gates(102, [(2, 0)]) # 左垂线向内
        attach_gates(130, [(0, 2), (2, 2)])  # 底部左
        attach_gates(140, [(2, 0), (2, -2)]) # 凹型左内侧
        attach_gates(141, [(-1, -2), (1, -2)]) # 凹型顶内侧
        attach_gates(142, [(-2, -2), (-2, 0)]) # 凹型右内侧
        attach_gates(150, [(-2, 2), (0, 2)])  # 底部右
        # 右垂线向内
        for pid in [160, 161, 162, 163, 190]:
            attach_gates(pid, [(-2, 0), (-2, 1)])

        # ---- 3. 构建节点与双向边 (解决可视化缺失问题) ----
        nodes = []
        for i, p in pos.items():
            ntype = NodeType.GATE if i < 30 else NodeType.INTERSECTION
            nodes.append(Node(i, p, ntype))
        
        edges = []
        for u, v in edge_data:
            # 确保 u, v 都在 pos 中
            if u in pos and v in pos:
                x1, y1 = pos[u]; x2, y2 = pos[v]
                d = math.hypot(x1-x2, y1-y2)
                ang = math.atan2(y2-y1, x2-x1)
                edges.append(Edge(u, v, d, ang))
                # 必须添加反向边，否则可视化和路径规划会断路
                edges.append(Edge(v, u, d, ang + math.pi))

        return nodes, edges
    @staticmethod
    def _build_layout_d():

        import math

        pos = {}
        edge_data = []

        # ===============================
        # 1 Taxiway nodes
        # ===============================

        pos[100] = (-10,15)
        pos[101] = (-6,15)
        pos[102] = (-2,15)
        pos[103] = (2,15)
        pos[104] = (6,15)
        pos[105] = (10,15)

        pos[111] = (-10,10)
        pos[112] = (-6,10)
        pos[113] = (-2,10)
        pos[114] = (2,10)
        pos[115] = (6,10)
        pos[116] = (10,10)

        pos[120] = (-10,5)
        pos[121] = (-6,5)
        pos[122] = (-2,5)
        pos[123] = (2,5)
        pos[124] = (6,5)
        pos[125] = (10,5)

        # ===============================
        # 2 Horizontal taxiway edges
        # ===============================

        edge_data += [
            (100,101),(101,102),(102,103),(103,104),(104,105),
            (111,112),(112,113),(113,114),(114,115),(115,116),
            (120,121),(121,122),(122,123),(123,124),(124,125)
        ]

        # ===============================
        # 3 Vertical taxiway edges
        # ===============================

        edge_data += [
            (100,111),(101,112),(102,113),(103,114),(104,115),(105,116),
            (111,120),(112,121),(113,122),(114,123),(115,124),(116,125)
        ]

        # ===============================
        # 4 Gate stub nodes
        # ===============================

# ===== 上排 gate =====

        pos[2000]=(-11,18); pos[0]=(-13,21)
        pos[2001]=(-7,18);  pos[1]=(-7,21)
        pos[2002]=(-3,18);  pos[2]=(-3,21)
        pos[2003]=(3,18);   pos[3]=(3,21)
        pos[2004]=(7,18);   pos[4]=(7,21)
        pos[2005]=(11,18);  pos[5]=(13,21)

        # ===== 下排 gate =====

        pos[2006]=(-11,2); pos[6]=(-13,-1)
        pos[2007]=(-7,2);  pos[7]=(-7,-1)
        pos[2008]=(-3,2);  pos[8]=(-3,-1)
        pos[2009]=(3,2);   pos[9]=(3,-1)
        pos[2010]=(7,2);   pos[10]=(7,-1)
        pos[2011]=(11,2);  pos[11]=(13,-1)

        # ===== 中排上侧 =====

        pos[2012]=(-11,12); pos[12]=(-13,14)
        pos[2013]=(-7,12);  pos[13]=(-7,14)
        pos[2014]=(-3,12);  pos[14]=(-3,14)
        pos[2015]=(3,12);   pos[15]=(3,14)
        pos[2016]=(7,12);   pos[16]=(7,14)
        pos[2017]=(11,12);  pos[17]=(13,14)

        # ===== 中排下侧 =====

        pos[2018]=(-11,8); pos[18]=(-13,6)
        pos[2019]=(-7,8);  pos[19]=(-7,6)
        pos[2020]=(-3,8);  pos[20]=(-3,6)
        pos[2021]=(3,8);   pos[21]=(3,6)
        pos[2022]=(7,8);   pos[22]=(7,6)
        pos[2023]=(11,8);  pos[23]=(13,6)

        # ===== 上排内侧备用 =====

        pos[2024]=(-9,17); pos[24]=(-9,19)
        pos[2025]=(-5,17); pos[25]=(-5,19)
        pos[2026]=(-1,17); pos[26]=(-1,19)
        pos[2027]=(1,17);  pos[27]=(1,19)
        pos[2028]=(5,17);  pos[28]=(5,19)
        pos[2029]=(9,17);  pos[29]=(9,19)

        # ===============================
        # 5 Gate edges
        # ===============================

        edge_data += [
            (100,2000),(2000,0),
            (101,2001),(2001,1),
            (102,2002),(2002,2),
            (103,2003),(2003,3),
            (104,2004),(2004,4),
            (105,2005),(2005,5),

            (120,2006),(2006,6),
            (121,2007),(2007,7),
            (122,2008),(2008,8),
            (123,2009),(2009,9),
            (124,2010),(2010,10),
            (125,2011),(2011,11),

            (111,2012),(2012,12),
            (112,2013),(2013,13),
            (113,2014),(2014,14),
            (114,2015),(2015,15),
            (115,2016),(2016,16),
            (116,2017),(2017,17),

            (111,2018),(2018,18),
            (112,2019),(2019,19),
            (113,2020),(2020,20),
            (114,2021),(2021,21),
            (115,2022),(2022,22),
            (116,2023),(2023,23),

            (100,2024),(2024,24),
            (101,2025),(2025,25),
            (102,2026),(2026,26),
            (103,2027),(2027,27),
            (104,2028),(2028,28),
            (105,2029),(2029,29)
        ]

        # ===============================
        # 6 Runway connection
        # ===============================

        ENTRY = 900
        EXIT = 901

        pos[ENTRY]=(-14,10)
        pos[EXIT]=(14,10)

        edge_data += [
            (ENTRY,111),
            (116,EXIT)
        ]

        # ===============================
        # 7 Node objects
        # ===============================

        nodes=[]

        for i,p in pos.items():
            if i<30:
                t=NodeType.GATE
            else:
                t=NodeType.INTERSECTION
            nodes.append(Node(i,p,t))

        # ===============================
        # 8 Edge objects
        # ===============================

        edges=[]

        for u,v in edge_data:

            x1,y1=pos[u]
            x2,y2=pos[v]

            d=math.hypot(x1-x2,y1-y2)
            ang=math.atan2(y2-y1,x2-x1)

            edges.append(Edge(u,v,d,ang))
            edges.append(Edge(v,u,d,ang+math.pi))

        return nodes,edges
    @staticmethod
    #纯对称、少机位
    def _build_layout_e():
        import math

        pos = {}
        edge_data = []

        # ============================================================
        # 1 Taxiway nodes (Main Grid: Top, Middle, Bottom)
        # ============================================================
        # Top line (y=15)
        pos[100] = (-10, 15); pos[101] = (-6, 15); pos[102] = (-2, 15)
        pos[103] = (2, 15); pos[104] = (6, 15); pos[105] = (10, 15)

        # Middle line (y=10)
        pos[111] = (-10, 10); pos[112] = (-6, 10); pos[113] = (-2, 10)
        pos[114] = (2, 10); pos[115] = (6, 10); pos[116] = (10, 10)

        # Bottom line (y=5)
        pos[120] = (-10, 5); pos[121] = (-6, 5); pos[122] = (-2, 5)
        pos[123] = (2, 5); pos[124] = (6, 5); pos[125] = (10, 5)

        # ============================================================
        # 2 Horizontal taxiway edges
        # ============================================================
        edge_data += [
            (100, 101), (101, 102), (102, 103), (103, 104), (104, 105),
            (111, 112), (112, 113), (113, 114), (114, 115), (115, 116),
            (120, 121), (121, 122), (122, 123), (123, 124), (124, 125)
        ]

        # ============================================================
        # 3 Vertical taxiway edges
        # ============================================================
        edge_data += [
            (100, 111), (101, 112), (102, 113), (103, 114), (104, 115), (105, 116),
            (111, 120), (112, 121), (113, 122), (114, 123), (115, 124), (116, 125)
        ]

        # ============================================================
        # 4 Gate stub nodes (0-23 remaining)
        # ============================================================
        # ===== 上排 gate (0-5) =====
        pos[2000] = (-11, 18); pos[0] = (-13, 21)
        pos[2001] = (-7, 18);  pos[1] = (-7, 21)
        pos[2002] = (-3, 18);  pos[2] = (-3, 21)
        pos[2003] = (3, 18);   pos[3] = (3, 21)
        pos[2004] = (7, 18);   pos[4] = (7, 21)
        pos[2005] = (11, 18);  pos[5] = (13, 21)

        # ===== 下排 gate (6-11) =====
        pos[2006] = (-11, 2); pos[6] = (-13, -1)
        pos[2007] = (-7, 2);  pos[7] = (-7, -1)
        pos[2008] = (-3, 2);  pos[8] = (-3, -1)
        pos[2009] = (3, 2);   pos[9] = (3, -1)
        pos[2010] = (7, 2);   pos[10] = (7, -1)
        pos[2011] = (11, 2);  pos[11] = (13, -1)

        # ===== 中排上侧機位 (17保留) =====
        pos[2017] = (11, 12);  pos[17] = (13, 14)

        # ===== 中排下侧機位 (23保留) =====
        pos[2023] = (11, 8);  pos[23] = (13, 6)

        # ============================================================
        # 5 Gate edges (Only 0-23 connections)
        # ============================================================
        edge_data += [
            (100, 2000), (2000, 0), (101, 2001), (2001, 1),
            (102, 2002), (2002, 2), (103, 2003), (2003, 3),
            (104, 2004), (2004, 4), (105, 2005), (2005, 5),

            (120, 2006), (2006, 6), (121, 2007), (2007, 7),
            (122, 2008), (2008, 8), (123, 2009), (2009, 9),
            (124, 2010), (2010, 10), (125, 2011), (2011, 11),

            (116, 2017), (2017, 17),

            (116, 2023), (2023, 23)
        ]

        # ============================================================
        # 6 Runway connection
        # ============================================================
        ENTRY = 900
        EXIT = 901
        pos[ENTRY] = (-14, 10)
        pos[EXIT] = (14, 10)

        edge_data += [
            (ENTRY, 111),
            (116, EXIT)
        ]

        # ============================================================
        # 7 Node objects
        # ============================================================
        nodes = []
        for i, p in pos.items():
            if i < 30:
                t = NodeType.GATE
            else:
                t = NodeType.INTERSECTION
            nodes.append(Node(i, p, t))

        # ============================================================
        # 8 Edge objects
        # ============================================================
        edges = []
        for u, v in edge_data:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            d = math.hypot(x1-x2, y1-y2)
            ang = math.atan2(y2-y1, x2-x1)
            edges.append(Edge(u, v, d, ang))
            edges.append(Edge(v, u, d, ang + math.pi))

        return nodes, edges

    @staticmethod
    def _build_layout_f():
        """构建新型 Layout F:
        拥有完整外围网络，U型(C型)港湾翻转了90度，包含30个机位。
        """
        import math
        pos = {}
        edge_data = []

        # ============================================================
        # 1 Base Lines and Structure
        # ============================================================
        y_b1 = 0; y_b2 = 5; y_mid_bot = 10; y_apron_top = 34
        x_fl = -30; x_al = -12; x_ac = 0; x_ar = 12; x_fr = 30

        # Bypass 1 (南部最外环)
        pos[101] = (x_fl, y_b1); pos[102] = (x_al, y_b1); pos[103] = (x_ar, y_b1); pos[104] = (x_fr, y_b1)
        pos[100] = (x_fl - 5, y_b1); pos[105] = (x_fr + 5, y_b1)
        
        # Bypass 2 (南部次外环)
        pos[201] = (x_fl, y_b2); pos[202] = (x_al, y_b2); pos[203] = (x_ar, y_b2); pos[204] = (x_fr, y_b2)
        pos[200] = (x_fl - 5, y_b2); pos[205] = (x_fr + 5, y_b2)

        # Y = 10 line (连接中心与两侧)
        pos[300] = (x_fl, y_mid_bot) 
        pos[301] = (-27, y_mid_bot); pos[302] = (-20, y_mid_bot); pos[303] = (-16, y_mid_bot) # 3 on left
        pos[304] = (x_al, y_mid_bot) 
        pos[305] = (x_ac, y_mid_bot) 
        pos[306] = (x_ar, y_mid_bot) 
        pos[307] = (18, y_mid_bot); pos[308] = (24, y_mid_bot) # 2 on right
        pos[309] = (x_fr, y_mid_bot)

        # Far Left Route (远西干道)
        pos[401] = (x_fl, 16); pos[402] = (x_fl, 22); pos[403] = (x_fl, 28); pos[404] = (x_fl, 34)
        pos[405] = (-26, 38); pos[406] = (-22, 42); pos[407] = (-18, 46); pos[408] = (-14, 50); pos[409] = (-10, 54)

        # Far Right Route (远东干道)
        pos[501] = (x_fr, 16); pos[502] = (x_fr, 22); pos[503] = (x_fr, 28); pos[504] = (x_fr, 34)
        pos[505] = (x_fr, 40); pos[506] = (x_fr, 46); pos[507] = (x_fr, 52)

        # Central Apron (中心翻转区)
        pos[601] = (x_al, 16); pos[602] = (x_al, 22); pos[603] = (x_al, 28); pos[604] = (x_al, 34)
        pos[605] = (x_al, 40)
        pos[611] = (x_ac, 25); pos[612] = (x_ac, 40)
        pos[621] = (x_ar, 16); pos[622] = (x_ar, 22); pos[623] = (x_ar, 28); pos[624] = (x_ar, 34)
        pos[625] = (x_ar, 40)
        pos[631] = (-8, 40); pos[632] = (-4, 40); pos[633] = (4, 40); pos[634] = (8, 40)

        # Edges
        edge_data += [
            (100, 101), (101, 102), (102, 103), (103, 104), (104, 105),
            (200, 201), (201, 202), (202, 203), (203, 204), (204, 205),
            (101, 201), (102, 202), (103, 203), (104, 204),
            (201, 300), (202, 304), (203, 306), (204, 309),
            
            (300, 301), (301, 302), (302, 303), (303, 304),
            (304, 305), (305, 306),
            (306, 307), (307, 308), (308, 309),

            (300, 401), (401, 402), (402, 403), (403, 404),
            (404, 405), (405, 406), (406, 407), (407, 408), (408, 409),

            (309, 501), (501, 502), (502, 503), (503, 504), (504, 505), (505, 506), (506, 507),

            (304, 601), (601, 602), (602, 603), (603, 604), (604, 605),
            (305, 611), (611, 612),
            (306, 621), (621, 622), (622, 623), (623, 624), (624, 625),
            (605, 631), (631, 632), (632, 612), (612, 633), (633, 634), (634, 625)
        ]

        # 30 Gates 分布挂载
        # Proportional network enlargement: edge 102-103 becomes 160 m.
        # The original coordinate distance is 24 model units.
        NETWORK_SCALE = 8.0 / 7.0
        for node_id in list(pos):
            x, y = pos[node_id]
            pos[node_id] = (x * NETWORK_SCALE, y * NETWORK_SCALE)
        # Move the two side stand groups upward while keeping their original
        # side orientation. The upper nodes remain below 605 and 625.
        side_shift = 2.5
        side_groups = [601, 602, 603, 604, 621, 622, 623, 624]
        for node_id in side_groups:
            x, y = pos[node_id]
            pos[node_id] = (x, y + side_shift)
        gate_id = 0
        def add_gate(parent_node, dx, dy):
            nonlocal gate_id
            stub_id = 2000 + gate_id
            direction_norm = math.hypot(dx, dy)
            ux, uy = dx / direction_norm, dy / direction_norm
            parent_to_stub = 16.0 / 5.833
            stub_to_gate = 32.0 / 5.833
            px, py = pos[parent_node]
            pos[stub_id] = (
                px + ux * parent_to_stub,
                py + uy * parent_to_stub,
            )
            pos[gate_id] = (
                px + ux * (parent_to_stub + stub_to_gate),
                py + uy * (parent_to_stub + stub_to_gate),
            )
            edge_data.append((parent_node, stub_id))
            edge_data.append((stub_id, gate_id))
            gate_id += 1
        # 1. 远西西北折线端共七个 (包含垂直段与斜线段)
        for n in [405, 406, 407, 408]: add_gate(n, -4, 4)
        for n in [401, 402, 403]: add_gate(n, -5, 0)
        # 2. 横线3个 (左横)
        for n in [301, 302, 303]: add_gate(n, 0, 5)
        # 3. 港湾区内 (中心U型翻转共计12个)
        for n in [601, 602, 603, 604]: add_gate(n, -5, 0)
        for n in [631, 632, 633, 634]: add_gate(n, 0, 5)
        for n in [621, 622, 623, 624]: add_gate(n, 5, 0)
        # 4. 右侧横线2个
        for n in [307, 308]: add_gate(n, 0, 5)
        # 5. 纵向线段上六个 (远东)
        for n in [501, 502, 503, 504, 505, 506]: add_gate(n, 5, 0)

        # Runway Entry/Exit connections for Layout F
        ENTRY = 900
        EXIT = 901
        pos[ENTRY] = (-25 * NETWORK_SCALE, -2 * NETWORK_SCALE)
        pos[EXIT] = (25 * NETWORK_SCALE, -2 * NETWORK_SCALE)
        edge_data += [
            (ENTRY, 101),
            (104, EXIT)
        ]

        nodes = []
        for i, p in pos.items():
            t = NodeType.GATE if i < 30 else NodeType.INTERSECTION
            nodes.append(Node(i, p, t))

        edges = []
        for u, v in edge_data:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            d = math.hypot(x1-x2, y1-y2)
            ang = math.atan2(y2-y1, x2-x1)
            edges.append(Edge(u, v, d, ang))
            edges.append(Edge(v, u, d, ang + math.pi))

        return nodes, edges

    @staticmethod
    def _build_layout_g():
        """Layout G：机位连接段扩展版（基于 Layout F）

        核心要求：
          - 严格保持 305→612 的实际长度 (30 units = 175m)
          - 保持港湾区内部滑行道（如 601-605）的纵向实际物理间距
          - 主要目标：将机位连接段 (Gate stubs) 扩大为原先的 3 倍长 (dx/dy = 15)
        实现方式：
          - 港湾区内部坐标系与 Layout F 保持绝对一致。
          - 将两侧的外围滑行道（Bypass、远东/远西干道）向外推移至 x=±50。
          - 调整横向连络道节点的 X 坐标，以防止被加长后的垂直机位连接段穿透。
        """
        import math
        pos = {}
        edge_data = []

        # ============================================================
        # 1 基准线定义
        # ============================================================
        y_b1 = 0; y_b2 = 5; y_mid_bot = 10
        # 304(-17,10) 到 305(0,10) 为 17 units，对应 17 * 5.833 = 99.16m (约100m)
        x_fl = -60; x_al = -17; x_ac = 0; x_ar = 17; x_fr = 60

        # Bypass 1 (南部最外环)
        pos[101] = (x_fl, y_b1); pos[102] = (x_al, y_b1)
        pos[103] = (x_ar, y_b1); pos[104] = (x_fr, y_b1)
        pos[100] = (x_fl - 5, y_b1); pos[105] = (x_fr + 5, y_b1)

        # Bypass 2 (南部次外环)
        pos[201] = (x_fl, y_b2); pos[202] = (x_al, y_b2)
        pos[203] = (x_ar, y_b2); pos[204] = (x_fr, y_b2)
        pos[200] = (x_fl - 5, y_b2); pos[205] = (x_fr + 5, y_b2)

        # Y=10 横向主干（调整左右节点位置防止机位交叉）
        # 左侧柱(x=-17)的机位向左延伸15unit到 x=-32。因此横向机位需在 x <= -35。
        pos[300] = (x_fl, y_mid_bot)
        pos[301] = (-51, y_mid_bot); pos[302] = (-43, y_mid_bot); pos[303] = (-35, y_mid_bot)
        pos[304] = (x_al, y_mid_bot)
        pos[305] = (x_ac, y_mid_bot)
        pos[306] = (x_ar, y_mid_bot)
        # 右侧柱(x=17)的机位向右延伸15unit到 x=32。因此横向机位需在 x >= 35。
        pos[307] = (35, y_mid_bot); pos[308] = (43, y_mid_bot)
        pos[309] = (x_fr, y_mid_bot)

        # Far Left Route (远西干道, 推至 x=-60)
        pos[401] = (x_fl, 16); pos[402] = (x_fl, 22)
        pos[403] = (x_fl, 28); pos[404] = (x_fl, 34)
        pos[405] = (-54, 40); pos[406] = (-48, 46)
        pos[407] = (-42, 52); pos[408] = (-36, 58); pos[409] = (-30, 64)

        # Far Right Route (远东干道, 推至 x=60)
        pos[501] = (x_fr, 16); pos[502] = (x_fr, 22)
        pos[503] = (x_fr, 28); pos[504] = (x_fr, 34)
        pos[505] = (x_fr, 40); pos[506] = (x_fr, 46); pos[507] = (x_fr, 52)

        # Central Apron U 型港湾（坐标与 Layout F 绝对一致，维持物理原貌）
        pos[601] = (x_al, 16); pos[602] = (x_al, 22)
        pos[603] = (x_al, 28); pos[604] = (x_al, 34); pos[605] = (x_al, 40)

        # 核心基准边：305(0,10) -> 612(0,40)，距离 30 units = 175m 不变！
        pos[611] = (x_ac, 25); pos[612] = (x_ac, 40)

        pos[621] = (x_ar, 16); pos[622] = (x_ar, 22)
        pos[623] = (x_ar, 28); pos[624] = (x_ar, 34); pos[625] = (x_ar, 40)

        # 调整顶部横梁间距使其均匀
        pos[631] = (-10, 40); pos[632] = (-5, 40)
        pos[633] = (5, 40);  pos[634] = (10, 40)

        # ============================================================
        # 2 Edge 连接（与 Layout F 拓扑完全一致）
        # ============================================================
        edge_data += [
            (100, 101), (101, 102), (102, 103), (103, 104), (104, 105),
            (200, 201), (201, 202), (202, 203), (203, 204), (204, 205),
            (101, 201), (102, 202), (103, 203), (104, 204),
            (201, 300), (202, 304), (203, 306), (204, 309),

            (300, 301), (301, 302), (302, 303), (303, 304),
            (304, 305), (305, 306),
            (306, 307), (307, 308), (308, 309),

            (300, 401), (401, 402), (402, 403), (403, 404),
            (404, 405), (405, 406), (406, 407), (407, 408), (408, 409),

            (309, 501), (501, 502), (502, 503), (503, 504),
            (504, 505), (505, 506), (506, 507),

            (304, 601), (601, 602), (602, 603), (603, 604), (604, 605),
            (305, 611), (611, 612),
            (306, 621), (621, 622), (622, 623), (623, 624), (624, 625),
            (605, 631), (631, 632), (632, 612), (612, 633), (633, 634), (634, 625),
        ]

        # ============================================================
        # 3 Gate Stubs（重点加长：所有连接段扩大3倍，dx/dy = 15）
        # ============================================================
        # Proportional network enlargement: edge 102-103 becomes 160 m.
        # The original coordinate distance is 24 model units.
        NETWORK_SCALE = 8.0 / 7.0
        for node_id in list(pos):
            x, y = pos[node_id]
            pos[node_id] = (x * NETWORK_SCALE, y * NETWORK_SCALE)
        # Move the two side stand groups upward while keeping their original
        # side orientation. The upper nodes remain below 605 and 625.
        side_shift = 2.5
        side_groups = [601, 602, 603, 604, 621, 622, 623, 624]
        for node_id in side_groups:
            x, y = pos[node_id]
            pos[node_id] = (x, y + side_shift)
        gate_id = 0
        def add_gate(parent_node, dx, dy):
            nonlocal gate_id
            stub_id = 2000 + gate_id
            direction_norm = math.hypot(dx, dy)
            ux, uy = dx / direction_norm, dy / direction_norm
            parent_to_stub = 16.0 / 5.833
            stub_to_gate = 32.0 / 5.833
            px, py = pos[parent_node]
            pos[stub_id] = (
                px + ux * parent_to_stub,
                py + uy * parent_to_stub,
            )
            pos[gate_id] = (
                px + ux * (parent_to_stub + stub_to_gate),
                py + uy * (parent_to_stub + stub_to_gate),
            )
            edge_data.append((parent_node, stub_id))
            edge_data.append((stub_id, gate_id))
            gate_id += 1
        # 1. 远西北折线段 (405-408): diagonal 加长 dx=-12, dy=12
        for n in [405, 406, 407, 408]: add_gate(n, -12, 12)
        # 2. 远西纵向 (401-403): dx=-15
        for n in [401, 402, 403]:      add_gate(n, -15, 0)
        # 3. 左侧横向 (301-303): dy=15
        for n in [301, 302, 303]:      add_gate(n, 0, 15)
        # 4. 港湾左柱 (601-604): dx=-15（抵达 x=-27，与左侧 303 安全错开）
        for n in [601, 602, 603, 604]: add_gate(n, -15, 0)
        # 5. 港湾顶梁 (631-634): dy=15
        for n in [631, 632, 633, 634]: add_gate(n, 0, 15)
        # 6. 港湾右柱 (621-624): dx=15（抵达 x=27，与右侧 307 安全错开）
        for n in [621, 622, 623, 624]: add_gate(n, 15, 0)
        # 7. 右侧横向 (307-308): dy=15
        for n in [307, 308]:           add_gate(n, 0, 15)
        # 8. 远东纵向 (501-506): dx=15
        for n in [501, 502, 503, 504, 505, 506]: add_gate(n, 15, 0)

        # ============================================================
        # 4 跑道入口/出口
        # ============================================================
        ENTRY = 900; EXIT = 901
        pos[ENTRY] = (x_fl - 5, -2)
        pos[EXIT]  = (x_fr + 5, -2)
        edge_data += [(ENTRY, 101), (104, EXIT)]

        # ============================================================
        # 5 构建 Node / Edge 对象
        # ============================================================
        nodes = []
        for i, p in pos.items():
            t = NodeType.GATE if i < 30 else NodeType.INTERSECTION
            nodes.append(Node(i, p, t))

        edges = []
        for u, v in edge_data:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            d   = math.hypot(x1 - x2, y1 - y2)
            ang = math.atan2(y2 - y1, x2 - x1)
            edges.append(Edge(u, v, d, ang))
            edges.append(Edge(v, u, d, ang + math.pi))

        return nodes, edges


    def visualize_network(self) -> None:
        """可视化滑行网络。"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.Graph()
            pos = {}
            for nid, node in self.nodes.items():
                G.add_node(nid, type=node.type.value)
                pos[nid] = node.position

            for edge in self.edges:
                G.add_edge(edge.start, edge.end, weight=round(edge.length, 2))

            plt.figure(figsize=(12, 10))
            
            # 根据类型填色
            node_colors = []
            for n in G.nodes():
                ntype = self.nodes[n].type
                if ntype == NodeType.GATE:
                    node_colors.append('salmon')
                else:
                    node_colors.append('skyblue')

            nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                    node_size=200, font_size=7, font_weight='bold')
            
            edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
            #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title("Airport Taxiing Network Topology")
            plt.show()
        except ImportError:
            print("Warning: networkx or matplotlib not installed. Skipping visualization.")

    # 兼容性别名属性
    @property
    def agent_positions(self): return self.pos
    @property
    def agent_goals(self): return self.goal
    @property
    def agent_on_edge(self): return self.on_edge
    @property
    def agent_speeds(self): return self.speed
    @property
    def agent_done(self): return self.done
    @property
    def conflict_count(self): return self.conflicts
    @conflict_count.setter
    def conflict_count(self, value): self.conflicts = value

    def get_graph_for_planner(self):
        return self.planner_graph()

    def get_neighbors(self, node_id: int) -> list[int]:
        return [v for v, eidx in self.adj.get(node_id, []) if not self.edges[eidx].closed]

    def get_neighbor_edges(self, node_id: int) -> list[tuple[int, int]]:
        return [(v, eidx) for v, eidx in self.adj.get(node_id, []) if not self.edges[eidx].closed]

    def get_edge_between(self, start: int, end: int) -> int | None:
        for v, eidx in self.adj.get(start, []):
            if v == end: return eidx
        return None

    def get_shortest_path_distance(self, start: int, end: int) -> float:
        return self.dist_to(start, end)


# =========================================================
# 初始化
# =========================================================

    def reset(self, starts=None, goals=None, delays=None):

        self.step_count=0
        self.conflicts=0
        self.collision_events=[]
        self.clearance_interventions=0
        self.clearance_intervention_by_agent=[0] * self.num_agents
        self.clearance_blocked=[False] * self.num_agents
        self.clearance_blocker_by_agent=[None] * self.num_agents
        self.clearance_min_distance_sq_by_agent=[float("inf")] * self.num_agents
        self.edge_hold_reason=[None] * self.num_agents

        for e in self.edges:
            e.closed=False
            e.occupied=False

        gates=[i for i,n in self.nodes.items() if n.type==NodeType.GATE]

        if starts is None and goals is None:
            if self.layout == "F":
                # Layout F：进出港非对称任务
                # 进港入口: 100(左侧外环起点) 或 200(次外环左起点)
                # 离港出口: 105(右侧外环终点) 或 205(次外环右终点)
                ARRIVAL_ENTRIES  = [100, 200]   # 进港出发点（跑道入口侧）
                DEPARTURE_EXITS  = [105, 205]   # 离港目标点（跑道出口侧）
                all_gates = [i for i, n in self.nodes.items() if n.type == NodeType.GATE]

                n_inbound  = self.num_agents // 2
                n_outbound = self.num_agents - n_inbound

                all_intersections = [
                    i for i, n in self.nodes.items()
                    if n.type == NodeType.INTERSECTION
                    and i not in ARRIVAL_ENTRIES
                    and i not in DEPARTURE_EXITS
                    and i < 2000
                    and i not in {900, 901}
                ]

                # 随机生成本 epoch 的进港/离港飞机数量比例 (混合随机流，避免死板的 50/50，但不全为0)
                if self.num_agents > 1 and getattr(self, "random_flow_mix", True):
                    n_inbound = random.randint(1, self.num_agents - 1)
                else:
                    n_inbound = max(1, self.num_agents // 2)
                n_outbound = self.num_agents - n_inbound

                starts = []
                goals = []
                
                # 进港 (Inbound): 目标必定是机位
                # 修正：当请求数量超过总机位数时，改用 random.choices (有放回采样) 以防止崩溃
                if n_inbound <= len(all_gates):
                    inbound_goals = random.sample(all_gates, n_inbound)
                else:
                    inbound_goals = random.choices(all_gates, k=n_inbound)
                
                goals.extend(inbound_goals)
                for _ in range(n_inbound):
                    # 70% 跑道入口(100/200)，30% 滑行道交叉口(模拟中途加入以增加训练泛化度)
                    # 取消 "not in starts" 限制：现在有 t0 排队机制，允许同时刻分配在同一起点排队！
                    if random.random() >= getattr(self, "midroute_start_prob", 0.0):
                        chosen_start = ARRIVAL_ENTRIES[len(starts) % len(ARRIVAL_ENTRIES)]
                    else:
                        chosen_start = random.choice(all_intersections)
                    starts.append(chosen_start)

                # 离港 (Outbound): 目标必定是跑道出口
                outbound_goals = [DEPARTURE_EXITS[i % len(DEPARTURE_EXITS)] for i in range(n_outbound)]
                goals.extend(outbound_goals)
                
                # 剩余可用作离港发车的机位
                remaining_gates = [g for g in all_gates if g not in inbound_goals]
                for _ in range(n_outbound):
                    # 70% 停机位发出，30% 滑行道交叉口(模拟中途加入)
                    if random.random() >= getattr(self, "midroute_start_prob", 0.0):
                        candidates = remaining_gates
                        if not candidates:
                            candidates = all_intersections
                    else:
                        candidates = all_intersections
                        
                    chosen_start = random.choice(candidates)
                    starts.append(chosen_start)
                    # 避免同一个机位被两家离港飞机占用(机位无容纳空间)
                    if chosen_start in remaining_gates:
                        remaining_gates.remove(chosen_start)
            else:
                # 其他 layout：保证起终点完全隔离
                # 修正：支持大容量采样
                if self.num_agents * 2 <= len(gates):
                    selected = random.sample(gates, self.num_agents * 2)
                else:
                    selected = random.choices(gates, k=self.num_agents * 2)
                starts = selected[:self.num_agents]
                goals = selected[self.num_agents:]
        else:
            if starts is None:
                starts = random.sample([g for g in gates if g not in goals], self.num_agents)
            if goals is None:
                goals = random.sample([g for g in gates if g not in starts], self.num_agents)


        self.pos=list(starts)
        self.goal=list(goals)

        self.on_edge=[False]*self.num_agents
        self.edge_idx=[None]*self.num_agents
        self.progress=[0.0]*self.num_agents
        self.speed=[1.0]*self.num_agents
        self.edge_speed_cap=[None]*self.num_agents
        self.turn_brake_active=[False]*self.num_agents
        self.reference_paths=[[] for _ in range(self.num_agents)]
        self.last_actions=[7]*self.num_agents
        self.forced_turn_brake_count=0
        self.done=[False]*self.num_agents
        self.crashed=[False]*self.num_agents
        self.delayed=[False]*self.num_agents
        self.priority=list(range(self.num_agents))
        self.prev_edge_dir=[None]*self.num_agents
        self.came_from=[None]*self.num_agents
        self.prev_speed_delta=[0.0]*self.num_agents  # 惯性约束：记录上步速度变化量

        # ================= 持久化事件追踪状态 ==================
        if not hasattr(self, 'delay_timers'):
            self.delay_timers = [0] * self.num_agents
        else:
            self.delay_timers = [0] * self.num_agents
        self.closure_timers = {}
        self.priority_cooldown = 0
        self.closure_event_counter = 0
        # ========================================================

        # 初始化错峰发车延时 (Staggered Release / t0 Schedule)
        self.shadow_release_adjustments = 0
        if delays is not None:
            # 测试阶段：使用外部传入的明确时刻表（来自 FIXED_TEST_CASES 的 t0(f) 字段）
            self.start_delays = list(delays)
        else:
            # 训练阶段：基于节点感知的智能错峰发车
            # 核心问题：多架飞机共享同一入场节点（100/200）时会争抢廊道方向锁，
            # 造成不可逆等待。通过确保共享同一入场节点的飞机之间有最小时间间隔来解决。
            ARRIVAL_ENTRIES_SET = {100, 200}
            # 物理标定：MIN_STAGGER = 120步 × 1.127s/步 ≈ 135秒 ≈ 2分钟最小进场间隔
            MIN_STAGGER = getattr(self, "entry_stagger_steps", 120)

            entry_next_avail = {}  # 每个入场节点下一个可用时刻
            shadow_next_avail = {}
            delays_list = []

            for i in range(self.num_agents):
                pos = self.pos[i]  # 此刻 self.pos 已经被赋值为 starts
                if pos in ARRIVAL_ENTRIES_SET:
                    # 入场节点：按顺序分配时间槽，确保相邻飞机间隔 MIN_STAGGER
                    t_slot = entry_next_avail.get(pos, 0)
                    entry_next_avail[pos] = t_slot + MIN_STAGGER
                    delays_list.append(t_slot)
                else:
                    # 非入场起点（机位/滑行道）：随机少量延迟，30%立即出发
                    # 物理标定：上限60步 × 1.127s = 67.6秒（机位准备推出时间）
                    departure_spread = max(int(getattr(self, "departure_stagger_steps", 60)), 0)
                    if departure_spread == 0 or random.random() < 0.3:
                        release_delay = 0
                    else:
                        release_delay = random.randint(0, departure_spread)

                    component = self.gate_shadow_components.get(pos)
                    headway = self.shadow_release_headway_steps
                    if component is not None and headway > 0:
                        next_available = shadow_next_avail.get(component, 0)
                        adjusted_delay = max(release_delay, next_available)
                        self.shadow_release_adjustments += int(
                            adjusted_delay > release_delay
                        )
                        shadow_next_avail[component] = adjusted_delay + headway
                        release_delay = adjusted_delay
                    delays_list.append(release_delay)

            self.start_delays = delays_list

        # 第一轮开始，重置所有廊道的占用状态
        for c in self.corridor_mgr.corridors:
            c.direction = None
            c.occupied_by.clear()

        return self._obs()


# =========================================================
# 邻接表
# =========================================================

    def _build_adj(self):

        self.adj={i:[] for i in self.nodes}
        for idx,e in enumerate(self.edges):
            self.adj[e.start].append((e.end,idx))


# =========================================================
# 翼尖净空阴影组构建
# =========================================================

    def _build_gate_shadow_groups(self) -> dict[int, set[int]]:
        """预计算每个机位节点的翼尖净空阴影互斥组。

        物理语义：
            港湾区机位沿联络道并排排列，飞机推出时翼尖扫掠弧面覆盖相邻机位
            出口通道。依据 ICAO Annex 14 Table 3-1，相邻机位飞机须等待当前
            推出航空器翼尖退出干涉区后方可启动推出。

        拓扑关系（Layout F）：
            gate_node(g) ← stub_node(2000+g) ← 父干道节点 p
            同父节点 p 的所有机位 = 直接阴影组（必然干涉）
            p 的相邻干道节点 p' 的机位 = 扩展阴影组（可能干涉）

        Returns:
            dict[gate_id, set(同阴影组其他机位 id)]
        """
        gate_to_parent: dict[int, int] = {}
        parent_to_gates: dict[int, list[int]] = {}

        for node_id, node in self.nodes.items():
            if node.type == NodeType.GATE:
                # 通过 stub_node 反查父干道节点
                stub_id = 2000 + node_id
                if stub_id in self.nodes:
                    for neighbor, _ in self.adj.get(stub_id, []):
                        if neighbor != node_id and neighbor < 2000:
                            gate_to_parent[node_id] = neighbor
                            parent_to_gates.setdefault(neighbor, []).append(node_id)
                            break
                else:
                    # 无 stub_node 的直连机位
                    for neighbor, _ in self.adj.get(node_id, []):
                        if neighbor < 2000:
                            gate_to_parent[node_id] = neighbor
                            parent_to_gates.setdefault(neighbor, []).append(node_id)
                            break

        shadow_groups: dict[int, set[int]] = {}
        for gate_id in gate_to_parent:
            group: set[int] = set()
            parent = gate_to_parent[gate_id]
            # 直接阴影：同父节点其他机位
            for sibling in parent_to_gates.get(parent, []):
                if sibling != gate_id:
                    group.add(sibling)
            # 扩展阴影：父节点相邻干道节点下的机位
            if self.gate_shadow_mode == "extended":
                for adj_parent, _ in self.adj.get(parent, []):
                    if adj_parent < 2000:
                        for cousin in parent_to_gates.get(adj_parent, []):
                            group.add(cousin)
            if self.gate_shadow_mode == "off":
                group.clear()
            shadow_groups[gate_id] = group

        gate_count = len(gate_to_parent)
        avg_shadow = sum(len(v) for v in shadow_groups.values()) / max(gate_count, 1)
        print(f"[TaxiGraphEnv] Gate Shadow Zone ({self.gate_shadow_mode}): "
              f"{gate_count} gates, avg {avg_shadow:.1f} shadow peers/gate.")
        return shadow_groups

    def _build_gate_shadow_components(self) -> dict[int, int]:
        """Map each stand to an undirected gate-shadow connected component."""
        components: dict[int, int] = {}
        component_id = 0
        for gate_id in self.gate_shadow_groups:
            if gate_id in components:
                continue
            stack = [gate_id]
            while stack:
                current = stack.pop()
                if current in components:
                    continue
                components[current] = component_id
                for peer in self.gate_shadow_groups.get(current, set()):
                    if peer not in components:
                        stack.append(peer)
            component_id += 1
        return components


# =========================================================
# 最短路缓存
# =========================================================

    def _precompute_all_pairs(self):

        self.dist={}

        for s in self.nodes:
            d={s:0}
            pq=[(0,s)]

            while pq:
                cd,u=heapq.heappop(pq)
                if cd>d[u]: continue
                for v,eidx in self.adj[u]:
                    w=self.edges[eidx].length
                    nd=cd+w
                    if nd<d.get(v,1e9):
                        d[v]=nd
                        heapq.heappush(pq,(nd,v))

            self.dist[s]=d


    def dist_to(self,a,b):
        return self.dist.get(a,{}).get(b,-1.0)


# =========================================================
# Step
# =========================================================

    def step(self,actions):

        self.step_count+=1

        self.last_actions = list(actions)
        self._dynamic_events()
        self._refresh_occupancy()

        for i,a in enumerate(actions):
            if self.done[i] or self.crashed[i]:
                continue
            # 若发车延时未结束，强制保持在出发节点(处于场外排队状态)
            if getattr(self, 'start_delays', None) and self.start_delays[i] > 0:
                if self.start_delays[i] > 1:
                    self.start_delays[i] -= 1
                else:
                    # delay == 1: 准备降临时，必须检查当前出生点是否被其他已降临的活跃飞机占据
                    occupied = False
                    for j in range(self.num_agents):
                        if i == j: continue
                        if not self.done[j] and not self.crashed[j] and self.start_delays[j] == 0:
                            # 活跃飞机j
                            if not self.on_edge[j] and self.pos[j] == self.pos[i]:
                                occupied = True
                                break
                            elif self.on_edge[j] and self.edges[self.edge_idx[j]].start == self.pos[i] and self.progress[j] < 0.2:
                                # 飞机j刚驶离该节点不足一点距离，暂不放行
                                occupied = True
                                break
                                
                    if not occupied:
                        self.start_delays[i] -= 1 # 降临成功，变为0，参与下次行动和碰撞检测
                continue
                
            self._act(i,a)

        self._move()
        self._detect_global_collisions()
        # 清理已完成/崩溃代理的廊道占用记录
        for i in range(self.num_agents):
            if self.done[i] or self.crashed[i]:
                self.corridor_mgr.leave_all(i)

        done=all(self.done) or self.step_count>=self.max_steps
        return self._obs(),done,list(self.crashed)
    
    def _get_agent_coords(self, i: int) -> tuple[float, float]:
        """获取智能体当前的物理坐标（插值）。"""
        if not self.on_edge[i]:
            return self.nodes[self.pos[i]].position
        e = self.edges[self.edge_idx[i]]
        s_pos = self.nodes[e.start].position
        e_pos = self.nodes[e.end].position
        p = self.progress[i]
        return (s_pos[0] + (e_pos[0] - s_pos[0]) * p, s_pos[1] + (e_pos[1] - s_pos[1]) * p)

    def _edge_coords_at_progress(self, edge, progress: float) -> tuple[float, float]:
        start_xy = self.nodes[edge.start].position
        end_xy = self.nodes[edge.end].position
        return (
            start_xy[0] + (end_xy[0] - start_xy[0]) * progress,
            start_xy[1] + (end_xy[1] - start_xy[1]) * progress,
        )

    def is_node_clearance_blocked(self, node_id: int, agent_id: int) -> bool:
        """Whether another active aircraft occupies the node clearance zone."""
        x_node, y_node = self.nodes[node_id].position
        for other_id in range(self.num_agents):
            if (
                other_id == agent_id
                or self.done[other_id]
                or self.crashed[other_id]
                or self.start_delays[other_id] > 0
            ):
                continue
            x_other, y_other = self._get_agent_coords(other_id)
            if (
                (x_node - x_other) ** 2 + (y_node - y_other) ** 2
                < self.physical_clearance_min_distance_sq
            ):
                return True
        return False

    def is_node_entry_blocked(self, node_id: int, agent_id: int) -> bool:
        """Return whether an aircraft should remain upstream of `
ode_id``.

        Continuous clearance alone is checked only once an aircraft is already
        on an edge.  This helper additionally treats another aircraft's final
        approach to the same node as a temporary reservation, keeping queues
        at nodes rather than parking them at the downstream end of an edge.
        """
        if self.is_node_clearance_blocked(node_id, agent_id):
            return True
        for other_id in range(self.num_agents):
            if (
                other_id == agent_id
                or self.done[other_id]
                or self.crashed[other_id]
                or self.start_delays[other_id] > 0
                or not self.on_edge[other_id]
            ):
                continue
            edge_index = self.edge_idx[other_id]
            if edge_index is None:
                continue
            other_edge = self.edges[edge_index]
            if (
                other_edge.end == node_id
                and self.progress[other_id] >= self.approach_reservation_progress
            ):
                return True
        return False

    def _clear_progress_before_conflict(self, agent_id: int, edge, candidate: float) -> float:
        """Limit an edge advance before it violates continuous clearance.

        The existing detector correctly identifies close approaches across
        different incident edges, but previously did so only after declaring a
        crash. This shield uses the identical squared-distance threshold while
        retaining the current sequential movement model.
        """
        if not self.physical_clearance_shield:
            return candidate
        current = self.progress[agent_id]
        if candidate <= current:
            return candidate

        def clear(progress: float) -> bool:
            x_i, y_i = self._edge_coords_at_progress(edge, progress)
            for other_id in range(self.num_agents):
                if (
                    other_id == agent_id
                    or self.done[other_id]
                    or self.crashed[other_id]
                    or self.start_delays[other_id] > 0
                ):
                    continue
                x_j, y_j = self._get_agent_coords(other_id)
                if (
                    (x_i - x_j) ** 2 + (y_i - y_j) ** 2
                    < self.physical_clearance_min_distance_sq
                ):
                    return False
            return True

        if clear(candidate):
            self.clearance_blocked[agent_id] = False
            self.clearance_blocker_by_agent[agent_id] = None
            self.clearance_min_distance_sq_by_agent[agent_id] = float("inf")
            return candidate

        # Motion along one edge is continuous. Stop at the last safe sampled
        # point before the first protected separation boundary.
        last_safe = current
        for fraction in np.linspace(0.05, 1.0, 20):
            trial = current + (candidate - current) * float(fraction)
            if not clear(trial):
                break
            last_safe = trial
        if not self.clearance_blocked[agent_id]:
            self.clearance_interventions += 1
            self.clearance_intervention_by_agent[agent_id] += 1
            self.clearance_blocked[agent_id] = True
        blocker = None
        min_distance_sq = float("inf")
        x_i, y_i = self._edge_coords_at_progress(edge, candidate)
        for other_id in range(self.num_agents):
            if (
                other_id == agent_id
                or self.done[other_id]
                or self.crashed[other_id]
                or self.start_delays[other_id] > 0
            ):
                continue
            x_j, y_j = self._get_agent_coords(other_id)
            distance_sq = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                blocker = other_id
        self.clearance_blocker_by_agent[agent_id] = blocker
        self.clearance_min_distance_sq_by_agent[agent_id] = min_distance_sq
        self.edge_hold_reason[agent_id] = "continuous_clearance"
        self.speed[agent_id] = 0.0
        return last_safe

    def _detect_global_collisions(self):
        """全局碰撞检测，一旦发生，双方 crash 并出局。支持节点和对向边碰撞。
        忽略已完成、已死亡，或处于 t0 发车延时中（场外排队）的飞机。
        """
        for i in range(self.num_agents):
            if self.done[i] or self.crashed[i]: continue
            if self.start_delays[i] > 0: continue  # t0 未到，飞机处于虚拟排队队列，不会产生物理实体碰撞

            for j in range(i+1, self.num_agents):
                if self.done[j] or self.crashed[j]: continue
                if self.start_delays[j] > 0: continue

                is_crashing = False
                collision_type = None
                separation_sq = None

                # 1. 节点相撞 (在同一个节点上)
                if not self.on_edge[i] and not self.on_edge[j]:
                    if self.pos[i] == self.pos[j]:
                        is_crashing = True
                        collision_type = "node_overlap"

                # 2. 对向边碰撞: A 在 u→v，B 在 v→u，且进度之和>=0.9
                elif self.on_edge[i] and self.on_edge[j]:
                    ei = self.edges[self.edge_idx[i]]
                    ej = self.edges[self.edge_idx[j]]
                    if ei.start == ej.end and ei.end == ej.start:
                        if self.progress[i] + self.progress[j] >= 0.5:
                            is_crashing = True
                            collision_type = "head_on_edge"
                    # 3. 同向追尾：同一条边且进度极小  
                    elif ei.start == ej.start and ei.end == ej.end:
                        if abs(self.progress[i] - self.progress[j]) < 0.1:
                            if not self._resolve_same_edge_clearance(i, j):
                                is_crashing = True
                                collision_type = "same_direction_spacing"

                # 4. 万能物理距离兜底 (拦截转角相撞、Node-Edge跨域穿模等拓扑遗漏)
                if not is_crashing:
                    x_i, y_i = self._get_agent_coords(i)
                    x_j, y_j = self._get_agent_coords(j)
                    # 两架飞机物理欧几里得距离 < 0.7 个坐标面单位即宣告机体相撞 (安全距离界限设定)
                    separation_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                if (
                    separation_sq is not None
                    and separation_sq < self.physical_clearance_min_distance_sq
                ):
                    if self._resolve_node_edge_clearance(i, j):
                        x_i, y_i = self._get_agent_coords(i)
                        x_j, y_j = self._get_agent_coords(j)
                        separation_sq = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
                    if separation_sq < self.physical_clearance_min_distance_sq:
                        is_crashing = True
                        collision_type = "physical_clearance"

                if is_crashing:
                    def edge_tuple(agent_id):
                        edge_index = self.edge_idx[agent_id]
                        if edge_index is None:
                            return None
                        edge = self.edges[edge_index]
                        return [edge.start, edge.end]
                    self.collision_events.append({
                        "step": self.step_count,
                        "type": collision_type or "unknown",
                        "agents": [i, j],
                        "positions": [self.pos[i], self.pos[j]],
                        "on_edge": [bool(self.on_edge[i]), bool(self.on_edge[j])],
                        "progress": [round(float(self.progress[i]), 4), round(float(self.progress[j]), 4)],
                        "edges": [edge_tuple(i), edge_tuple(j)],
                        "separation_sq": round(float(separation_sq), 6) if separation_sq is not None else None,
                    })
                    self.crashed[i] = True
                    self.crashed[j] = True
                    self.done[i] = True
                    self.done[j] = True
                    self.conflicts += 1
                    # 立即释放双方所有廊道占用，防止级联死锁
                    self.corridor_mgr.leave_all(i)
                    self.corridor_mgr.leave_all(j)
                    break

    def _resolve_same_edge_clearance(self, i: int, j: int) -> bool:
        """Back the trailing aircraft away from a same-edge spacing conflict."""
        if not (self.on_edge[i] and self.on_edge[j]):
            return False
        edge_i = self.edges[self.edge_idx[i]]
        edge_j = self.edges[self.edge_idx[j]]
        if (edge_i.start, edge_i.end) != (edge_j.start, edge_j.end):
            return False

        front, rear = (i, j) if self.progress[i] >= self.progress[j] else (j, i)
        edge = edge_i
        front_progress = float(self.progress[front])
        rear_progress = float(self.progress[rear])
        front_x, front_y = self._edge_coords_at_progress(edge, front_progress)

        def separation_sq(progress: float) -> float:
            x, y = self._edge_coords_at_progress(edge, progress)
            return (x - front_x) ** 2 + (y - front_y) ** 2

        if separation_sq(rear_progress) >= self.physical_clearance_min_distance_sq:
            safe_progress = min(rear_progress, max(0.0, front_progress - 0.100001))
        else:
            if separation_sq(0.0) < self.physical_clearance_min_distance_sq:
                return False
            low, high = 0.0, rear_progress
            for _ in range(24):
                mid = (low + high) * 0.5
                if separation_sq(mid) >= self.physical_clearance_min_distance_sq:
                    low = mid
                else:
                    high = mid
            safe_progress = min(low, max(0.0, front_progress - 0.100001))
        if (
            separation_sq(safe_progress) < self.physical_clearance_min_distance_sq
            or front_progress - safe_progress < 0.099999
        ):
            return False

        self.progress[rear] = safe_progress
        self.speed[rear] = 0.0
        self.edge_hold_reason[rear] = "same_edge_clearance"
        self.clearance_blocked[rear] = True
        self.clearance_blocker_by_agent[rear] = front
        return True
    def _resolve_node_edge_clearance(self, i: int, j: int) -> bool:
        """Recover a node-edge clearance race without weakening the shield."""
        if self.on_edge[i] == self.on_edge[j]:
            return False
        edge_agent = i if self.on_edge[i] else j
        node_agent = j if edge_agent == i else i
        edge_index = self.edge_idx[edge_agent]
        if edge_index is None:
            return False

        edge = self.edges[edge_index]
        current = float(self.progress[edge_agent])
        node_x, node_y = self._get_agent_coords(node_agent)

        def separation_sq(progress: float) -> float:
            x, y = self._edge_coords_at_progress(edge, progress)
            return (x - node_x) ** 2 + (y - node_y) ** 2

        if separation_sq(current) >= self.physical_clearance_min_distance_sq:
            return False
        if separation_sq(0.0) < self.physical_clearance_min_distance_sq:
            return False

        low, high = 0.0, current
        for _ in range(24):
            mid = (low + high) * 0.5
            if separation_sq(mid) >= self.physical_clearance_min_distance_sq:
                low = mid
            else:
                high = mid

        self.progress[edge_agent] = low
        self.speed[edge_agent] = 0.0
        self.edge_hold_reason[edge_agent] = "continuous_clearance"
        if not self.clearance_blocked[edge_agent]:
            self.clearance_interventions += 1
            self.clearance_intervention_by_agent[edge_agent] += 1
        self.clearance_blocked[edge_agent] = True
        self.clearance_blocker_by_agent[edge_agent] = node_agent
        self.clearance_min_distance_sq_by_agent[edge_agent] = separation_sq(high)
        return True


# =========================================================
# 动作执行
# =========================================================

    def _edge_is_gate_connector(self, edge) -> bool:
        """Return True for either segment between a stand and its stub node."""
        a, b = int(edge.start), int(edge.end)
        return (0 <= a < 30 and 2000 <= b < 2030) or (
            2000 <= a < 2030 and 0 <= b < 30
        )

    def _is_turn(self, previous_edge, next_edge) -> bool:
        if previous_edge is None or next_edge is None:
            return False
        angle_diff = abs(next_edge.direction - previous_edge.direction)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        return angle_diff > math.pi / 3

    def set_reference_path(self, agent_id: int, path: list[int] | None) -> None:
        """Set the route used only for turn-ahead speed preview."""
        if not hasattr(self, "reference_paths"):
            self.reference_paths = [[] for _ in range(self.num_agents)]
        self.reference_paths[agent_id] = list(path or [])

    def _edge_has_turn_ahead(self, edge, agent_id: int | None = None) -> bool:
        """Detect a turn at edge.end using the current route when available."""
        if agent_id is not None and hasattr(self, "reference_paths"):
            path = self.reference_paths[agent_id]
            for k in range(len(path) - 2):
                if path[k] == edge.start and path[k + 1] == edge.end:
                    next_pair = (path[k + 1], path[k + 2])
                    for _, next_idx in self.adj.get(edge.end, []):
                        next_edge = self.edges[next_idx]
                        if (next_edge.start, next_edge.end) == next_pair:
                            return self._is_turn(edge, next_edge)
                    return False

        # Without a route, only apply the conservative rule when every
        # non-reverse exit at the node requires a turn.
        candidates = []
        for _, next_idx in self.adj.get(edge.end, []):
            next_edge = self.edges[next_idx]
            if next_edge.end != edge.start:
                candidates.append(next_edge)
        return bool(candidates) and all(self._is_turn(edge, item) for item in candidates)

    def _turn_braking_distance(self, speed: float) -> float:
        """Distance needed to reach turn_speed_limit under max discrete braking."""
        excess = max(0.0, float(speed) - self.turn_speed_limit)
        brake_steps = int(math.ceil(excess / self.turn_brake_decel))
        if brake_steps <= 0:
            return self.turn_brake_buffer
        mean_speed = (float(speed) + self.turn_speed_limit) * 0.5
        return brake_steps * mean_speed + self.turn_brake_buffer
    def _act(self,i,a):

        # ================== 持久化延误扰动拦截 ==================
        if getattr(self, 'delayed', [False]*self.num_agents)[i]:
            self.speed[i] = 0.0
            return
        # ========================================================

        if self.on_edge[i]:

            if a==5: self._apply_speed_delta(i, -0.21)   # 对应 -0.98 m/s² @T_step=1.127s
            elif a==6: self._apply_speed_delta(i, +0.21)  # 对应 +0.98 m/s² @T_step=1.127s
            return

        if a<=4:

            nbrs=self.adj[self.pos[i]]
            if a>=len(nbrs): return

            nxt,eidx=nbrs[a]

            e=self.edges[eidx]
            if e.closed: return

            # Do not enter a downstream-constrained edge. The clearance
            # shield remains the final geometric safeguard while moving, but
            # admission control keeps avoidable queues at the upstream node.
            if self.pre_entry_clearance_guard and self.is_node_entry_blocked(nxt, i):
                return

            # 廊道预留检查：若下一条边属于某个廊道，检查是否允许进入
            corridor = self.corridor_mgr.get_corridor(e.start, e.end)
            if corridor is not None:
                # 确定进入方向
                entry_dir = corridor.edge_direction(e.start, e.end)
                if not self.corridor_mgr.can_enter(corridor, entry_dir, agent_id=i):
                    # 廊道已被对向流占用，强制等候
                    return
                # 内部节点补充登记：若当前节点在廊道内且尚未登记，补充登记
                interior = self.corridor_mgr._interior_nodes.get(self.pos[i])
                if interior is not None and interior.get("corridor") == corridor:
                    if i not in corridor.occupied_by:
                        self.corridor_mgr.enter(corridor, i, entry_dir)

            # ── 转弯强制减速（Turn Speed Constraint）──────────────────
            # 检测新进入边与上一条边的夹角，若转弯角 > 60° 则强制限速。
            # 物理标定： V_turn = 5.13 m/s，对应代码 = 5.13 * 1.127 / 5.833 = 0.991 ≈ 1.0 unit/step
            # Persistent speed cap for the selected edge.
            speed_cap = None
            previous_edge = None
            # edge_idx is cleared when the previous edge is completed.
            # Recover it from the last node pair so a real turn entry
            # still receives the turn-speed cap.
            if self.came_from[i] is not None:
                previous_idx = self.get_edge_between(self.came_from[i], self.pos[i])
                if previous_idx is not None:
                    previous_edge = self.edges[previous_idx]
            if self._is_turn(previous_edge, e):
                speed_cap = self.turn_speed_limit
            if self._edge_is_gate_connector(e):
                speed_cap = self.gate_speed_limit if speed_cap is None else min(
                    speed_cap, self.gate_speed_limit
                )
            self.edge_speed_cap[i] = speed_cap
            self.turn_brake_active[i] = (
                self.enable_anticipatory_turn_braking
                and self._edge_has_turn_ahead(e, i)
            )

            self.on_edge[i]=True
            self.edge_idx[i]=eidx
            self.progress[i]=0

            # 登记进入廊道
            if corridor is not None:
                self.corridor_mgr.enter(corridor, i, entry_dir)

        elif a==5: self._apply_speed_delta(i, -0.21)   # 强减速: -0.98 m/s² → Δv=-0.98*1.127²/5.833=-0.213 unit/step
        elif a==6: self._apply_speed_delta(i, +0.21)   # 强加速: +0.98 m/s² → Δv=+0.213 unit/step


    def _apply_speed_delta(self, i: int, raw_delta: float) -> None:
        """受惯性约束的速度增量控制。

        核心物理语义：
          速度变化量不是「直接设置」，而是「上一步增量的延续」。
          若本步加速方向与上步相反（换向），则最大变化量衰减为 50%，
          模拟真实飞机的动力学惯性与液压刹车延迟。

        效果：
          ① 速度变化连续性（inertia）：不可能单步剧烈跳变
          ② 行为惯性：上一步加速后，当步减速幅度受限
          ③ 抑制震荡（oscillation suppression）：策略反复横跳时自动衰减
        """
        INERTIA_FACTOR = 0.5   # 反向换向时的幅度衰减系数
        prev = self.prev_speed_delta[i]

        # 若换向（符号相反且上步非零），则衰减变化量
        if prev != 0.0 and (raw_delta * prev < 0):
            delta = raw_delta * INERTIA_FACTOR
        else:
            delta = raw_delta

        new_speed = max(0.0, min(self.max_speed, self.speed[i] + delta))
        self.prev_speed_delta[i] = new_speed - self.speed[i]  # 记录实际发生的变化量
        self.speed[i] = new_speed


# =========================================================
# 冲突检测（完整工业版）
# =========================================================

    def _conflict(self,i,nxt):

        cur=self.pos[i]

        for j in range(self.num_agents):
            if j==i or self.done[j]:
                continue

            # 节点冲突
            if self.pos[j]==nxt and not self.on_edge[j]:
                if self.priority[i]>self.priority[j]:
                    return True

            # 对向冲突
            if self.on_edge[j]:
                ej=self.edge_idx[j]
                e=self.edges[ej]
                if e.start==nxt and e.end==cur:
                    return True

        return False


# =========================================================
# 位置更新（含转向角减速 + GATE到达限速）
# =========================================================

    def _move(self):

        for i in range(self.num_agents):

            if not self.on_edge[i] or self.done[i] or self.crashed[i]: # Added crashed check
                continue

            e=self.edges[self.edge_idx[i]]

            # --- 自动巡航 (Automated Cruise Control) ---
            # 环境强制接管加速度，只要不死锁必定加速，全权免除 RL 震荡
            # Automatic cruise control yields to explicit speed actions.
            last_action = getattr(self, "last_actions", [7] * self.num_agents)[i]
            if getattr(self, "auto_speed", True) and last_action not in (5, 6, 7):
                speed_step_val = getattr(self, "speed_step", 0.21)
                max_speed_val = getattr(self, "max_speed", 3.0)
                self.speed[i] = min(max_speed_val, self.speed[i] + speed_step_val)

            if self.edge_speed_cap[i] is not None:
                self.speed[i] = min(self.speed[i], self.edge_speed_cap[i])

            # Gate connectors represent pushback / stand-entry movement. Keep
            # normal motion at a constant low speed; safety checks below may
            # still reduce it to zero when a blockage is detected.
            if self._edge_is_gate_connector(e):
                self.speed[i] = self.gate_speed_limit

            # Hard anticipatory braking overrides acceleration/hold actions.
            if self.turn_brake_active[i] and self.speed[i] > self.turn_speed_limit:
                remaining = max(0.0, (1.0 - self.progress[i]) * e.length)
                if remaining <= self._turn_braking_distance(self.speed[i]):
                    self.speed[i] = max(
                        self.turn_speed_limit,
                        self.speed[i] - self.turn_brake_decel,
                    )
                    self.forced_turn_brake_count += 1

            effective_speed = self.speed[i]
            prog_increment = effective_speed / e.length
            potential_prog = self.progress[i] + prog_increment
            
            # --- 1. 跟车防撞雷达 (Car-Following Rear-end Buffering) ---
            ahead_prog = 2.0
            for j in range(self.num_agents):
                if j != i and not self.done[j] and not self.crashed[j] and self.on_edge[j]:
                    if self.edge_idx[j] == self.edge_idx[i]:
                        # 找到同向同边上的前车
                        if self.progress[j] > self.progress[i]:
                            ahead_prog = min(ahead_prog, self.progress[j])
            
            # 刹车判定距离：0.06。不足安全距离则物理制动
            if potential_prog > ahead_prog - 0.06:
                potential_prog = max(self.progress[i], ahead_prog - 0.06)
                self.speed[i] = 0.0
                self.edge_hold_reason[i] = "same_edge_following"
            
            # --- 2. 交叉口红绿灯防撞 (Node Intersection Buffering) ---
            if potential_prog >= 1.0:
                dest = e.end
                dest_occupied = False
                for j in range(self.num_agents):
                    if j != i and not self.done[j] and not self.crashed[j]:
                        if self.start_delays[j] > 0: continue # 排队中的虚拟飞机不可见，不应当阻挡实体飞机
                        if self.pos[j] == dest and not self.on_edge[j]:
                            dest_occupied = True
                            break
                if dest_occupied:
                    potential_prog = 0.99
                    self.speed[i] = 0.0
                    self.edge_hold_reason[i] = "destination_node_occupied"

            potential_prog = self._clear_progress_before_conflict(i, e, potential_prog)

            self.progress[i] = potential_prog

            if self.progress[i]>=1:

                dest=e.end

                # 离开廊道检测优化：即使走的是非廊道边，只要目的地已经不在廊道内，也应取消占用！
                # 为确保安全，每次步进结算都检查代理是否已不在之前记录的廊道中
                interior = self.corridor_mgr._interior_nodes.get(dest)
                if interior is None:
                    # 如果目的地不属于任何廊道（例如进入了从廊道横突出的连接点 2000 等），强行释放它在所有廊道里的占号
                    self.corridor_mgr.leave_all(i)
                else:
                    # ????????????????????????????????????????
                    corridor = self.corridor_mgr.get_corridor(e.start, e.end)
                    if corridor is not None:
                        self.corridor_mgr.leave(corridor, i)

                self.prev_edge_dir[i]=e.direction  # record completed edge direction

                self.edge_speed_cap[i] = None

                self.turn_brake_active[i] = False

                # GATE 限速：到达机位时限速
                if self.nodes[dest].type==NodeType.GATE:
                    self.speed[i]=min(self.speed[i],0.5)

                self.came_from[i]=self.pos[i]  # 记录到达 dest 之前所在的节点
                self.pos[i]=dest
                self.on_edge[i]=False
                self.edge_idx[i]=None
                self.progress[i]=0

                # 新增检查：进入错误机位连接点判负逻辑
                # stub_node 为 >= 2000 的节点。如果进入的不是自身目标 gate 的 stub_node，任务失败！
                if dest >= 2000:
                    target_gate = dest - 2000
                    # 如果是从主干道进入 stub_node（came_from != target_gate）且不是自己家的 goal
                    if self.came_from[i] != target_gate and self.goal[i] != target_gate:
                        self.collision_events.append({
                            "step": self.step_count,
                            "type": "invalid_stand_stub",
                            "agents": [i],
                            "positions": [self.pos[i]],
                            "on_edge": [False],
                            "progress": [0.0],
                            "edges": [None],
                            "separation_sq": None,
                        })
                        self.crashed[i] = True
                        self.done[i] = True
                        self.conflicts += 1

                if self.pos[i]==self.goal[i]:
                    self.done[i]=True


# =========================================================
# 占用刷新
# =========================================================

    def _refresh_occupancy(self):

        for e in self.edges:
            e.occupied=False

        for i in range(self.num_agents):
            if self.on_edge[i]:
                self.edges[self.edge_idx[i]].occupied=True


# =========================================================
# 动态事件
# =========================================================

    def _dynamic_events(self):
        """基于时间持久化的事件驱动扰动机制"""
        # 1. 持久化滑行道关闭事件 (Taxiway Closure Perturbation)
        self.closure_event_counter += 1
        
        # 恢复已到期的关闭
        active_closures = list(getattr(self, 'closure_timers', {}).keys())
        for e_idx in active_closures:
            self.closure_timers[e_idx] -= 1
            if self.closure_timers[e_idx] <= 0:
                e = self.edges[e_idx]
                e.closed = False
                for rev_e in self.edges:
                    if rev_e.start == e.end and rev_e.end == e.start:
                        rev_e.closed = False
                del self.closure_timers[e_idx]

        # 每 40 步作为一次事件触发窗口进行大封路结算
        if self.closure_event_counter >= 40:
            self.closure_event_counter = 0
            if self.close_prob > 0 and random.random() < self.close_prob * 10: 
                open_edges = [i for i, e in enumerate(self.edges) if not e.closed and not getattr(e, 'occupied', False)]
                if open_edges:
                    num_to_close = random.randint(1, min(2, len(open_edges)))
                    chosen_edges = random.sample(open_edges, num_to_close)
                    for e_idx in chosen_edges:
                        self._close_pair(self.edges[e_idx])
                        self.closure_timers[e_idx] = random.randint(20, 50)

        # 2. 持久化优先级洗牌事件 (Priority Shuffle Perturbation)
        if getattr(self, 'priority_cooldown', 0) > 0:
            self.priority_cooldown -= 1
        else:
            if self.priority_prob > 0 and random.random() < self.priority_prob * 3:
                random.shuffle(self.priority)
                self.priority_cooldown = random.randint(20, 40)

        # 3. 持久化航行延误事件 (Delay Perturbation)
        for i in range(self.num_agents):
            if self.done[i]: continue
                
            if self.delay_timers[i] > 0:
                self.delay_timers[i] -= 1
                if self.delay_timers[i] == 0:
                    self.delayed[i] = False
            else:
                if self.delay_prob > 0 and random.random() < self.delay_prob:
                    self.delayed[i] = True
                    self.delay_timers[i] = random.randint(3, 10)


    def _close_pair(self,e):

        e.closed=True
        for x in self.edges:
            if x.start==e.end and x.end==e.start:
                x.closed=True


# =========================================================
# 观测
# =========================================================

    def _obs(self):

        return {
            "pos":list(self.pos),
            "goal":list(self.goal),
            "speed":list(self.speed),
            "done":list(self.done),
            "edge":list(self.on_edge),
            "conflicts":self.conflicts,
            "collision_events":list(getattr(self, "collision_events", [])),
            "clearance_interventions":int(getattr(self, "clearance_interventions", 0)),
            "clearance_intervention_by_agent":list(getattr(self, "clearance_intervention_by_agent", [])),
            "step":self.step_count,
            "delayed": list(self.delayed),
            "closure": dict(getattr(self, 'closure_timers', {})),
            "priority": list(self.priority)
        }


# =========================================================
# Planner接口
# =========================================================

    def planner_graph(self):
        g={}
        for u in self.nodes:
            g[u]=[]
            for v,eidx in self.adj[u]:
                if not self.edges[eidx].closed:
                    g[u].append((v,self.edges[eidx].length))
        return g
