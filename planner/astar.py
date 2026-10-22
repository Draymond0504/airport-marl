"""A* 路径规划模块。

实现标准 A* 搜索算法，不依赖外部路径库。
仅考虑拓扑和距离，忽略动态冲突。
"""

from __future__ import annotations

import heapq
import math
from typing import Any


class AStarPlanner:
    """A* 路径规划器。"""

    def __init__(self, nodes: dict[int, tuple[float, float]]) -> None:
        """初始化规划器。

        Args:
            nodes: 节点坐标字典 {node_id: (x, y)}。
        """
        self.nodes = nodes

    def _heuristic(self, a: int, b: int) -> float:
        """欧几里得距离启发函数。

        Args:
            a: 起点节点 ID。
            b: 终点节点 ID。

        Returns:
            估计距离。
        """
        ax, ay = self.nodes[a]
        bx, by = self.nodes[b]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def plan_path(
        self,
        start: int,
        goal: int,
        graph: dict[int, list[tuple[int, float]]],
        blocked_nodes: set[int] | None = None,
    ) -> list[int]:
        """使用 A* 搜索从 start 到 goal 的最短路径，规避 blocked_nodes。

        Args:
            start: 起点节点 ID。
            goal: 目标节点 ID。
            graph: 邻接表 {node_id: [(neighbor_id, distance), ...]}。
            blocked_nodes: 需要屏蔽规避的拥堵节点集合。

        Returns:
            路径节点列表 [start, ..., goal]，不可达返回空列表。
        """
        if blocked_nodes is None:
            blocked_nodes = set()
        open_set: list[tuple[float, int]] = [(0.0, start)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start: 0.0}
        f_score: dict[int, float] = {start: self._heuristic(start, goal)}
        closed: set[int] = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            for neighbor, dist in graph.get(current, []):
                if neighbor in closed or neighbor in blocked_nodes:
                    continue
                tentative_g = g_score[current] + dist
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 不可达

    @staticmethod
    def _reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
        """从 came_from 字典回溯路径。

        Args:
            came_from: 父节点映射。
            current: 终点。

        Returns:
            完整路径列表。
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
