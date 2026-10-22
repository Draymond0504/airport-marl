"""课程学习 (Curriculum Learning) 调度器。

用于逐步增加任务难度，从简单到复杂。
"""

from dataclasses import dataclass
from enum import Enum


class CurriculumPhase(Enum):
    """课程阶段定义。"""
    SINGLE_AGENT = 1       # 单个agent
    DUAL_AGENTS = 2        # 双agent
    MULTI_AGENTS = 4       # 四agent（完整任务）


@dataclass
class CurriculumStage:
    """课程阶段配置。"""
    phase: CurriculumPhase
    num_agents: int
    max_steps: int
    episode_start: int
    episode_end: int
    description: str


class CurriculumScheduler:
    """课程学习调度器。
    
    支持三个阶段:
    1. 单agent导航 (0-10k episodes)
    2. 双agent协调 (10k-25k episodes)
    3. 四agent完整任务 (25k-50k episodes)
    """

    def __init__(self, total_episodes: int = 50000) -> None:
        """初始化课程调度器。
        
        Args:
            total_episodes: 总训练 episode 数。
        """
        self.total_episodes = total_episodes
        self.stages = self._create_stages()

    def _create_stages(self) -> list[CurriculumStage]:
        """创建课程阶段。"""
        return [
            CurriculumStage(
                phase=CurriculumPhase.SINGLE_AGENT,
                num_agents=1,
                max_steps=100,
                episode_start=1,
                episode_end=int(0.2 * self.total_episodes),
                description="Single agent basic navigation",
            ),
            CurriculumStage(
                phase=CurriculumPhase.DUAL_AGENTS,
                num_agents=2,
                max_steps=150,
                episode_start=int(0.2 * self.total_episodes) + 1,
                episode_end=int(0.5 * self.total_episodes),
                description="Dual agents with basic coordination",
            ),
            CurriculumStage(
                phase=CurriculumPhase.MULTI_AGENTS,
                num_agents=4,
                max_steps=200,
                episode_start=int(0.5 * self.total_episodes) + 1,
                episode_end=self.total_episodes,
                description="Four agents full multi-agent task",
            ),
        ]

    def get_stage(self, episode: int) -> CurriculumStage:
        """获取当前阶段配置。
        
        Args:
            episode: 当前 episode 号。
        
        Returns:
            当前阶段的配置。
        """
        for stage in self.stages:
            if stage.episode_start <= episode <= stage.episode_end:
                return stage
        
        # 如果超出范围，返回最后阶段
        return self.stages[-1]

    def get_num_agents(self, episode: int) -> int:
        """获取当前应该使用的 agent 数量。"""
        return self.get_stage(episode).num_agents

    def get_max_steps(self, episode: int) -> int:
        """获取当前应该使用的最大步数。"""
        return self.get_stage(episode).max_steps

    def get_phase_name(self, episode: int) -> str:
        """获取当前阶段名称。"""
        return self.get_stage(episode).description

    def print_schedule(self) -> None:
        """打印课程安排。"""
        print("\n" + "=" * 70)
        print("CURRICULUM LEARNING SCHEDULE")
        print("=" * 70)
        for stage in self.stages:
            print(
                f"  Stage {stage.phase.value}: {stage.description}\n"
                f"    Episodes: {stage.episode_start:>6d} - {stage.episode_end:<6d}\n"
                f"    Agents: {stage.num_agents}, Max Steps: {stage.max_steps}"
            )
        print("=" * 70 + "\n")

    def is_phase_transition(self, episode: int) -> bool:
        """检查是否即将发生阶段转换。
        
        在阶段边界返回 True。
        """
        stage = self.get_stage(episode)
        return episode == stage.episode_end or (
            episode + 1 <= self.total_episodes and 
            self.get_stage(episode + 1) != stage
        )

    def get_progress(self, episode: int) -> float:
        """获取当前课程进度 (0.0 - 1.0)。"""
        return min(episode / self.total_episodes, 1.0)


class ProgressiveScheduler(CurriculumScheduler):
    """渐进式调度器 - 连续的难度增加。
    
    相比标准课程学习，采用平滑的渐进式增长而非阶段性跳跃。
    """

    def get_num_agents(self, episode: int) -> int:
        """根据进度平滑地增加 agent 数量。
        
        0-20%: 1个agent
        20-50%: 1-2个agent的平滑过渡
        50-100%: 2-4个agent的平滑过渡
        """
        progress = episode / self.total_episodes
        
        if progress < 0.2:
            return 1
        elif progress < 0.5:
            # 20%-50%: 线性插值从1到2
            alpha = (progress - 0.2) / (0.5 - 0.2)
            return 1 + int(alpha)
        else:
            # 50%-100%: 线性插值从2到4
            alpha = (progress - 0.5) / (0.5)
            return 2 + int(2 * alpha)

    def get_max_steps(self, episode: int) -> int:
        """根据进度平滑地增加最大步数。"""
        progress = episode / self.total_episodes
        
        # 从100步线性增至200步
        base_steps = 100
        max_steps = base_steps + int((200 - base_steps) * progress)
        return max_steps


if __name__ == "__main__":
    # 演示课程调度
    scheduler = CurriculumScheduler(total_episodes=50000)
    scheduler.print_schedule()
    
    # 展示几个关键点的配置
    test_episodes = [1, 10000, 25000, 50000]
    print("Sample configurations:")
    for ep in test_episodes:
        stage = scheduler.get_stage(ep)
        print(
            f"  Episode {ep:>5d}: {stage.num_agents} agents, "
            f"max_steps={stage.max_steps} ({stage.description})"
        )
    
    print("\n" + "=" * 70)
    print("Progressive Scheduler (continuous difficulty increase):")
    print("=" * 70)
    
    prog_scheduler = ProgressiveScheduler(total_episodes=50000)
    for ep in [1, 5000, 10000, 15000, 25000, 35000, 50000]:
        agents = prog_scheduler.get_num_agents(ep)
        steps = prog_scheduler.get_max_steps(ep)
        progress = prog_scheduler.get_progress(ep) * 100
        print(f"  Episode {ep:>5d}: {agents} agents, max_steps={steps} ({progress:.1f}%)")
