"""配置管理模块。

从 YAML 文件加载并解析项目配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    """环境配置。"""
    num_agents: int = 4
    max_steps: int = 200
    max_speed: float = 3.0
    min_speed: float = 0.0
    speed_step: float = 0.5
    # Operational low-speed and turn-speed limits in code units per step.
    gate_speed_limit: float = 0.15
    turn_speed_limit: float = 1.0
    turn_brake_buffer: float = 0.5
    enable_anticipatory_turn_braking: bool = True
    max_neighbors: int = 5
    taxiway_close_prob: float = 0.01
    priority_change_prob: float = 0.01
    delay_prob: float = 0.02
    layout: str = "default"  # "default", "A", "B"
    auto_speed: bool = False
    disable_corridor: bool = False
    # Corridor protection is a physical apron rule. Deadlock recovery is
    # enabled only for training and disabled in operational evaluation.
    enable_deadlock_recovery: bool = True
    midroute_start_prob: float = 0.0
    random_flow_mix: bool = True
    entry_stagger_steps: int = 120
    # Departure release spread used only when reset() creates a training
    # scenario. Keeping this separate from entry staggering lets dense
    # fine-tuning increase interactions without spawning two aircraft at the
    # same arrival entry node.
    departure_stagger_steps: int = 60
    # "extended" preserves the current topology proxy, "direct" keeps only
    # same-parent stands, and "off" is reserved for diagnostics/ablations.
    gate_shadow_mode: str = "extended"
    # Minimum separation inside a gate-shadow component at reset time.
    # Zero preserves the previous independent departure sampling.
    shadow_release_headway_steps: int = 0
    # [Ablation Study] MAPPO-NoGuide: 设为 False 可屏蔽观测中的 A* 先验特征
    # 对应实验：对比"规划引导 MARL"与"纯端到端 MARL"的性能差异
    use_astar_guide: bool = True
    # Feasible-action refinements used for the recovery phase. They preserve
    # the action space and can therefore reuse an existing actor checkpoint.
    prevent_occupied_target_entry: bool = False
    enforce_reference_progress: bool = False
    preemptive_replan: bool = False
    preemptive_replan_cooldown_steps: int = 3
    # Low-level continuous separation shield. It prevents a candidate edge
    # advance that would violate the same geometric clearance used by the
    # collision detector.
    physical_clearance_shield: bool = False
    # Squared graph-coordinate separation shared by the continuous shield,
    # node-clearance checks, and collision detector.
    physical_clearance_min_distance_sq: float = 3.0
    # Expose near-node continuous-clearance occupancy through the existing
    # per-neighbor risk feature and avoid forcing reference-path progress into
    # that protected region.
    clearance_risk_observation: bool = False
    clearance_aware_reference_progress: bool = False
    # Keep aircraft at the upstream node when the downstream clearance area is
    # occupied or another aircraft has already entered its final approach.
    # This avoids converting a node queue into an edge-blocking queue.
    pre_entry_clearance_guard: bool = False
    approach_reservation_progress: float = 0.60


@dataclass
class RewardConfig:
    """奖励配置。"""
    completion: float = 50.0        # 完成奖励（提升正向信号主导）
    conflict: float = -5.0          # 冲突重罚（不太轻、不太重的平衡点）
    wait: float = -0.2              # 适度等待惩罚
    proximity: float = 0.8          # 节点跳跃式靠近奖
    step_penalty: float = -0.1      # 每步固定惩罚
    path_deviation: float = -0.5    # 偏离先验路径惩罚
    efficiency_rate: float = 0.1    # 效率奖励系数（×剩余步数）
    deadlock: float = -2.0          # 训练期停滞恢复信号，不是运行性能指标
    team_bonus: float = 20.0        # 团队完成奖励
    yield_bonus: float = 0.3        # 让行成功奖励：主动等待且对方实际推进
    blocking: float = -0.2          # 阻塞惩罚
    team_progress: float = 10.0     # 团队进度奖
    # Terminal penalty for active aircraft that remain unfinished when the
    # episode horizon is reached. Zero preserves legacy behavior.
    timeout: float = 0.0
    # Extra penalty only when the policy chooses WAIT at a node despite at
    # least one mask-legal, immediately clear route. It does not punish a
    # forced hold caused by a closure, occupancy, or corridor reservation.
    avoidable_wait: float = 0.0


@dataclass
class TrainingConfig:
    """训练配置。"""
    total_episodes: int = 50000
    batch_size: int = 256
    buffer_size: int = 10000
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_mini_batches: int = 4
    hidden_dim: int = 128
    num_envs: int = 8
    save_interval: int = 1000
    log_interval: int = 100
    enable_lr_decay: bool = True
    # Deterministic evaluation used for model selection and curriculum handoff.
    eval_interval: int = 500
    eval_episodes: int = 20
    # Kept fixed across checkpoints so evaluation curves measure policy change
    # rather than a changing sample of random flight assignments.
    eval_seed: int = 100042
    # Independent final report set. It is never used to select a checkpoint.
    holdout_eval_seed: int = 200042
    holdout_eval_episodes: int = 100
    # Robust checkpoint selection on the fixed evaluation set.
    selection_full_weight: float = 0.5
    selection_timeout_weight: float = 0.2
    selection_crash_weight: float = 2.0
    # Credit assignment: 0 keeps pure team-MAPPO, values in (0, 1) blend a
    # local return signal into the actor advantage while the critic remains
    # trained on the centralized team return.
    individual_advantage_weight: float = 0.0
    # Set to a positive value to stop PPO epochs once the sampled KL estimate
    # exceeds the threshold. Zero disables early stopping.
    target_kl: float = 0.0
    # Number of independent episodes collected before one PPO update. Zero
    # uses num_envs as the synchronous rollout-batch size.
    rollouts_per_update: int = 0
    # Deterministic action-level audit, written at each evaluation checkpoint.
    # Zero disables it. The audit does not contribute to model selection.
    action_diagnostic_episodes: int = 0
    action_diagnostic_trace_episodes: int = 2


@dataclass
class TestConfig:
    """测试配置。"""
    num_episodes: int = 50
    model_path: str = "checkpoints/latest.pt"


@dataclass
class Config:
    """全局配置容器。"""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test: TestConfig = field(default_factory=TestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        """从 YAML 文件加载配置。

        Args:
            path: YAML 文件路径。为 None 时加载默认配置。

        Returns:
            Config 实例。
        """
        if path is None:
            #path = Path(__file__).parent / "default.yaml"
            path = Path(__file__).parent / "improved_v2.yaml"

        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        cfg = cls()
        if "env" in raw:
            cfg.env = EnvConfig(**raw["env"])
        if "reward" in raw:
            cfg.reward = RewardConfig(**raw["reward"])
        if "training" in raw:
            cfg.training = TrainingConfig(**raw["training"])
        if "test" in raw:
            cfg.test = TestConfig(**raw["test"])
        return cfg
