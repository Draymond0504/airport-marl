"""MAPPO（Multi-Agent PPO）训练模块。

完整手动实现，不调用现成强化学习库。
包含 Actor 网络、Critic 网络、GAE 计算、PPO clip loss。
支持 centralized critic + decentralized actor。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ======================================================================
# 网络定义
# ======================================================================

class Actor(nn.Module):
    """去中心化策略网络。

    输入局部观测，输出动作概率分布。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> Categorical:
        """前向传播。

        Args:
            obs: 观测 (batch, obs_dim)。
            action_mask: 动作掩码 (batch, action_dim)，1 表示可执行。

        Returns:
            动作的 Categorical 分布。
        """
        logits = self.net(obs)
        if action_mask is not None:
            # 将不可执行动作的 logit 设为极小值
            logits = logits.masked_fill(action_mask == 0, -1e8)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """中心化价值网络。

    输入全局状态，输出状态价值。
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            state: 全局状态 (batch, state_dim)。

        Returns:
            状态价值 (batch, 1)。
        """
        return self.net(state)


# ======================================================================
# Rollout Buffer
# ======================================================================

class RolloutBuffer:
    """轨迹缓存。"""

    def __init__(self) -> None:
        self.observations: list[list[np.ndarray]] = []      # [step][agent]
        self.global_states: list[np.ndarray] = []            # [step]
        self.actions: list[list[int]] = []                   # [step][agent]
        self.action_masks: list[list[np.ndarray]] = []       # [step][agent]
        self.rewards: list[list[float]] = []                 # [step][agent]
        self.dones: list[list[bool]] = []                    # [step][agent]
        self.log_probs: list[list[float]] = []               # [step][agent]
        self.values: list[float] = []                        # [step]

    def add(
        self,
        obs: list[np.ndarray],
        global_state: np.ndarray,
        actions: list[int],
        action_masks: list[np.ndarray],
        rewards: list[float],
        dones: list[bool],
        log_probs: list[float],
        value: float,
    ) -> None:
        """添加一步数据。"""
        self.observations.append(obs)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.action_masks.append(action_masks)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(value)

    def clear(self) -> None:
        """清空缓存。"""
        self.observations.clear()
        self.global_states.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.observations)


# ======================================================================
# MAPPO Trainer
# ======================================================================

class MAPPOTrainer:
    """MAPPO 训练器。

    centralized critic + decentralized actors。
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        num_mini_batches: int = 4,
        device: str = "cpu",
        enable_lr_decay: bool = True,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_mini_batches = num_mini_batches
        self.device = torch.device(device)
        self.enable_lr_decay = enable_lr_decay
        self.lr_actor_init = lr_actor
        self.lr_critic_init = lr_critic

        # 共享参数的 Actor（所有智能体共享同一策略）
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(global_state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def select_actions(
        self,
        observations: list[np.ndarray],
        action_masks: list[np.ndarray],
        global_state: np.ndarray,
    ) -> tuple[list[int], list[float], float]:
        """根据当前策略选择动作。

        Args:
            observations: 每个智能体的观测。
            action_masks: 每个智能体的动作掩码。
            global_state: 全局状态。

        Returns:
            (actions, log_probs, value)
        """
        actions = []
        log_probs_list = []

        with torch.no_grad():
            for i in range(self.num_agents):
                obs_t = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                mask_t = torch.FloatTensor(action_masks[i]).unsqueeze(0).to(self.device)
                dist = self.actor(obs_t, mask_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action.item())
                log_probs_list.append(log_prob.item())

            state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = self.critic(state_t).item()

        return actions, log_probs_list, value

    def store_transition(
        self,
        obs: list[np.ndarray],
        global_state: np.ndarray,
        actions: list[int],
        action_masks: list[np.ndarray],
        rewards: list[float],
        dones: list[bool],
        log_probs: list[float],
        value: float,
    ) -> None:
        """存储一步转移。"""
        self.buffer.add(obs, global_state, actions, action_masks, rewards, dones, log_probs, value)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 Generalized Advantage Estimation。

        Args:
            rewards: (T,) 奖励序列。
            values: (T,) 值函数序列。
            dones: (T,) 终止标志。
            next_value: 下一步的值函数估计。

        Returns:
            (advantages, returns)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, episode: int = 0, total_episodes: int = 0) -> dict[str, float]:
        """更新策略和价值网络。

        Args:
            episode: 当前 episode 号（用于学习率线性衰减）。
            total_episodes: 总 episode 数。

        Returns:
            训练指标字典。
        """
        if len(self.buffer) == 0:
            return {}

        # 学习率线性衰减
        if self.enable_lr_decay and episode > 0 and total_episodes > 0:
            decay = max(1.0 - episode / total_episodes, 0.05)
            for pg in self.actor_optimizer.param_groups:
                pg['lr'] = self.lr_actor_init * decay
            for pg in self.critic_optimizer.param_groups:
                pg['lr'] = self.lr_critic_init * decay

        T = len(self.buffer)

        # 整理数据：按智能体分别计算 GAE
        all_obs = []        # (T * num_agents, obs_dim)
        all_actions = []    # (T * num_agents,)
        all_masks = []      # (T * num_agents, action_dim)
        all_old_log_probs = []  # (T * num_agents,)
        all_advantages = []     # (T * num_agents,)
        all_returns = []        # (T * num_agents,)
        all_states = []         # (T, state_dim)

        values_np = np.array(self.buffer.values, dtype=np.float32)
        all_states_np = np.array(self.buffer.global_states, dtype=np.float32)

        # 计算最后一步的 next_value
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.global_states[-1]).unsqueeze(0).to(self.device)
            next_value = self.critic(last_state).item()

        for agent_id in range(self.num_agents):
            agent_rewards = np.array(
                [self.buffer.rewards[t][agent_id] for t in range(T)], dtype=np.float32
            )
            agent_dones = np.array(
                [self.buffer.dones[t][agent_id] for t in range(T)], dtype=np.float32
            )

            advantages, returns = self.compute_gae(
                agent_rewards, values_np, agent_dones, next_value
            )

            for t in range(T):
                all_obs.append(self.buffer.observations[t][agent_id])
                all_actions.append(self.buffer.actions[t][agent_id])
                all_masks.append(self.buffer.action_masks[t][agent_id])
                all_old_log_probs.append(self.buffer.log_probs[t][agent_id])
                all_advantages.append(advantages[t])
                all_returns.append(returns[t])

        # 转为 tensor
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(self.device)
        action_tensor = torch.LongTensor(all_actions).to(self.device)
        mask_tensor = torch.FloatTensor(np.array(all_masks)).to(self.device)
        old_log_prob_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        return_tensor = torch.FloatTensor(all_returns).to(self.device)

        # ── per-agent 独立 Advantage 归一化 ──────────────────────────────
        # all_advantages 是按 [agent0 T步, agent1 T步, ...] 顺序排列的
        adv_np = np.array(all_advantages, dtype=np.float32)
        for a in range(self.num_agents):
            s, e = a * T, (a + 1) * T
            mean = adv_np[s:e].mean()
            std  = adv_np[s:e].std() + 1e-8
            adv_np[s:e] = (adv_np[s:e] - mean) / std
        advantage_tensor = torch.FloatTensor(adv_np).to(self.device)

        # 全局状态：每个 agent 共享同一 state，所以重复 num_agents 次
        states_expanded = np.repeat(all_states_np, self.num_agents, axis=0)
        state_tensor = torch.FloatTensor(states_expanded).to(self.device)

        total_samples = len(obs_tensor)
        batch_size = max(total_samples // self.num_mini_batches, 1)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                b_obs = obs_tensor[idx]
                b_actions = action_tensor[idx]
                b_masks = mask_tensor[idx]
                b_old_log_probs = old_log_prob_tensor[idx]
                b_advantages = advantage_tensor[idx]
                b_returns = return_tensor[idx]
                b_states = state_tensor[idx]

                # Actor loss
                dist = self.actor(b_obs, b_masks)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic loss
                values = self.critic(b_states).squeeze(-1)
                critic_loss = F.mse_loss(values, b_returns)

                # 更新 actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 更新 critic
                self.critic_optimizer.zero_grad()
                (self.value_loss_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        self.buffer.clear()

        return {
            "actor_loss": total_actor_loss / max(update_count, 1),
            "critic_loss": total_critic_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
        }

    def save(self, path: str) -> None:
        """保存模型。

        Args:
            path: 保存路径。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """加载模型。

        Args:
            path: 模型路径。
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
