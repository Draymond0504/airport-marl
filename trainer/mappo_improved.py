"""改进版 MAPPO 训练器 - 添加学习率衰减和增强稳定性。

相比原版的改进:
1. 添加学习率衰减策略
2. 改进奖励标准化
3. 分离 Actor/Critic 更新 (可选)
4. 更好的梯度监控
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """去中心化策略网络。"""

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
        """前向传播。"""
        logits = self.net(obs)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e8)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """中心化价值网络。"""

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
        """前向传播。"""
        return self.net(state)


class RolloutBuffer:
    """轨迹缓存。"""

    def __init__(self) -> None:
        self.observations: list[list[np.ndarray]] = []
        self.global_states: list[np.ndarray] = []
        self.next_global_states: list[np.ndarray] = []
        self.actions: list[list[int]] = []
        self.action_masks: list[list[np.ndarray]] = []
        self.active_masks: list[list[bool]] = []
        self.rewards: list[list[float]] = []
        self.dones: list[list[bool]] = []
        self.log_probs: list[list[float]] = []
        self.values: list[float] = []

    def add(
        self,
        obs: list[np.ndarray],
        global_state: np.ndarray,
        next_global_state: np.ndarray,
        actions: list[int],
        action_masks: list[np.ndarray],
        active_masks: list[bool],
        rewards: list[float],
        dones: list[bool],
        log_probs: list[float],
        value: float,
    ) -> None:
        """添加一步数据。"""
        self.observations.append(obs)
        self.global_states.append(global_state)
        self.next_global_states.append(next_global_state)
        self.actions.append(actions)
        self.action_masks.append(action_masks)
        self.active_masks.append(active_masks)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(value)

    def clear(self) -> None:
        """清空缓存。"""
        self.observations.clear()
        self.global_states.clear()
        self.next_global_states.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.active_masks.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.observations)


class MAPPOTrainerImproved:
    """改进版 MAPPO 训练器。

    改进点：
    - 学习率衰减
    - 更好的奖励标准化
    - 可选的 Actor/Critic 分离更新
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
        individual_advantage_weight: float = 0.0,
        target_kl: float = 0.0,
        critic_state_schema: str = "legacy_v1",
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
        self.individual_advantage_weight = float(
            np.clip(individual_advantage_weight, 0.0, 1.0)
        )
        self.target_kl = max(float(target_kl), 0.0)
        self.critic_state_schema = str(critic_state_schema)
        
        # 记录初始学习率（用于衰减）
        self.lr_actor_init = lr_actor
        self.lr_critic_init = lr_critic

        # 共享参数的 Actor
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(global_state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()
        
        # 训练步数计数
        self.update_step = 0

    def select_actions(
        self,
        observations: list[np.ndarray],
        action_masks: list[np.ndarray],
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[list[int], list[float], float]:
        """选择动作和计算价值。"""
        self.actor.eval()
        self.critic.eval()

        actions = []
        log_probs_list = []

        with torch.no_grad():
            # 将所有智能体的观测和掩码打包成 Batch 一次性搬运到 GPU，减少 PCIe 传输开销
            obs_batch = torch.FloatTensor(np.array(observations)).to(self.device)
            mask_batch = torch.FloatTensor(np.array(action_masks)).to(self.device)
            
            # 一次性前向传播得到所有分布
            dist = self.actor(obs_batch, mask_batch)
            
            if deterministic:
                actions_t = torch.argmax(dist.probs, dim=-1)
            else:
                actions_t = dist.sample()
                
            log_probs_t = dist.log_prob(actions_t)
            
            # 将结果转回 CPU (仅需一次搬运)
            actions = actions_t.cpu().tolist()
            log_probs_list = log_probs_t.cpu().tolist()

            state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = self.critic(state_t).item()

        return actions, log_probs_list, value

    def action_probabilities(
        self,
        observations: list[np.ndarray],
        action_masks: list[np.ndarray],
    ) -> np.ndarray:
        """Return masked policy probabilities for action-level diagnostics.

        This method is intentionally separate from ``select_actions`` so the
        normal PPO collection path keeps its existing return contract and does
        not pay an unnecessary CPU transfer cost.
        """
        self.actor.eval()
        with torch.no_grad():
            obs_batch = torch.FloatTensor(np.array(observations)).to(self.device)
            mask_batch = torch.FloatTensor(np.array(action_masks)).to(self.device)
            probabilities = self.actor(obs_batch, mask_batch).probs
        return probabilities.cpu().numpy()

    def store_transition(
        self,
        obs: list[np.ndarray],
        global_state: np.ndarray,
        next_global_state: np.ndarray,
        actions: list[int],
        action_masks: list[np.ndarray],
        active_masks: list[bool],
        rewards: list[float],
        dones: list[bool],
        log_probs: list[float],
        value: float,
    ) -> None:
        """存储一步转移。"""
        self.buffer.add(
            obs,
            global_state,
            next_global_state,
            actions,
            action_masks,
            active_masks,
            rewards,
            dones,
            log_probs,
            value,
        )

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 Generalized Advantage Estimation。"""
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

    def _apply_lr_decay(self, episode: int, total_episodes: int) -> None:
        """应用学习率衰减策略。
        
        线性衰减: lr = lr_init * (1 - episode / total_episodes)
        """
        if not self.enable_lr_decay or total_episodes == 0:
            return
        
        decay_factor = 1.0 - episode / total_episodes
        new_lr_actor = self.lr_actor_init * decay_factor
        new_lr_critic = self.lr_critic_init * decay_factor
        
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr_actor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr_critic

    def set_learning_rates(self, actor_lr: float, critic_lr: float) -> None:
        """Apply configured rates after loading an optimizer checkpoint."""
        self.lr_actor_init = float(actor_lr)
        self.lr_critic_init = float(critic_lr)
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.lr_actor_init
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.lr_critic_init

    def update(self, episode: int = 0, total_episodes: int = 0) -> dict[str, float]:
        """更新策略和价值网络。
        
        Args:
            episode: 当前 episode 号（用于学习率衰减）
            total_episodes: 总 episode 数（用于学习率衰减）
        
        Returns:
            训练指标字典。
        """
        if len(self.buffer) == 0:
            return {}

        # 应用学习率衰减
        if episode > 0 and total_episodes > 0:
            self._apply_lr_decay(episode, total_episodes)

        T = len(self.buffer)

        # 整理数据
        all_obs = []
        all_actions = []
        all_masks = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_states = []
        all_agent_ids = []

        values_np = np.array(self.buffer.values, dtype=np.float32)
        with torch.no_grad():
            pass

        # 计算 rollout 尾部的 bootstrap value。必须使用 step 后的 next_global_state，
        # 否则 GAE 会把动作前状态错当成尾状态，长 episode 中偏差会被放大。
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.next_global_states[-1]).unsqueeze(0).to(self.device)
            next_value = self.critic(last_state).item()

        team_rewards = np.array(
            [sum(self.buffer.rewards[t]) / max(self.num_agents, 1) for t in range(T)],
            dtype=np.float32,
        )
        team_dones = np.array(
            [all(self.buffer.dones[t]) for t in range(T)],
            dtype=np.float32,
        )
        team_advantages, team_returns = self.compute_gae(
            team_rewards, values_np, team_dones, next_value
        )

        for agent_id in range(self.num_agents):
            agent_rewards = np.array(
                [self.buffer.rewards[t][agent_id] for t in range(T)], dtype=np.float32
            )
            agent_dones = np.array(
                [self.buffer.dones[t][agent_id] for t in range(T)], dtype=np.float32
            )

            local_advantages, _ = self.compute_gae(
                agent_rewards, values_np, agent_dones, next_value
            )

            for t in range(T):
                if not self.buffer.active_masks[t][agent_id]:
                    continue
                all_obs.append(self.buffer.observations[t][agent_id])
                all_actions.append(self.buffer.actions[t][agent_id])
                all_masks.append(self.buffer.action_masks[t][agent_id])
                all_old_log_probs.append(self.buffer.log_probs[t][agent_id])
                # The centralized critic is fitted to the team return, while
                # this local term gives an individual aircraft direct credit
                # for a safe completion, collision, or blocking decision.
                # Blending preserves cooperative optimization without making
                # every aircraft learn from an almost identical signal.
                blended_advantage = (
                    (1.0 - self.individual_advantage_weight) * team_advantages[t]
                    + self.individual_advantage_weight * local_advantages[t]
                )
                all_advantages.append(blended_advantage)
                all_returns.append(team_returns[t])
                all_states.append(self.buffer.global_states[t])
                all_agent_ids.append(agent_id)

        if not all_obs:
            self.buffer.clear()
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # 转为 tensor
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(self.device)
        action_tensor = torch.LongTensor(all_actions).to(self.device)
        mask_tensor = torch.FloatTensor(np.array(all_masks)).to(self.device)
        old_log_prob_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        return_tensor = torch.FloatTensor(all_returns).to(self.device)
        adv_np = np.array(all_advantages, dtype=np.float32)

        # Per-agent 独立 Advantage 标准化，只基于真实参与决策的样本。
        adv_np = np.array(all_advantages, dtype=np.float32)
        for a in range(self.num_agents):
            idx_a = np.array([idx for idx, aid in enumerate(all_agent_ids) if aid == a], dtype=np.int64)
            if len(idx_a) == 0:
                continue
            mean = adv_np[idx_a].mean()
            std = adv_np[idx_a].std() + 1e-8
            adv_np[idx_a] = (adv_np[idx_a] - mean) / std
        advantage_tensor = torch.FloatTensor(adv_np).to(self.device)
        state_tensor = torch.FloatTensor(np.array(all_states, dtype=np.float32)).to(self.device)

        # 全局状态扩展
        state_tensor = torch.FloatTensor(np.array(all_states, dtype=np.float32)).to(self.device)

        total_samples = len(obs_tensor)
        batch_size = max(total_samples // self.num_mini_batches, 1)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        update_count = 0
        early_stopped = False

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

                # A non-negative sampled approximation of KL(old || new).
                log_ratio = new_log_probs - b_old_log_probs
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                if self.target_kl > 0 and approx_kl.item() > self.target_kl:
                    early_stopped = True
                    break

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
                total_kl += approx_kl.item()
                update_count += 1

            if early_stopped:
                break

        self.buffer.clear()
        self.update_step += 1

        return {
            "actor_loss": total_actor_loss / max(update_count, 1),
            "critic_loss": total_critic_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
            "approx_kl": total_kl / max(update_count, 1),
            "early_stopped": float(early_stopped),
        }

    def save(self, path: str) -> None:
        """保存模型。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_state_schema": self.critic_state_schema,
            "critic_state_dim": self.global_state_dim,
        }, path)

    def load(self, path: str, load_critic: bool = True, load_optimizer: bool = True) -> None:
        """加载模型。
        
        Args:
            path: 模型文件路径。
            load_critic: 是否尝试加载 Critic 权重（课程学习跨阶段时由于 global_state_dim 改变，Critic 须重新初始化）。
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        
        # 兼容性加载对应优化器
        if load_optimizer and "actor_optimizer" in checkpoint and checkpoint["actor_optimizer"]:
            try:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            except Exception as e:
                print(f"[Trainer] Warning: Could not load Actor optimizer state. Error: {e}")

        checkpoint_schema = checkpoint.get("critic_state_schema", "legacy_v1")
        checkpoint_state_dim = checkpoint.get("critic_state_dim")
        critic_compatible = (
            checkpoint_schema == self.critic_state_schema
            and (checkpoint_state_dim is None or checkpoint_state_dim == self.global_state_dim)
        )
        if load_critic and "critic" in checkpoint and critic_compatible:
            try:
                self.critic.load_state_dict(checkpoint["critic"])
                if load_optimizer and "critic_optimizer" in checkpoint and checkpoint["critic_optimizer"]:
                    self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
                print("[Trainer] Successfully loaded both Actor and Critic weights.")
            except Exception as e:
                print(f"[Trainer] Notice: Critic load skipped due to param mismatch (expected in Curriculum Learning). Re-initializing Critic. MSG: {e}")
        else:
            if load_critic and "critic" in checkpoint and not critic_compatible:
                print(
                    "[Trainer] Critic state schema changed "
                    f"({checkpoint_schema} -> {self.critic_state_schema}); "
                    "loaded actor only and initialized a fresh Critic."
                )
            else:
                print("[Trainer] Critic loading disabled or no critical payload found. Initialized fresh Critic for new curriculum stage.")
