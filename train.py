"""训练主程序。

初始化环境、创建智能体、循环训练、保存模型、输出分项奖励监控。
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo import MAPPOTrainer


# 奖励分项键列表（与 marl_env._compute_rewards 中 rb 键名一致）
REWARD_KEYS = [
    "step", "completion", "efficiency", "proximity", "conflict", "wait",
    "speed_dev", "path_dev", "deadlock", "team_bonus", "team_prog",
    "yield", "block", "danger", "timeout",
]


def train(config_path: str | None = None) -> None:
    """运行 MAPPO 训练。

    Args:
        config_path: 配置文件路径。
    """
    config = Config.from_yaml(config_path)
    tc = config.training

    env = MARLEnv(config)

    # 可视化网络图
    print("Visualizing taxiway network...")
    env.graph_env.visualize_network()

    trainer = MAPPOTrainer(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=tc.hidden_dim,
        lr_actor=tc.lr_actor,
        lr_critic=tc.lr_critic,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        clip_epsilon=tc.clip_epsilon,
        entropy_coef=tc.entropy_coef,
        value_loss_coef=tc.value_loss_coef,
        max_grad_norm=tc.max_grad_norm,
        update_epochs=tc.update_epochs,
        num_mini_batches=tc.num_mini_batches,
        enable_lr_decay=True,
    )

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MAPPO Training - Airport Taxiway Multi-Agent RL")
    print(f"  Agents: {env.num_agents}")
    print(f"  Obs dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Global state dim: {env.global_state_dim}")
    print(f"  Total episodes: {tc.total_episodes}")
    print("=" * 70)
    # 打印奖励配置概览
    rc = config.reward
    print(f"  Reward | completion={rc.completion} team_bonus={rc.team_bonus} efficiency_rate={rc.efficiency_rate}"
          f"  proximity={rc.proximity}")
    print(f"         | step={rc.step_penalty}  conflict={rc.conflict}"
          f"  wait={rc.wait}  path_dev={rc.path_deviation}")
    print("=" * 70)

    episode_rewards_history: list[float] = []
    # 分项累积历史（每 episode 的分项总和），用于绘图
    breakdown_history: dict[str, list[float]] = {k: [] for k in REWARD_KEYS}

    # CSV 日志文件
    csv_path = save_dir / "training_log.csv"
    _init_csv(csv_path)
    start_time = time.time()

    # 死锁位置热力图：按 (node_or_edge) -> 次数 累积
    from collections import Counter
    deadlock_heatmap: Counter = Counter()

    for episode in range(1, tc.total_episodes + 1):
        obs, masks = env.reset()
        global_state = env.get_global_state()

        episode_reward = 0.0
        done_all = False
        step_count = 0
        # 本 episode 各子项累计
        ep_breakdown: dict[str, float] = {k: 0.0 for k in REWARD_KEYS}

        while not done_all:
            actions, log_probs, value = trainer.select_actions(obs, masks, global_state)

            next_obs, rewards, dones, next_masks, info = env.step(actions)
            next_global_state = env.get_global_state()

            trainer.store_transition(
                obs, global_state, actions, masks, rewards, dones, log_probs, value
            )

            # 累积分项奖励
            step_rb = info.get("reward_breakdown", {})
            for k in REWARD_KEYS:
                ep_breakdown[k] += step_rb.get(k, 0.0)

            obs = next_obs
            masks = next_masks
            global_state = next_global_state
            episode_reward += sum(rewards)
            step_count += 1
            # 更新真实完成和撞机统计（以最新的 step info 为准）
            ep_goal_reached = info.get("goal_reached", [False] * env.num_agents)
            ep_crashed = info.get("crashed", [False] * env.num_agents)

            done_all = all(dones) or step_count >= config.env.max_steps

        # 收集本 episode 的死锁事件到热力图
        for evt in info.get("deadlock_events", []):
            if evt["on_edge"] and evt["edge"]:
                key = f"edge({evt['edge'][0]}->{evt['edge'][1]})"
            else:
                key = f"node({evt['node']})"
            deadlock_heatmap[key] += 1

        # 更新网络
        metrics = trainer.update(episode=episode, total_episodes=tc.total_episodes)
        episode_rewards_history.append(episode_reward)
        for k in REWARD_KEYS:
            breakdown_history[k].append(ep_breakdown[k])

        # 真正到达目标的数量（非 crash 导致的 done）
        goal_count = sum(1 for g in ep_goal_reached if g)
        crash_count = sum(1 for c in ep_crashed if c)
        _append_csv(csv_path, episode, step_count, episode_reward, goal_count,
                    ep_breakdown, metrics)

        # 日志输出
        if episode % tc.log_interval == 0:
            avg_reward = np.mean(episode_rewards_history[-100:])
            # 近 log_interval 个 episode 的各子项均值
            n = min(tc.log_interval, len(episode_rewards_history))
            avg_bd = {k: np.mean(breakdown_history[k][-n:]) for k in REWARD_KEYS}

            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            print(
                f"Episode {episode:>6d} | "
                f"Steps: {step_count:>4d} | "
                f"Reward: {episode_reward:>8.1f} | "
                f"Avg100: {avg_reward:>8.1f} | "
                f"Goal: {goal_count}/{env.num_agents} Crash: {crash_count} | "
                f"EPS: {eps_per_sec:.2f} | "
                f"A_loss: {metrics.get('actor_loss', 0):.4f} | "
                f"C_loss: {metrics.get('critic_loss', 0):.4f}"
            )
            print(
                f"         Sub-rewards(Avg{n}) | "
                f"comp={avg_bd['completion']:>7.2f} | "
                f"team={avg_bd['team_bonus']:>6.2f} | "
                f"eff={avg_bd['efficiency']:>6.2f} | "
                f"prox={avg_bd['proximity']:>7.2f} | "
                f"step={avg_bd['step']:>7.2f} | "
                f"conf={avg_bd['conflict']:>7.2f} | "
                f"wait={avg_bd['wait']:>7.2f} | "
                f"spd={avg_bd['speed_dev']:>6.2f} | "
                f"pdev={avg_bd['path_dev']:>7.2f} | "
                f"deadl={avg_bd['deadlock']:>6.2f}"
            )

        # 保存模型 + 打印死锁热点
        if episode % tc.save_interval == 0:
            path = str(save_dir / f"model_ep{episode}.pt")
            trainer.save(path)
            trainer.save(str(save_dir / "latest.pt"))
            print(f"  -> Model saved to {path}")

            # 死锁热点 Top 10
            if deadlock_heatmap:
                print(f"  ** Deadlock Hotspots (last {tc.save_interval} eps) **")
                for loc, cnt in deadlock_heatmap.most_common(10):
                    print(f"     {loc}: {cnt} times")
                deadlock_heatmap.clear()


    # 最终保存
    trainer.save(str(save_dir / "latest.pt"))
    print("\nTraining completed. Final model saved.")

    # 绘制奖励曲线（总奖励 + 各子项）
    _plot_reward_curve(episode_rewards_history, breakdown_history, save_dir)


def _init_csv(path: Path) -> None:
    """初始化 CSV 日志文件，写入表头。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "steps", "reward", "avg100", "done_count",
            *REWARD_KEYS,
            "actor_loss", "critic_loss", "entropy",
        ])


def _append_csv(
    path: Path,
    episode: int,
    steps: int,
    reward: float,
    done_count: int,
    breakdown: dict[str, float],
    metrics: dict[str, float],
) -> None:
    """追加一行 CSV 记录。"""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode, steps, f"{reward:.2f}", "",  # avg100 在打印时计算，CSV留空
            done_count,
            *[f"{breakdown.get(k, 0.0):.4f}" for k in REWARD_KEYS],
            f"{metrics.get('actor_loss', 0):.6f}",
            f"{metrics.get('critic_loss', 0):.6f}",
            f"{metrics.get('entropy', 0):.6f}",
        ])


def _plot_reward_curve(
    rewards: list[float],
    breakdown_history: dict[str, list[float]],
    save_dir: Path,
) -> None:
    """绘制并保存训练奖励曲线，包含总奖励和各子项曲线。

    Args:
        rewards: 每回合总奖励列表。
        breakdown_history: 每回合各子项奖励列表。
        save_dir: 保存目录。
    """
    try:
        import matplotlib.pyplot as plt

        window = 100

        def moving_avg(data: list[float]) -> list[float]:
            if len(data) < window:
                return []
            return [np.mean(data[max(0, i - window):i]) for i in range(1, len(data) + 1)]

        # ---- 图1：总奖励 ----
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
        avg = moving_avg(rewards)
        if avg:
            ax.plot(avg, color="orangered", linewidth=2, label=f"{window}-ep Moving Avg")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("MAPPO Training Reward Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(str(save_dir / "reward_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- 图2：奖励分项曲线（2×4 子图） ----
        positive_keys = ["completion", "efficiency", "proximity"]
        negative_keys = ["step", "conflict", "wait", "speed_dev", "path_dev"]
        all_plot_keys = positive_keys + negative_keys

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes_flat = axes.flatten()

        colors = {
            "completion": "green", "efficiency": "limegreen", "proximity": "teal",
            "step": "steelblue", "conflict": "crimson", "wait": "orange",
            "speed_dev": "purple", "path_dev": "brown",
        }

        for idx, key in enumerate(all_plot_keys):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            data = breakdown_history.get(key, [])
            ax.plot(data, alpha=0.3, color=colors.get(key, "gray"))
            avg = moving_avg(data)
            if avg:
                ax.plot(avg, color=colors.get(key, "gray"), linewidth=1.5)
            ax.set_title(key)
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)

        # 隐藏多余子图
        for idx in range(len(all_plot_keys), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Reward Sub-components (Moving Avg)", fontsize=14)
        fig.tight_layout()
        fig.savefig(str(save_dir / "reward_breakdown_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Reward curves saved to {save_dir}/reward_curve.png & reward_breakdown_curve.png")
    except ImportError:
        print("Warning: matplotlib not installed. Skipping reward curve plot.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPPO Training for Airport Taxiway MARL")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. Defaults to configs/default.yaml."
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
