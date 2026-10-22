"""改进版训练脚本 - 使用改进的配置和 MAPPO。

使用方法:
    python train_improved.py --config configs/improved_v2.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import time
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from configs.config import Config
from env.marl_env import MARLEnv
from micro_action_diagnostics import ActionDiagnosticCollector
from trainer.mappo_improved import MAPPOTrainerImproved


# Keep the logged decomposition aligned with MARLEnv._compute_rewards.
# ``speed_dev`` remains for compatibility with older plotting scripts.
REWARD_KEYS = [
    "step", "completion", "efficiency", "proximity", "conflict", "wait",
    "speed_dev", "path_dev", "deadlock", "team_bonus", "team_prog",
    "yield", "block", "danger", "timeout", "avoidable_wait",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(
    trainer: MAPPOTrainerImproved,
    config: Config,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate the current actor without exploration or gradient updates."""
    if episodes <= 0:
        return {}

    # Evaluation must not perturb the RNG stream used by the next training episode.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        set_seed(seed)
        # Operational evaluation excludes training-only deadlock recovery.
        eval_config = copy.deepcopy(config)
        eval_config.env.enable_deadlock_recovery = False
        eval_env = MARLEnv(eval_config)
        safe_counts = []
        crash_counts = []

        step_counts = []
        full_success = 0
        timeout_episodes = 0

        for _ in range(episodes):
            obs, masks = eval_env.reset()
            global_state = eval_env.get_global_state()
            done_all = False
            step_count = 0
            info = {}
            dones = [False] * eval_env.num_agents

            while not done_all:
                actions, _, _ = trainer.select_actions(
                    obs, masks, global_state, deterministic=True
                )
                obs, _, dones, masks, info = eval_env.step(actions)
                global_state = eval_env.get_global_state()
                step_count += 1
                done_all = all(dones) or step_count >= config.env.max_steps

            goals = info.get("goal_reached", [False] * eval_env.num_agents)
            crashed = info.get("crashed", [False] * eval_env.num_agents)
            safe_count = sum(bool(value) for value in goals)
            crash_count = sum(bool(value) for value in crashed)

            unfinished = sum(not bool(value) for value in dones)

            safe_counts.append(safe_count)
            crash_counts.append(crash_count)

            step_counts.append(step_count)
            full_success += int(safe_count == eval_env.num_agents)
            timeout_episodes += int(step_count >= config.env.max_steps and unfinished > 0)

        total_tasks = max(episodes * eval_env.num_agents, 1)
        return {
            "eval_safe_completion_rate": 100.0 * sum(safe_counts) / total_tasks,
            "eval_full_success_rate": 100.0 * full_success / episodes,
            "eval_mean_completed": float(np.mean(safe_counts)),
            "eval_mean_crashes": float(np.mean(crash_counts)),

            "eval_mean_steps": float(np.mean(step_counts)),
            "eval_timeout_rate": 100.0 * timeout_episodes / episodes,
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_action_diagnostic(
    trainer: MAPPOTrainerImproved,
    config: Config,
    episodes: int,
    seed: int,
    output_dir: Path,
    trace_episodes: int,
) -> dict[str, object]:
    """Replay deterministic episodes and export micro-action explanations.

    The audit is deliberately separate from checkpoint selection. Its purpose
    is to identify whether a weak evaluation comes from forced infeasibility,
    voluntary waiting, or route choices that deviate from the A* reference.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        set_seed(seed)
        env = MARLEnv(config)
        collector = ActionDiagnosticCollector(trace_episodes=trace_episodes)
        collision_events: list[dict[str, object]] = []
        seen_collision_events = 0
        for episode in range(episodes):
            observations, masks = env.reset()
            global_state = env.get_global_state()
            collector.begin_episode(env.num_agents)
            steps = 0
            dones = [False] * env.num_agents
            info: dict[str, object] = {}

            while steps < config.env.max_steps and not all(dones):
                probabilities = trainer.action_probabilities(observations, masks)
                actions, _, _ = trainer.select_actions(
                    observations, masks, global_state, deterministic=True
                )
                pending_rows = collector.observe(
                    env, observations, masks, actions, probabilities, steps + 1
                )
                observations, rewards, dones, masks, info = env.step(actions)
                global_state = env.get_global_state()
                logged_events = info.get("collision_events", [])
                if len(logged_events) < seen_collision_events:
                    seen_collision_events = 0
                for event in logged_events[seen_collision_events:]:
                    payload = dict(event)
                    payload["episode"] = episode
                    collision_events.append(payload)
                seen_collision_events = len(logged_events)
                collector.finalize_step(env, pending_rows, rewards, episode)
                steps += 1

            graph = env.graph_env
            safe_completed = sum(
                graph.done[i] and not graph.crashed[i] and graph.pos[i] == graph.goal[i]
                for i in range(env.num_agents)
            )
            collector.finish_episode(
                episode=episode,
                steps=steps,
                safe_completed=int(safe_completed),
                crashed_agents=int(sum(graph.crashed)),
                timed_out=bool(steps >= config.env.max_steps and not all(dones)),
            )

        report = collector.export(
            output_dir,
            {
                "seed": seed,
                "deterministic": True,
                "num_agents": env.num_agents,
                "max_steps": config.env.max_steps,
                "layout": config.env.layout,
            },
        )
        collision_counts = Counter(
            str(event.get("type", "unknown")) for event in collision_events
        )
        report["collision_event_count"] = len(collision_events)
        report["collision_type_counts"] = dict(collision_counts)
        with (output_dir / "collision_events.json").open("w", encoding="utf-8") as handle:
            json.dump(collision_events, handle, ensure_ascii=False, indent=2)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def checkpoint_score(evaluation: dict[str, float], training_config) -> tuple[float, ...]:
    """Rank checkpoints using operational safety and completion metrics."""
    safe = float(evaluation.get("eval_safe_completion_rate", 0.0))
    full = float(evaluation.get("eval_full_success_rate", 0.0))
    timeout = float(evaluation.get("eval_timeout_rate", 100.0))
    crashes = float(evaluation.get("eval_mean_crashes", 0.0))

    composite = (
        safe
        + float(getattr(training_config, "selection_full_weight", 0.5)) * full
        - float(getattr(training_config, "selection_timeout_weight", 0.2)) * timeout
        - float(getattr(training_config, "selection_crash_weight", 2.0)) * crashes
    )
    return composite, safe, full, -timeout, -crashes


def train_improved(config_path: str | None = None, resume: str | None = None, override_config: Config | None = None, load_critic: bool = True, save_prefix: str = "", seed: int = 42, save_dir_override: str | None = None, visualize: bool = False, load_optimizer: bool = True, episodes_override: int | None = None) -> dict[str, object]:
    """运行改进版 MAPPO 训练。

    Args:
        config_path: 配置文件路径。
        resume: 可选，要从中继续训练的 checkpoint 路径（.pt 文件）。
    """
    if override_config is not None:
        config = override_config
    else:
        config = Config.from_yaml(config_path)
    tc = config.training
    if episodes_override is not None:
        if episodes_override <= 0:
            raise ValueError("episodes_override must be positive")
        tc.total_episodes = int(episodes_override)

    set_seed(seed)
    print(f"[Train] Random seed set to: {seed}")

    env = MARLEnv(config)

    # GPU / CPU 自动检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  |  显存: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

    # 可视化网络图
    if visualize:
        print("Visualizing taxiway network...")
        env.graph_env.visualize_network()

    trainer = MAPPOTrainerImproved(
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
        individual_advantage_weight=getattr(tc, "individual_advantage_weight", 0.0),
        target_kl=getattr(tc, "target_kl", 0.0),
        critic_state_schema=env.critic_state_schema,
        enable_lr_decay=True,
        device=device,           # GPU/CPU 自动切换
    )
    trainer.enable_lr_decay = getattr(tc, "enable_lr_decay", True)

    if save_dir_override:
        save_dir = Path(save_dir_override)
    else:
        save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 从 checkpoint 恢复（fine-tune 模式）
    start_episode = 1
    if resume:
        trainer.load(resume, load_critic=load_critic, load_optimizer=load_optimizer)
        # Checkpoints carry their old optimizer rates. Fine-tuning may use a
        # smaller rate, so restore the rates requested by the active config
        # while retaining Adam's accumulated moments.
        trainer.set_learning_rates(tc.lr_actor, tc.lr_critic)
        # 尝试从文件名中解析上次保存的 episode 号（如 model_ep45000.pt → 45000）
        import re
        m = re.search(r'ep(\d+)', Path(resume).stem)
        if m:
            start_episode = int(m.group(1)) + 1
        print(f"[Resume] 从 {resume} 加载成功，继续从 episode {start_episode} 开始训练")

    print("=" * 70)
    print("MAPPO Training - Airport Taxiway Multi-Agent RL (IMPROVED)")
    print(f"  Agents: {env.num_agents}")
    print(f"  Obs dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Global state dim: {env.global_state_dim}")
    print(f"  Total episodes: {tc.total_episodes}")
    print("=" * 70)
    
    # 打印奖励配置概览
    rc = config.reward
    print(f"  Reward Config:")
    print(f"    - completion={rc.completion}, team_bonus={rc.team_bonus}, efficiency={rc.efficiency_rate}")
    #print(f"    - proximity={rc.proximity}, step_penalty={rc.step_penalty}")
    print(f"    - conflict={rc.conflict}, deadlock={rc.deadlock}")
    print(f"  Learning Config:")
    print(f"    - lr_actor={tc.lr_actor}, lr_critic={tc.lr_critic}")
    print(f"    - entropy_coef={tc.entropy_coef}, gae_lambda={tc.gae_lambda}")
    print(f"    - clip_epsilon={tc.clip_epsilon}")
    print(
        "    - local_adv_weight="
        f"{getattr(tc, 'individual_advantage_weight', 0.0)}, "
        f"target_kl={getattr(tc, 'target_kl', 0.0)}"
    )
    print(
        "    - selection_eval="
        f"{getattr(tc, 'eval_episodes', 20)} episodes @ seed "
        f"{getattr(tc, 'eval_seed', 100042)}"
    )
    print("=" * 70)

    episode_rewards_history: list[float] = []
    breakdown_history: dict[str, list[float]] = {k: [] for k in REWARD_KEYS}
    rollouts_per_update = max(
        int(getattr(tc, "rollouts_per_update", 0) or tc.num_envs), 1
    )
    print(f"  Rollouts per PPO update: {rollouts_per_update}")

    # CSV 日志文件
    prefix = f"{save_prefix}_" if save_prefix else ""
    csv_path = save_dir / f"{prefix}training_log_improved.csv"
    _init_csv(csv_path, preserve_existing=resume is not None)
    eval_csv_path = save_dir / f"{prefix}evaluation_log.csv"
    _init_eval_csv(eval_csv_path, preserve_existing=resume is not None)

    # ``latest.pt`` does not encode an episode number. Recover it from the
    # existing log so a resumed run does not restart at episode 1.
    if resume and start_episode == 1:
        last_episode = _read_last_episode(csv_path)
        if last_episode is not None:
            start_episode = last_episode + 1
            print(f"[Resume] 从训练日志恢复 episode {start_episode}")

    best_eval_score: tuple[float, ...] | None = None
    best_eval_path: Path | None = None
    best_eval_summary: dict[str, float] | None = None

    # Preserve the previous best checkpoint when continuing the same stage.
    best_json_path = save_dir / f"{prefix}best_eval.json"
    if resume and best_json_path.exists():
        try:
            with best_json_path.open("r", encoding="utf-8") as f:
                previous_best = json.load(f)
            best_eval_score = checkpoint_score(previous_best, tc)
            best_eval_summary = previous_best
            previous_best_path = save_dir / f"{prefix}best_eval.pt"
            if previous_best_path.exists():
                best_eval_path = previous_best_path
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            pass

    start_time = time.time()

    # 训练恢复诊断热点
    from collections import Counter
    deadlock_heatmap: Counter = Counter()
    latest_metrics: dict[str, float] = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "early_stopped": 0.0,
    }

    for episode in range(start_episode, start_episode + tc.total_episodes):
        obs, masks = env.reset()
        global_state = env.get_global_state()

        episode_reward = 0.0
        done_all = False
        step_count = 0
        ep_breakdown: dict[str, float] = {k: 0.0 for k in REWARD_KEYS}

        while not done_all:
            active_masks = [
                (not env.graph_env.done[i]) and env.graph_env.start_delays[i] == 0
                for i in range(env.num_agents)
            ]
            actions, log_probs, value = trainer.select_actions(obs, masks, global_state)

            next_obs, rewards, dones, next_masks, info = env.step(actions)
            next_global_state = env.get_global_state()

            # Treat a time-limit truncation as terminal for this finite-horizon
            # objective. Otherwise GAE bootstraps from a state that belongs to
            # a new episode and biases the critic target.
            time_limit_reached = step_count + 1 >= config.env.max_steps
            transition_dones = list(dones)
            if time_limit_reached:
                transition_dones = [True] * env.num_agents

            trainer.store_transition(
                obs,
                global_state,
                next_global_state,
                actions,
                masks,
                active_masks,
                rewards,
                transition_dones,
                log_probs,
                value,
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

            done_all = all(dones) or time_limit_reached

        # 收集本 episode 的死锁事件到热力图
        for evt in info.get("deadlock_events", []):
            if evt["on_edge"] and evt["edge"]:
                key = f"edge({evt['edge'][0]}->{evt['edge'][1]})"
            else:
                key = f"node({evt['node']})"
            deadlock_heatmap[key] += 1

        # 更新网络（传入 episode 信息用于学习率衰减）
        completed_rollouts = episode - start_episode + 1
        is_final_episode = episode == start_episode + tc.total_episodes - 1
        eval_interval = int(getattr(tc, "eval_interval", 500))
        save_interval = int(getattr(tc, "save_interval", 1000))
        update_due = (
            completed_rollouts % rollouts_per_update == 0
            or is_final_episode
            or (eval_interval > 0 and episode % eval_interval == 0)
            or (save_interval > 0 and episode % save_interval == 0)
        )
        if update_due:
            latest_metrics = trainer.update(
                episode=episode,
                total_episodes=tc.total_episodes,
            )
        metrics = latest_metrics
        episode_rewards_history.append(episode_reward)
        for k in REWARD_KEYS:
            breakdown_history[k].append(ep_breakdown[k])

        # 真正到达目标的数量（非 crash 导致的 done）
        goal_count = sum(1 for g in ep_goal_reached if g)
        crash_count = sum(1 for c in ep_crashed if c)
        eval_result: dict[str, float] = {}
        eval_episodes = int(getattr(tc, "eval_episodes", 20))
        is_eval_episode = (
            eval_interval > 0
            and (episode % eval_interval == 0 or is_final_episode)
        )
        if is_eval_episode:
            eval_result = evaluate_policy(
                trainer,
                config,
                episodes=eval_episodes,
                seed=int(getattr(tc, "eval_seed", 100042)),
            )
            _append_eval_csv(eval_csv_path, episode, eval_result)

            diagnostic_episodes = int(
                getattr(tc, "action_diagnostic_episodes", 0)
            )
            if diagnostic_episodes > 0:
                diagnostic_dir = (
                    save_dir / f"{prefix}micro_diagnostics" / f"eval_ep{episode}"
                )
                try:
                    action_report = run_action_diagnostic(
                        trainer,
                        config,
                        episodes=diagnostic_episodes,
                        seed=int(getattr(tc, "eval_seed", 100042)),
                        output_dir=diagnostic_dir,
                        trace_episodes=int(
                            getattr(tc, "action_diagnostic_trace_episodes", 2)
                        ),
                    )
                    print(
                        "  [Micro diagnostic] "
                        f"avoidable_wait={action_report['avoidable_wait_pct']:.1f}% | "
                        f"forced_wait={action_report['forced_wait_pct']:.1f}% | "
                        f"route_choice={action_report['route_choice_pct']:.1f}% | "
                        f"edge_hold={action_report['edge_no_progress_pct']:.1f}% | "
                        f"traces={action_report['representative_episodes']}"
                    )
                except OSError as exc:
                    # Diagnostics must never discard a completed PPO update
                    # because a CSV/JSON file is open in another application.
                    print(f"  [Micro diagnostic] skipped: {exc}")

            score = checkpoint_score(eval_result, tc)
            if best_eval_score is None or score > best_eval_score:
                best_eval_score = score
                best_eval_summary = dict(eval_result)
                best_eval_summary["episode"] = float(episode)
                best_eval_summary["selection_score"] = float(score[0])
                best_eval_path = save_dir / f"{prefix}best_eval.pt"
                trainer.save(str(best_eval_path))
                with best_eval_path.with_suffix(".json").open("w", encoding="utf-8") as f:
                    json.dump(best_eval_summary, f, ensure_ascii=False, indent=2)
                print(
                    f"  -> Best eval model saved: {best_eval_path} | "
                    f"score={score[0]:.2f} safe={score[1]:.1f}% | "
                    f"full={score[2]:.1f}% timeout={-score[3]:.1f}%"
                )

        _append_csv(csv_path, episode, step_count, episode_reward, goal_count,
                    ep_breakdown, metrics, eval_result)

        # 日志输出
        if episode % tc.log_interval == 0:
            avg_reward = np.mean(episode_rewards_history[-100:])
            n = min(tc.log_interval, len(episode_rewards_history))
            avg_bd = {k: np.mean(breakdown_history[k][-n:]) for k in REWARD_KEYS}
            
            elapsed = time.time() - start_time
            episodes_per_sec = episode / elapsed if elapsed > 0 else 0

            print(
                f"Episode {episode:>6d} | "
                f"Steps: {step_count:>4d} | "
                f"Reward: {episode_reward:>8.1f} | "
                f"Avg100: {avg_reward:>8.1f} | "
                f"Goal: {goal_count}/{env.num_agents} Crash: {crash_count} | "
                f"Episodes/s: {episodes_per_sec:.2f}"
            )
            print(
                f"         Loss: A={metrics.get('actor_loss', 0):.4f} | "
                f"C={metrics.get('critic_loss', 0):.4f} | "
                f"Ent={metrics.get('entropy', 0):.4f} | "
                f"KL={metrics.get('approx_kl', 0):.4f}"
            )
            print(
                f"         Rewards: comp={avg_bd['completion']:>7.2f} | "
                f"team={avg_bd['team_bonus']:>6.2f} | "
                f"eff={avg_bd['efficiency']:>6.2f} | "
                f"prox={avg_bd['proximity']:>7.2f} | "
                f"deadlock={avg_bd['deadlock']:>7.2f} | "
                f"conflict={avg_bd['conflict']:>7.2f} | "
                f"avoidWait={avg_bd['avoidable_wait']:>7.2f}"

            )

        # 定期保存模型 + 打印死锁热点
        if episode % tc.save_interval == 0 or episode == start_episode + tc.total_episodes - 1:
            model_path = save_dir / f"{prefix}model_ep{episode}.pt"
            trainer.save(str(model_path))
            latest_path = save_dir / f"{prefix}latest.pt"
            trainer.save(str(latest_path))
            print(f"  -> Model saved to {model_path}")

            # 死锁热点 Top 10（按用户要求：不再全局累计，每 1000 轮清空一次）
            if deadlock_heatmap:
                print(f"  ** Training Recovery Hotspots (top 10, ep {max(0, episode-tc.save_interval)} to {episode}) **")
                for loc, cnt in deadlock_heatmap.most_common(10):
                    print(f"     {loc}: {cnt} times")
                deadlock_heatmap.clear()

    holdout_result: dict[str, float] | None = None
    holdout_path: Path | None = None
    if best_eval_path:
        # The holdout set is intentionally independent of checkpoint selection.
        # Reload the selected model because the in-memory trainer may contain a
        # later, worse policy after the best checkpoint was saved.
        trainer.load(str(best_eval_path), load_critic=True, load_optimizer=False)
        holdout_result = evaluate_policy(
            trainer,
            config,
            episodes=int(getattr(tc, "holdout_eval_episodes", 100)),
            seed=int(getattr(tc, "holdout_eval_seed", 200042)),
        )
        holdout_path = save_dir / f"{prefix}best_eval_holdout.json"
        with holdout_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "selection": best_eval_summary,
                    "selection_seed": int(getattr(tc, "eval_seed", 100042)),
                    "selection_episodes": int(getattr(tc, "eval_episodes", 20)),
                    "holdout": holdout_result,
                    "holdout_seed": int(getattr(tc, "holdout_eval_seed", 200042)),
                    "holdout_episodes": int(getattr(tc, "holdout_eval_episodes", 100)),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training completed! Total time: {total_time / 3600:.2f} hours")
    print(f"Logs saved to: {csv_path}")
    print(f"Evaluation logs saved to: {eval_csv_path}")
    if best_eval_path:
        print(f"Best evaluation model: {best_eval_path}")
    if holdout_result and holdout_path:
        print(
            "Best-model holdout: "
            f"safe={holdout_result['eval_safe_completion_rate']:.1f}% | "
            f"full={holdout_result['eval_full_success_rate']:.1f}% | "
            f"timeout={holdout_result['eval_timeout_rate']:.1f}%"
        )
        print(f"Holdout report saved to: {holdout_path}")
    print("=" * 70)

    return {
        "best_eval_path": str(best_eval_path) if best_eval_path else None,
        "best_eval": best_eval_summary,
        "best_holdout": holdout_result,
        "training_log": str(csv_path),
        "evaluation_log": str(eval_csv_path),
    }



def _init_csv(csv_path: Path, preserve_existing: bool = False) -> None:
    """初始化 CSV 文件。"""
    header = [
        'episode', 'steps', 'total_reward', 'completed', 'actor_loss',
        'critic_loss', 'entropy', 'approx_kl', 'early_stopped',
    ] + REWARD_KEYS + [
        'eval_safe_completion_rate', 'eval_full_success_rate', 'eval_mean_completed',
        'eval_mean_crashes', 'eval_mean_steps', 'eval_timeout_rate'
    ]
    if preserve_existing and csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows and rows[0] == header:
                return
            if rows:
                legacy_path = csv_path.with_name(csv_path.stem + "_legacy.csv")
                if not legacy_path.exists():
                    shutil.copy2(csv_path, legacy_path)
                old_header = rows[0]
                data_rows = [
                    [dict(zip(old_header, row)).get(key, "") for key in header]
                    for row in rows[1:]
                ]
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + data_rows)
                return
        except (OSError, UnicodeError):
            pass
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)


def _append_csv(
    csv_path: Path,
    episode: int,
    steps: int,
    total_reward: float,
    completed: int,
    breakdown: dict[str, float],
    metrics: dict[str, float],
    evaluation: dict[str, float] | None = None,
) -> None:
    """追加一行到 CSV 文件。"""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = [
            episode,
            steps,
            f"{total_reward:.4f}",
            completed,
            f"{metrics.get('actor_loss', 0):.6f}",
            f"{metrics.get('critic_loss', 0):.6f}",
            f"{metrics.get('entropy', 0):.6f}",
            f"{metrics.get('approx_kl', 0):.6f}",
            f"{metrics.get('early_stopped', 0):.0f}",
        ] + [f"{breakdown.get(k, 0):.4f}" for k in REWARD_KEYS]
        evaluation = evaluation or {}
        row += [
            f"{evaluation[key]:.4f}" if key in evaluation else ""
            for key in (
                'eval_safe_completion_rate', 'eval_full_success_rate',
                'eval_mean_completed', 'eval_mean_crashes',
                'eval_mean_steps', 'eval_timeout_rate'
            )
        ]
        writer.writerow(row)


def _init_eval_csv(csv_path: Path, preserve_existing: bool = False) -> None:
    header = [
        "episode", "eval_safe_completion_rate", "eval_full_success_rate",
        "eval_mean_completed", "eval_mean_crashes",
        "eval_mean_steps", "eval_timeout_rate",
    ]
    if preserve_existing and csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows and rows[0] == header:
                return
            if rows:
                legacy_path = csv_path.with_name(csv_path.stem + "_legacy.csv")
                if not legacy_path.exists():
                    shutil.copy2(csv_path, legacy_path)
                old_header = rows[0]
                data_rows = [
                    [dict(zip(old_header, row)).get(key, "") for key in header]
                    for row in rows[1:]
                ]
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([header] + data_rows)
                return
        except (OSError, UnicodeError):
            pass
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def _read_last_episode(csv_path: Path) -> int | None:
    """Return the last valid episode number from a possibly partial CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    last_episode: int | None = None
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    last_episode = int(row["episode"])
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, UnicodeError):
        return None
    return last_episode


def _append_eval_csv(csv_path: Path, episode: int, evaluation: dict[str, float]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            f"{evaluation['eval_safe_completion_rate']:.4f}",
            f"{evaluation['eval_full_success_rate']:.4f}",
            f"{evaluation['eval_mean_completed']:.4f}",
            f"{evaluation['eval_mean_crashes']:.4f}",

            f"{evaluation['eval_mean_steps']:.4f}",
            f"{evaluation['eval_timeout_rate']:.4f}",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved MAPPO model for airport taxiway")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 继续训练，例：--resume checkpoints/model_ep45000.pt")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save-dir", type=str, default=None, help="覆盖模型和日志保存路径")
    parser.add_argument("--visualize", action="store_true", help="训练开始前显示滑行道网络图")
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override training.total_episodes without editing the YAML config",
    )
    parser.add_argument(
        "--actor-only",
        action="store_true",
        help="Resume only the shared Actor; initialize a fresh Critic and Adam states",
    )
    args = parser.parse_args()

    train_improved(
        args.config,
        resume=args.resume,
        seed=args.seed,
        save_dir_override=args.save_dir,
        visualize=args.visualize,
        load_critic=not args.actor_only,
        load_optimizer=not args.actor_only,
        episodes_override=args.episodes,
    )
