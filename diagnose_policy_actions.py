"""Run a deterministic micro-action audit for any MAPPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.config import Config
from env.marl_env import MARLEnv
from train_improved import run_action_diagnostic
from trainer.mappo_improved import MAPPOTrainerImproved


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose masked MAPPO actions")
    parser.add_argument("--config", required=True, help="Environment/config YAML")
    parser.add_argument("--model", required=True, help="Checkpoint to inspect")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--trace-episodes", type=int, default=3)
    parser.add_argument(
        "--output-dir", default="results/diagnostics/policy_action_audit"
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    env = MARLEnv(config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MAPPOTrainerImproved(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=config.training.hidden_dim,
        lr_actor=config.training.lr_actor,
        lr_critic=config.training.lr_critic,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_epsilon=config.training.clip_epsilon,
        entropy_coef=config.training.entropy_coef,
        value_loss_coef=config.training.value_loss_coef,
        max_grad_norm=config.training.max_grad_norm,
        update_epochs=config.training.update_epochs,
        num_mini_batches=config.training.num_mini_batches,
        critic_state_schema=env.critic_state_schema,
        device=device,
    )
    # The audit concerns the decentralized action policy only. This also makes
    # it safe to inspect a checkpoint with an older critic-state schema.
    trainer.load(args.model, load_critic=False, load_optimizer=False)
    report = run_action_diagnostic(
        trainer,
        config,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        trace_episodes=args.trace_episodes,
    )
    print(
        "Action audit complete: "
        f"avoidable_wait={report['avoidable_wait_pct']:.2f}% | "
        f"forced_wait={report['forced_wait_pct']:.2f}% | "
        f"edge_hold={report['edge_no_progress_pct']:.2f}%"
    )
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
