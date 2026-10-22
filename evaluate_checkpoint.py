"""Evaluate a MAPPO checkpoint on fixed selection and holdout task sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from configs.config import Config
from env.marl_env import MARLEnv
from train_improved import evaluate_policy
from trainer.mappo_improved import MAPPOTrainerImproved


def build_trainer(config: Config, env: MARLEnv) -> MAPPOTrainerImproved:
    training = config.training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MAPPOTrainerImproved(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=training.hidden_dim,
        lr_actor=training.lr_actor,
        lr_critic=training.lr_critic,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        clip_epsilon=training.clip_epsilon,
        entropy_coef=training.entropy_coef,
        value_loss_coef=training.value_loss_coef,
        max_grad_norm=training.max_grad_norm,
        update_epochs=training.update_epochs,
        num_mini_batches=training.num_mini_batches,
        individual_advantage_weight=training.individual_advantage_weight,
        target_kl=training.target_kl,
        critic_state_schema=env.critic_state_schema,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on fixed MAPPO task sets."
    )
    parser.add_argument("--config", default="configs/fine_tune_8agents.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection-episodes", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--holdout-episodes", type=int, default=None)
    parser.add_argument("--holdout-seed", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    env = MARLEnv(config)
    trainer = build_trainer(config, env)
    trainer.load(args.model, load_critic=True, load_optimizer=False)

    selection_episodes = args.selection_episodes or config.training.eval_episodes
    selection_seed = args.selection_seed or config.training.eval_seed
    holdout_episodes = args.holdout_episodes or config.training.holdout_eval_episodes
    holdout_seed = args.holdout_seed or config.training.holdout_eval_seed

    result = {
        "model": str(Path(args.model)),
        "config": args.config,
        "selection": evaluate_policy(
            trainer, config, episodes=selection_episodes, seed=selection_seed
        ),
        "selection_seed": selection_seed,
        "selection_episodes": selection_episodes,
        "holdout": evaluate_policy(
            trainer, config, episodes=holdout_episodes, seed=holdout_seed
        ),
        "holdout_seed": holdout_seed,
        "holdout_episodes": holdout_episodes,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        "Selection: "
        f"safe={result['selection']['eval_safe_completion_rate']:.1f}% | "
        f"full={result['selection']['eval_full_success_rate']:.1f}%"
    )
    print(
        "Holdout: "
        f"safe={result['holdout']['eval_safe_completion_rate']:.1f}% | "
        f"full={result['holdout']['eval_full_success_rate']:.1f}%"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
