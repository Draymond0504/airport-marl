"""Run guided vs no-guide MAPPO ablation and plot convergence curves."""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from train_improved import train_improved


TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
NUM_AGENTS = 8
SMOOTH_WINDOW = 100
FCFS_BASELINE = 0.65
OUTPUT_DIR = "paper_figures"


EXPERIMENTS = [
    {
        "name": "guided",
        "label": "Planning-guided MAPPO",
        "color": "#e63946",
        "config": "configs/improved_v2.yaml",
        "stable_config": "configs/improved_stable.yaml",
        "save_dir": f"checkpoints/ablation_quick_{TIMESTAMP}/guided",
    },
    {
        "name": "no_guide",
        "label": "Pure MARL (No Guide)",
        "color": "#457b9d",
        "config": "configs/ablation_no_guide.yaml",
        "stable_config": "configs/ablation_no_guide_stable.yaml",
        "save_dir": f"checkpoints/ablation_quick_{TIMESTAMP}/no_guide",
    },
]


def run_training(episodes_override: int | None = None, stable: bool = False) -> None:
    for exp in EXPERIMENTS:
        config_path = exp["stable_config"] if stable else exp["config"]
        print("\n" + "=" * 70)
        print(f"Training [{exp['name']}] with {config_path} -> {exp['save_dir']}")
        print("=" * 70)

        if episodes_override is not None:
            cfg = Config.from_yaml(config_path)
            cfg.training.total_episodes = episodes_override
            train_improved(
                config_path=None,
                override_config=cfg,
                seed=42,
                save_dir_override=exp["save_dir"],
            )
        else:
            train_improved(
                config_path=config_path,
                seed=42,
                save_dir_override=exp["save_dir"],
            )


def load_log(save_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(save_dir, "training_log_improved.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def plot_comparison(stable: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title_suffix = "Stable" if stable else "Perturbed"
    fig.suptitle(
        f"Ablation Study ({title_suffix}): Planning-guided MAPPO vs Pure MARL",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    for exp in EXPERIMENTS:
        try:
            df = load_log(exp["save_dir"])
        except FileNotFoundError as exc:
            print(f"[WARN] Missing log: {exc}")
            continue

        ep = df["episode"].values
        reward_raw = df["total_reward"].values
        reward_smooth = pd.Series(reward_raw).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
        completion_raw = df["completed"].values / NUM_AGENTS
        completion_smooth = pd.Series(completion_raw).rolling(SMOOTH_WINDOW, min_periods=1).mean().values

        axes[0].plot(ep, reward_smooth, label=exp["label"], color=exp["color"], linewidth=2)
        axes[0].fill_between(
            ep,
            reward_smooth - reward_raw.std() * 0.3,
            reward_smooth + reward_raw.std() * 0.3,
            color=exp["color"],
            alpha=0.12,
        )

        axes[1].plot(ep, completion_smooth, label=exp["label"], color=exp["color"], linewidth=2)
        axes[1].fill_between(
            ep,
            completion_smooth - completion_raw.std() * 0.3,
            completion_smooth + completion_raw.std() * 0.3,
            color=exp["color"],
            alpha=0.12,
        )

    axes[0].set_title("(a) Average Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend(loc="lower right")

    axes[1].axhline(
        y=FCFS_BASELINE,
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label=f"FCFS+A* Baseline ({FCFS_BASELINE:.0%})",
    )
    axes[1].set_title("(b) Task Completion Rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Completion Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    mode = "stable" if stable else "perturbed"
    out_path = os.path.join(OUTPUT_DIR, f"ablation_convergence_{mode}_{TIMESTAMP}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run guided/no-guide MAPPO ablation.")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes.")
    parser.add_argument("--plot-only", action="store_true", help="Skip training and only plot logs.")
    parser.add_argument("--stable", action="store_true", help="Use stable no-perturbation configs.")
    args = parser.parse_args()

    if not args.plot_only:
        run_training(episodes_override=args.episodes, stable=args.stable)

    plot_comparison(stable=args.stable)
