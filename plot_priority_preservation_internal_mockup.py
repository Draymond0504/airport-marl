"""Create an explicitly labelled internal-only priority figure mock-up."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["MAPPO-Guided", "FCFS+A*"]
BASE_COLORS = ["#70A6B3", "#E3949D"]
BOOST_COLORS = ["#007C91", "#C25B35"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/internal_reference_mockup")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Illustrative values supplied for internal narrative/layout review only.
    baseline_priority = np.array([0.90, 1.68])
    boosted_priority = np.array([0.48, 0.60])
    normal_delta = np.array([0.02, 0.21])
    x = np.arange(2, dtype=float)
    width = 0.31

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    for i, method in enumerate(METHODS):
        ax.bar(i - width / 2, baseline_priority[i], width, color=BASE_COLORS[i], label="Baseline (No Disturbance)" if i == 0 else None)
        ax.bar(i + width / 2, boosted_priority[i], width, color=BOOST_COLORS[i], label="Priority Boosted" if i == 0 else None)
        ax.text(i - width / 2, baseline_priority[i] + 0.06, f"{baseline_priority[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, boosted_priority[i] + 0.06, f"{boosted_priority[i]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Priority-aircraft total service time")
    ax.set_ylabel("Mean total service time (min)")
    ax.set_xticks(x, METHODS)
    ax.set_ylim(0, 2.35)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    ax = axes[1]
    for i, method in enumerate(METHODS):
        ax.bar(i, normal_delta[i], width=0.40, color=BOOST_COLORS[i])
        ax.text(i, normal_delta[i] + 0.015, f"+{normal_delta[i]:.2f}", ha="center", va="bottom", fontsize=8, color=BOOST_COLORS[i], fontweight="bold")
    ax.axhline(0, color="#5B6570", linewidth=0.9)
    ax.set_title("Normal-aircraft delay relative to baseline")
    ax.set_ylabel("Change in mean total service time (min)")
    ax.set_xticks(x, METHODS)
    ax.set_ylim(0, 0.27)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    fig.suptitle("Priority preservation and normal-service impact", fontsize=14, color="#263238", y=1.02)
    fig.text(0.5, -0.035, "Illustrative internal mock-up only; values are not empirical results and must not be used in the paper.", ha="center", fontsize=8, color="#9F1239", style="italic")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_preservation_internal_mockup.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote internal mock-up to {out_dir}")


if __name__ == "__main__":
    main()
