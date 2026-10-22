"""Create the paper table/figure for MAPPO-Guided vs MAPPO-NoGuide."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description="Plot A* guidance ablation")
    parser.add_argument("--guided", default="results/paper_final/baseline/MAPPO_summary.json")
    parser.add_argument("--no-guide", default="results/paper_final/ablation/no_guide/baseline/MAPPO_summary.json")
    parser.add_argument("--out-dir", default="results/paper_final/figures")
    args = parser.parse_args()

    guided = load(args.guided)
    no_guide = load(args.no_guide)
    methods = [("MAPPO-Guided", guided, "#1F77B4"), ("MAPPO-NoGuide", no_guide, "#D95F02")]
    metrics = [
        ("completion_rate_pct", "Safe completion (%)"),
        ("eventual_completion_rate_pct", "Eventual service (%)"),
        ("deadlock_count", "Deadlocks"),
        ("avg_total_time_min", "Avg total time (min)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    x = np.arange(len(methods))
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        values = [number(summary.get(key)) for _, summary, _ in methods]
        bars = ax.bar(x, values, color=[color for _, _, color in methods], alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(0.5, abs(value) * 0.02),
                    f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in methods], rotation=12)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.suptitle("Ablation of A* path guidance", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.90,
                        wspace=0.30, hspace=0.38)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_ablation_astar_guidance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "ablation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method", "safe_completion_pct", "eventual_service_pct", "deadlocks", "avg_total_time_min"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, summary, _ in methods:
            writer.writerow({
                "method": label,
                "safe_completion_pct": number(summary.get("completion_rate_pct")),
                "eventual_service_pct": number(summary.get("eventual_completion_rate_pct")),
                "deadlocks": number(summary.get("deadlock_count")),
                "avg_total_time_min": number(summary.get("avg_total_time_min")),
            })
    print(f"[OK] {out_dir / 'fig_ablation_astar_guidance.png'}")
    print(f"[OK] {out_dir / 'ablation_metrics.csv'}")


if __name__ == "__main__":
    main()
