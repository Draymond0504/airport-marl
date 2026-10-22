"""Priority-preservation figure using in-network mission time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_experiment_enhancements import read_csv


METHODS = {
    "MAPPO": {"label": "MAPPO-Guided", "base": "#70A6B3", "boost": "#007C91"},
    "FCFS": {"label": "FCFS+A*", "base": "#E3949D", "boost": "#C25B35"},
}


def value(row: dict) -> float:
    return float(row.get("taxi_sec") or 0.0) / 60.0


def completed(path: Path, high: bool | None) -> list[float]:
    result = []
    for row in read_csv(path):
        if row.get("status") != "completed":
            continue
        is_high = float(row.get("priority") or 1.0) > 1.0
        if high is None or is_high == high:
            result.append(value(row))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root, out_dir = Path(args.result_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for method, spec in METHODS.items():
        baseline = completed(root / "baseline" / f"{method}_per_agent.csv", False)
        boosted = completed(root / "priority" / f"{method}_per_agent.csv", True)
        priority_normal = completed(root / "priority" / f"{method}_per_agent.csv", False)
        if not baseline or not boosted or not priority_normal:
            raise RuntimeError(f"Missing completed priority samples for {spec['label']}.")
        records[method] = {
            "method": spec["label"],
            "baseline_mean": float(np.mean(baseline)),
            "boosted_mean": float(np.mean(boosted)),
            "normal_delta": float(np.mean(priority_normal) - np.mean(baseline)),
            "baseline_n": len(baseline),
            "boosted_n": len(boosted),
            "priority_normal_n": len(priority_normal),
        }

    x = np.arange(len(METHODS), dtype=float)
    width = 0.31
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    for index, (method, spec) in enumerate(METHODS.items()):
        row = records[method]
        ax.bar(index - width / 2, row["baseline_mean"], width, color=spec["base"], label="Baseline (No Disturbance)" if index == 0 else None)
        ax.bar(index + width / 2, row["boosted_mean"], width, color=spec["boost"], label="Priority Boosted" if index == 0 else None)
        ax.text(index - width / 2, row["baseline_mean"] + max(row["baseline_mean"] * 0.025, 0.03), f"{row['baseline_mean']:.2f}\nn={row['baseline_n']}", ha="center", va="bottom", fontsize=8)
        ax.text(index + width / 2, row["boosted_mean"] + max(row["boosted_mean"] * 0.025, 0.03), f"{row['boosted_mean']:.2f}\nn={row['boosted_n']}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Priority-aircraft in-network mission time")
    ax.set_ylabel("Mean in-network mission time (min)")
    ax.set_xticks(x, [spec["label"] for spec in METHODS.values()])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    ax = axes[1]
    for index, (method, spec) in enumerate(METHODS.items()):
        delta = records[method]["normal_delta"]
        ax.bar(index, delta, width=0.40, color=spec["boost"])
        offset = max(abs(delta) * 0.06, 0.02)
        ax.text(index, delta + (offset if delta >= 0 else -offset), f"{delta:+.2f}", ha="center", va="bottom" if delta >= 0 else "top", fontsize=8, color=spec["boost"], fontweight="bold")
    ax.axhline(0.0, color="#5B6570", linewidth=0.9)
    ax.set_title("Normal-aircraft time change relative to baseline")
    ax.set_ylabel("Change in mean in-network mission time (min)")
    ax.set_xticks(x, [spec["label"] for spec in METHODS.values()])
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    fig.suptitle("Priority preservation and normal-service impact", fontsize=14, color="#263238", y=1.02)
    fig.text(0.5, -0.03, "Normal-aircraft values are measured against each method's respective baseline. Entry-queue delay is excluded and reported separately.", ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_preservation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_priority_preservation_network_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = list(records.values())
    with (out_dir / "priority_preservation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote network-time priority figure to {out_dir}")


if __name__ == "__main__":
    main()
