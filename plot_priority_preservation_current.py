"""Plot priority preservation with the established paper colour/style."""

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


def number(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def total_service_minutes(row: dict) -> float:
    # Keep the definition used by the existing priority-preservation figure:
    # in-network taxi time plus entry-queue waiting time.
    return number(row.get("taxi_sec")) / 60.0 + number(row.get("wait_time_min"))


def completed_values(path: Path, high_priority: bool | None) -> list[float]:
    values = []
    for row in read_csv(path):
        if row.get("status") != "completed":
            continue
        high = number(row.get("priority"), 1.0) > 1.0
        if high_priority is not None and high != high_priority:
            continue
        values.append(total_service_minutes(row))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = root / "baseline"
    priority_dir = root / "priority"
    records = {}
    for method, spec in METHODS.items():
        baseline_normal = completed_values(baseline_dir / f"{method}_per_agent.csv", False)
        priority_high = completed_values(priority_dir / f"{method}_per_agent.csv", True)
        priority_normal = completed_values(priority_dir / f"{method}_per_agent.csv", False)
        if not baseline_normal or not priority_high or not priority_normal:
            raise RuntimeError(f"Incomplete priority samples for {spec['label']}.")
        base_mean = float(np.mean(baseline_normal))
        boosted_mean = float(np.mean(priority_high))
        normal_delta = float(np.mean(priority_normal)) - base_mean
        records[method] = {
            "label": spec["label"],
            "baseline_mean": base_mean,
            "boosted_mean": boosted_mean,
            "normal_delta": normal_delta,
            "baseline_n": len(baseline_normal),
            "boosted_n": len(priority_high),
            "priority_normal_n": len(priority_normal),
        }

    x = np.arange(len(METHODS), dtype=float)
    width = 0.31
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    for index, (method, spec) in enumerate(METHODS.items()):
        row = records[method]
        bars = ax.bar(index - width / 2, row["baseline_mean"], width, color=spec["base"], label="Baseline (No Disturbance)" if index == 0 else None)
        bars2 = ax.bar(index + width / 2, row["boosted_mean"], width, color=spec["boost"], label="Priority Boosted" if index == 0 else None)
        ax.text(index - width / 2, row["baseline_mean"] + max(row["baseline_mean"] * 0.025, 0.03),
                f"{row['baseline_mean']:.2f}\nn={row['baseline_n']}", ha="center", va="bottom", fontsize=8)
        ax.text(index + width / 2, row["boosted_mean"] + max(row["boosted_mean"] * 0.025, 0.03),
                f"{row['boosted_mean']:.2f}\nn={row['boosted_n']}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Priority-aircraft total service time")
    ax.set_ylabel("Mean total service time (min)")
    ax.set_xticks(x, [spec["label"] for spec in METHODS.values()])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    ax = axes[1]
    for index, (method, spec) in enumerate(METHODS.items()):
        row = records[method]
        color = spec["boost"]
        bar = ax.bar(index, row["normal_delta"], width=0.40, color=color)
        value = row["normal_delta"]
        va = "bottom" if value >= 0 else "top"
        offset = max(abs(value) * 0.06, 0.02)
        ax.text(index, value + (offset if value >= 0 else -offset), f"{value:+.2f}", ha="center", va=va, fontsize=8, color=color, fontweight="bold")
    ax.axhline(0.0, color="#5B6570", linewidth=0.9)
    ax.set_title("Normal-aircraft delay relative to baseline")
    ax.set_ylabel("Change in mean total service time (min)")
    ax.set_xticks(x, [spec["label"] for spec in METHODS.values()])
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

    fig.suptitle("Priority preservation and normal-service impact", fontsize=14, color="#263238", y=1.02)
    fig.text(0.5, -0.03,
             "Normal-aircraft values are measured against each method's respective baseline scenario. Values include entry-queue waiting, consistent with the original priority analysis.",
             ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_preservation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_priority_preservation_current.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for method, record in records.items():
        rows.append({"method": record["label"], **{key: value for key, value in record.items() if key != "label"}})
    with (out_dir / "priority_preservation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote priority-preservation figure to {out_dir}")


if __name__ == "__main__":
    main()
