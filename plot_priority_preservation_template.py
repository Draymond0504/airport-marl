"""Template for the priority-preservation figure.

Edit the DATA TO EDIT section, then run:
    python plot_priority_preservation_template.py --out results/fig_priority_preservation.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Data are loaded from an external JSON file. See priority_preservation_data_template.json.

BASELINE_COLORS = ["#74A9B5", "#DF929D"]
BOOSTED_COLORS = ["#07879A", "#C65A32"]


def add_label(ax, bar, value: float, color: str, signed: bool = False) -> None:
    offset = max(0.012, ax.get_ylim()[1] * 0.025)
    text = f"{value:+.2f}" if signed else f"{value:.2f}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
    )


def style_axis(ax) -> None:
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)


def load_data(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    required = {
        "methods",
        "priority_aircraft_total_service_time_min",
        "normal_aircraft_delay_relative_to_baseline_min",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing JSON fields: {sorted(missing)}")
    priority = data["priority_aircraft_total_service_time_min"]
    if not {"baseline", "priority_boosted"}.issubset(priority):
        raise ValueError("priority_aircraft_total_service_time_min requires baseline and priority_boosted")
    return data


def plot(out_path: Path, data: dict) -> None:
    methods = data["methods"]
    priority_baseline = data["priority_aircraft_total_service_time_min"]["baseline"]
    priority_boosted = data["priority_aircraft_total_service_time_min"]["priority_boosted"]
    normal_delay = data["normal_aircraft_delay_relative_to_baseline_min"]
    if len(priority_baseline) != len(methods):
        raise ValueError("Priority baseline data must contain one value per method.")
    if len(priority_boosted) != len(methods):
        raise ValueError("Priority boosted data must contain one value per method.")
    if len(normal_delay) != len(methods):
        raise ValueError("Normal-aircraft delay data must contain one value per method.")
    if any(v < 0 for v in priority_baseline + priority_boosted + normal_delay):
        raise ValueError("All service-time values must be non-negative.")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    left, right = axes
    x = np.arange(len(methods))
    width = 0.34

    # Left panel: priority-aircraft service time.
    baseline_bars = left.bar(
        x - width / 2,
        priority_baseline,
        width,
        color=BASELINE_COLORS,
        alpha=0.92,
        label="Baseline (No Disturbance)",
    )
    boosted_bars = left.bar(
        x + width / 2,
        priority_boosted,
        width,
        color=BOOSTED_COLORS,
        alpha=0.98,
        label="Priority Boosted",
    )
    for bar, value, color in zip(baseline_bars, priority_baseline, BASELINE_COLORS):
        add_label(left, bar, value, color)
    for bar, value, color in zip(boosted_bars, priority_boosted, BOOSTED_COLORS):
        add_label(left, bar, value, color)
    left.set_title("Priority-aircraft total service time", fontsize=13, pad=10)
    left.set_ylabel("Mean total service time (min)", fontsize=10)
    left.set_xticks(x, methods)
    left.tick_params(axis="x", labelsize=9)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(left)

    # Right panel: normal-aircraft change relative to baseline.
    right_bars = right.bar(
        x,
        normal_delay,
        width=0.55,
        color=BOOSTED_COLORS,
        alpha=0.98,
    )
    for bar, value, color in zip(right_bars, normal_delay, BOOSTED_COLORS):
        add_label(right, bar, value, color, signed=True)
    right.set_title("Normal-aircraft delay relative to baseline", fontsize=13, pad=10)
    right.set_ylabel("Change in mean total service time (min)", fontsize=10)
    right.set_xticks(x, methods)
    right.tick_params(axis="x", labelsize=9)
    style_axis(right)

    ymax_left = max(priority_baseline + priority_boosted) * 1.28
    ymax_right = max(normal_delay) * 1.28
    left.set_ylim(0, ymax_left)
    right.set_ylim(0, ymax_right)

    fig.suptitle("Priority preservation and normal-service impact", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.83, wspace=0.28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Figure written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot priority-preservation template")
    parser.add_argument("--data", required=True, help="Input JSON data file.")
    parser.add_argument(
        "--out",
        default="results/fig_priority_preservation_template.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()
    plot(Path(args.out), load_data(Path(args.data)))


if __name__ == "__main__":
    main()
