"""Plot the paper-facing Guided/NoGuide ablation curves.

The figure uses only metrics present in the current training/evaluation logs:
training reward, safe completion, full success, and timeout rate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"MAPPO-Guided": "#1F77B4", "MAPPO-NoGuide": "#D95F02"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def moving_average(values: list[float], window: int) -> list[float | None]:
    if window <= 1:
        return list(values)
    result: list[float | None] = []
    queue: list[float] = []
    total = 0.0
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        result.append(total / window if len(queue) == window else None)
    return result


def plot(args: argparse.Namespace) -> None:
    specs = [
        ("reward", "Training reward", "Reward"),
        ("eval_safe_completion_rate", "Safe service completion", "Rate (%)"),
        ("eval_full_success_rate", "Full mission success", "Rate (%)"),
        ("eval_timeout_rate", "Timeout episodes", "Rate (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex="col")
    axes = axes.flatten()

    for path_text, label in zip(args.evaluation_files, args.labels):
        path = Path(path_text)
        eval_rows = read_rows(path)
        color = COLORS.get(label, "#444444")

        x_eval = [number(row, "episode") for row in eval_rows]
        for axis, (key, title, ylabel) in zip(axes[1:], specs[1:]):
            points = [
                (x, number(row, key))
                for x, row in zip(x_eval, eval_rows)
                if x is not None and number(row, key) is not None
            ]
            if not points:
                continue
            px, py = zip(*points)
            axis.plot(px, py, marker="o", markersize=3.5, linewidth=1.7,
                      color=color, label=label)

        if args.training_files:
            training_index = list(args.evaluation_files).index(path_text)
            training_rows = read_rows(Path(args.training_files[training_index]))
            reward_points = [
                (number(row, "episode"), number(row, "total_reward"))
                for row in training_rows
            ]
            reward_points = [
                (x, y) for x, y in reward_points if x is not None and y is not None
            ]
            if reward_points:
                reward_points.sort()
                x, y = zip(*reward_points)
                smooth = moving_average(list(y), args.reward_window)
                points = [(xx, yy) for xx, yy in zip(x, smooth) if yy is not None]
                if points:
                    px, py = zip(*points)
                    axes[0].plot(px, py, linewidth=1.8, color=color, label=label)

    for axis, (_, title, ylabel) in zip(axes, specs):
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, linestyle="--")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False)

    axes[2].set_xlabel("Training episode")
    axes[3].set_xlabel("Training episode")
    axes[0].set_xlabel(f"Training episode ({args.reward_window}-episode moving mean)")
    axes[1].set_xlabel("Training episode")
    fig.suptitle("MAPPO-Guided and MAPPO-NoGuide ablation", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-files", nargs=2, required=True)
    parser.add_argument("--training-files", nargs=2, required=True)
    parser.add_argument("--labels", nargs=2, default=["MAPPO-Guided", "MAPPO-NoGuide"])
    parser.add_argument("--output", default="results/ablation_diagnostics/ablation_training_curves.png")
    parser.add_argument("--reward-window", type=int, default=100)
    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()
