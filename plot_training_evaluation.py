"""Plot deterministic evaluation logs for the paper training figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#1F77B4", "#D95F02", "#2CA02C", "#9467BD"]


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def value(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def rolling_mean(values, window: int):
    if window <= 1:
        return values
    result = []
    total = 0.0
    queue = []
    for item in values:
        queue.append(item)
        total += item
        if len(queue) > window:
            total -= queue.pop(0)
        # Do not display a partial-window mean. It creates a misleading
        # vertical startup artifact when a training run begins with a few
        # atypical episodes.
        result.append(total / window if len(queue) == window else None)
    return result


def plot_stage_rewards(files, labels, output: Path, window: int) -> None:
    """Plot reward by local episode without inventing cross-stage continuity."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for index, (file_name, label) in enumerate(zip(files, labels)):
        rows = read_rows(Path(file_name))
        points = [
            (value(row, "episode"), value(row, "total_reward"))
            for row in rows
        ]
        points = sorted(
            ((x, y) for x, y in points if x is not None and y is not None),
            key=lambda item: item[0],
        )
        if not points:
            continue
        x, y = zip(*points)
        smooth = rolling_mean(list(y), window)
        display = [(xx, yy) for xx, yy in zip(x, smooth) if yy is not None]
        if not display:
            continue
        x_plot, y_plot = zip(*display)
        ax.plot(
            x_plot,
            y_plot,
            color=COLORS[index % len(COLORS)],
            linewidth=1.6,
            label=label,
        )
    ax.set_xlabel("Stage-local training episode")
    ax.set_ylabel(f"Episode reward ({window}-episode moving mean)")
    ax.set_title("Within-stage training reward")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot deterministic training evaluation curves")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Evaluation CSV paths")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="One label per evaluation CSV")
    parser.add_argument("--out", default="results/paper_final/figures/fig_training_evaluation.png")
    parser.add_argument(
        "--stitch", action="store_true",
        help="Place sequential fine-tuning stages on one cumulative x-axis."
    )
    parser.add_argument(
        "--training-files", nargs="*",
        help="Optional training logs used only for a separate local-reward plot."
    )
    parser.add_argument(
        "--reward-out", default="results/paper_final/figures/fig_training_reward_by_stage.png"
    )
    parser.add_argument("--reward-window", type=int, default=100)
    args = parser.parse_args()
    if len(args.files) != len(args.labels):
        raise SystemExit("--files and --labels must have the same length")
    if args.training_files and len(args.training_files) != len(args.labels):
        raise SystemExit("--training-files must have one path per --labels entry")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    specs = [
        ("eval_safe_completion_rate", "Safe completion (%)"),
        ("eval_full_success_rate", "Full success (%)"),
        ("eval_mean_deadlocks", "Mean deadlocks"),
        ("eval_timeout_rate", "Timeout rate (%)"),
    ]
    cursor = 0.0
    phase_starts: list[tuple[float, str]] = []
    for file_index, (file_name, label) in enumerate(zip(args.files, args.labels)):
        rows = read_rows(Path(file_name))
        raw_x = [value(row, "episode") for row in rows]
        valid_x = [item for item in raw_x if item is not None]
        if args.stitch and valid_x:
            first_episode = min(valid_x)
            x = [None if item is None else cursor + item - first_episode for item in raw_x]
            phase_starts.append((cursor, label))
            cursor = max(item for item in x if item is not None)
        else:
            x = raw_x
        color = COLORS[file_index % len(COLORS)]
        for ax, (key, ylabel) in zip(axes.flat, specs):
            y = [value(row, key) for row in rows]
            points = [(xx, yy) for xx, yy in zip(x, y) if xx is not None and yy is not None]
            if points:
                px, py = zip(*points)
                ax.plot(px, py, marker="o", markersize=3, linewidth=1.8,
                        color=color, label=label)

    for ax, (_, ylabel) in zip(axes.flat, specs):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(frameon=False)
        if args.stitch:
            for boundary, _ in phase_starts[1:]:
                ax.axvline(boundary, color="#777777", linestyle=":", linewidth=1.0, zorder=0)
    axes[1, 0].set_xlabel("Training episode")
    axes[1, 1].set_xlabel("Training episode")
    title = "Deterministic evaluation across fine-tuning stages" if args.stitch else "Deterministic evaluation during training"
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output}")

    if args.training_files:
        plot_stage_rewards(
            args.training_files,
            args.labels,
            Path(args.reward_out),
            max(args.reward_window, 1),
        )


if __name__ == "__main__":
    main()
