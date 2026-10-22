"""Additional paper metrics for the planning-guided MARL experiments.

The script intentionally separates aggregate priority preservation from the
event-local trajectory case. The former is the primary effectiveness figure;
the latter remains a mechanism illustration.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.taxi_graph_env import TaxiGraphEnv
from planner.astar import AStarPlanner


METHODS = {
    "MAPPO": {"label": "MAPPO-Guided", "color": "#007C91"},
    "FCFS": {"label": "FCFS+A*", "color": "#C25B35"},
}
INK = "#263238"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def topology_tools() -> tuple[AStarPlanner, dict[tuple[int, int], float]]:
    env = TaxiGraphEnv(
        num_agents=1,
        max_steps=1,
        close_prob=0.0,
        delay_prob=0.0,
        priority_prob=0.0,
        layout="F",
        auto_speed=True,
        random_flow_mix=False,
    )
    planner = AStarPlanner({node_id: node.position for node_id, node in env.nodes.items()})
    lengths = {(edge.start, edge.end): edge.length for edge in env.edges}
    return planner, lengths


def compact_path(value: str) -> list[int]:
    try:
        raw = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    result = []
    for node in raw:
        node = int(node)
        if not result or node != result[-1]:
            result.append(node)
    return result


def path_deviation_rows(directory: Path) -> list[dict]:
    planner, lengths = topology_tools()
    graph = TaxiGraphEnv(layout="F", num_agents=1, max_steps=1).get_graph_for_planner()
    rows = read_csv(directory / "MAPPO_per_agent.csv")
    output = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        path = compact_path(row.get("path_nodes", "[]"))
        if len(path) < 2:
            continue
        reference = planner.plan_path(path[0], path[-1], graph) or []
        if len(reference) < 2:
            continue
        actual_length = sum(lengths.get((a, b), 0.0) for a, b in zip(path, path[1:]))
        reference_length = sum(lengths.get((a, b), 0.0) for a, b in zip(reference, reference[1:]))
        if reference_length <= 0:
            continue
        output.append({
            "flight_index": int(row["flight_index"]),
            "actual_path_distance": actual_length,
            "reference_path_distance": reference_length,
            "path_deviation_ratio": (actual_length - reference_length) / reference_length,
        })
    return output


def plot_path_deviation(guided_dir: Path, noguide_dir: Path, out_dir: Path) -> None:
    samples = {
        "MAPPO-Guided": path_deviation_rows(guided_dir),
        "MAPPO-NoGuide": path_deviation_rows(noguide_dir),
    }
    data = [[row["path_deviation_ratio"] for row in samples[label]] for label in samples]
    if not all(data):
        raise RuntimeError("Path-deviation plot requires completed MAPPO rows for both ablation arms.")
    labels = list(samples)
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(box["boxes"], ("#007C91", "#8E5EA2")):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    medians = [float(np.median(values)) for values in data]
    for index, (values, median) in enumerate(zip(data, medians), start=1):
        ax.scatter(index, median, color="#202124", marker="D", s=30, zorder=4)
        ax.text(index, median, f"median={median:.3f}\nn={len(values)}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="#6B7280", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Relative path deviation ratio, PDI")
    ax.set_title("A* guidance ablation: global path consistency", color=INK, pad=12)
    ax.text(
        0.5,
        -0.20,
        "PDI = (actual traversed distance - A* reference distance) / A* reference distance.\n"
        "Lower values indicate consistency with the global reference, not mandatory path following.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#5F6B72",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_path_deviation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fields = ["method", "n", "pdi_median", "pdi_iqr", "pdi_mean"]
    with (out_dir / "ablation_path_deviation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, values in zip(labels, data):
            writer.writerow({
                "method": label,
                "n": len(values),
                "pdi_median": round(float(np.median(values)), 5),
                "pdi_iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 5),
                "pdi_mean": round(float(np.mean(values)), 5),
            })


def completed_total_times(rows: list[dict], high_priority: bool | None) -> list[float]:
    values = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        is_high = number(row.get("priority"), 1.0) > 1.0
        if high_priority is not None and is_high != high_priority:
            continue
        values.append(number(row.get("taxi_sec")) / 60.0 + number(row.get("wait_time_min")))
    return values


def plot_priority_preservation(root: Path, out_dir: Path) -> None:
    baseline = root / "baseline"
    priority = root / "priority"
    high, normal_delta, labels = [], [], []
    for method in METHODS:
        baseline_rows = read_csv(baseline / f"{method}_per_agent.csv")
        priority_rows = read_csv(priority / f"{method}_per_agent.csv")
        high_values = completed_total_times(priority_rows, True)
        priority_normal = completed_total_times(priority_rows, False)
        baseline_normal = completed_total_times(baseline_rows, False)
        high.append(float(np.mean(high_values)) if high_values else np.nan)
        normal_delta.append(
            (float(np.mean(priority_normal)) - float(np.mean(baseline_normal)))
            if priority_normal and baseline_normal else np.nan
        )
        labels.append(METHODS[method]["label"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = [METHODS[method]["color"] for method in METHODS]
    bars = axes[0].bar(np.arange(2), high, color=colors, width=0.58)
    for bar, value in zip(bars, high):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Priority-aircraft total service time")
    axes[0].set_ylabel("Mean total service time (min)")
    axes[0].set_xticks(np.arange(2), labels)

    bars = axes[1].bar(np.arange(2), normal_delta, color=colors, width=0.58)
    for bar, value in zip(bars, normal_delta):
        va = "bottom" if value >= 0 else "top"
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.2f}", ha="center", va=va, fontsize=8)
    axes[1].axhline(0.0, color="#6B7280", linewidth=0.9)
    axes[1].set_title("Normal-aircraft delay relative to baseline")
    axes[1].set_ylabel("Change in mean total service time (min)")
    axes[1].set_xticks(np.arange(2), labels)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Priority preservation and normal-service impact", fontsize=14, color=INK, y=1.02)
    fig.text(0.5, -0.03, "Normal-aircraft values are measured against the same method's baseline scenario.", ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_preservation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rolling_peak(events: list[dict], window_steps: int = 100) -> int:
    admissions = sorted(int(event["step"]) for event in events if event.get("type") == "admit")
    peak = left = 0
    for right, step in enumerate(admissions):
        while admissions[left] < step - window_steps:
            left += 1
        peak = max(peak, right - left + 1)
    return peak


def export_disturbance_intensity(root: Path, out_dir: Path) -> None:
    baseline_events = read_json(root / "baseline" / "MAPPO_events.json").get("events", [])
    baseline_peak = rolling_peak(baseline_events)
    rows = []
    for scenario in ("delay", "closure", "priority"):
        events = read_json(root / scenario / "MAPPO_events.json").get("events", [])
        meta = read_json(root / scenario / "experiment_meta.json")
        time_scale = number(meta.get("time_scale"), 1.0)
        if scenario == "delay":
            event = next(event for event in events if event.get("type") == "delay_release_plan")
            peak = rolling_peak(events)
            rows.append({
                "scenario": "Delay release",
                "event_count": event.get("count"),
                "duration_min": round(number(event.get("duration")) * time_scale / 60.0, 2),
                "peak_admissions_per_100_steps": peak,
                "peak_to_baseline_ratio": round(peak / max(baseline_peak, 1), 2),
                "detail": event.get("release_window"),
            })
        elif scenario == "closure":
            for index, event in enumerate(event for event in events if event.get("type") == "closure"):
                rows.append({
                    "scenario": f"Closure {index + 1}",
                    "event_count": 1,
                    "duration_min": round(number(event.get("duration")) * time_scale / 60.0, 2),
                    "peak_admissions_per_100_steps": "N/A",
                    "peak_to_baseline_ratio": "N/A",
                    "detail": f"edge {tuple(event.get('edge_nodes', []))}, step {event.get('step')}",
                })
        else:
            boosts = [event for event in events if event.get("type") == "priority_boost"]
            rows.append({
                "scenario": "Priority boosts",
                "event_count": len(boosts),
                "duration_min": round(number(boosts[0].get("duration")) * time_scale / 60.0, 2) if boosts else 0.0,
                "peak_admissions_per_100_steps": "N/A",
                "peak_to_baseline_ratio": "N/A",
                "detail": "scheduled priority targets",
            })
    fields = ["scenario", "event_count", "duration_min", "peak_admissions_per_100_steps", "peak_to_baseline_ratio", "detail"]
    with (out_dir / "perturbation_intensity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Scenario | Events | Duration (min) | Peak release ratio | Detail |", "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['event_count']} | {row['duration_min']} | {row['peak_to_baseline_ratio']} | {row['detail']} |")
    (out_dir / "perturbation_intensity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enhanced paper metrics and figures.")
    parser.add_argument("--result-root", required=True, help="Rolling paper experiment root.")
    parser.add_argument("--guided-dir", default=None, help="Guided ablation rolling baseline directory.")
    parser.add_argument("--noguide-dir", default=None, help="NoGuide ablation rolling baseline directory.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_priority_preservation(root, out_dir)
    export_disturbance_intensity(root, out_dir)
    if args.guided_dir and args.noguide_dir:
        plot_path_deviation(Path(args.guided_dir), Path(args.noguide_dir), out_dir)
    print(f"[OK] Enhanced paper metrics written to {out_dir}")


if __name__ == "__main__":
    main()
