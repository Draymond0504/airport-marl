"""Overwrite disturbance figures with actual operational-event windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_experiment_enhancements import plot_priority_preservation


METHODS = {"MAPPO": ("MAPPO-Guided", "#007C91"), "FCFS": ("FCFS+A*", "#C25B35")}
EVENT_TYPES = {
    "delay": {"delay_interruption"},
    "closure": {"closure"},
    "priority": {"priority_boost"},
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cumulative_points(rows: list[dict], total: int):
    return [
        (float(row["time_min"]), 100.0 * float(row["cumulative_success"]) / total)
        for row in rows
    ]


def plot_timeseries(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=False)
    for ax, scenario in zip(axes, ("delay", "closure", "priority")):
        meta = read_json(root / scenario / "experiment_meta.json")
        scale = float(meta["time_scale"])
        mappo_events = read_json(root / scenario / "MAPPO_events.json").get("events", [])
        for event in mappo_events:
            if event.get("type") not in EVENT_TYPES[scenario]:
                continue
            start = float(event["step"]) * scale / 60.0
            duration = float(event.get("duration", 0.0)) * scale / 60.0
            if duration > 0:
                ax.axvspan(start, start + duration, color="#7B8794", alpha=0.16, zorder=0)
            ax.axvline(start, color="#475569", linewidth=0.9, linestyle="--", zorder=1)
        for method, (label, color) in METHODS.items():
            summary = read_json(root / scenario / f"{method}_summary.json")
            points = cumulative_points(read_csv(root / scenario / f"{method}_timeseries.csv"), int(summary["total_count"]))
            if not points:
                continue
            x, y = zip(*points)
            ax.plot(x, y, color=color, linewidth=2.0, label=label)
            ax.scatter(x[-1], y[-1], color=color, s=24, zorder=3)
            ax.annotate(f"{y[-1]:.1f}%", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=color)
        ax.set_title(scenario.title())
        ax.set_ylabel("Cumulative window-safe service (%)")
        ax.set_ylim(0, 105)
        ax.legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("Simulation time (min)")
    fig.suptitle("Cumulative service response under operational disturbances", fontsize=14, y=0.995)
    fig.text(0.5, 0.01, "Shaded intervals denote actual delay interruption, closure, or priority-boost windows.", ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_dir / "fig_perturbation_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_intensity(root: Path, out_dir: Path) -> None:
    rows = []
    for scenario in ("delay", "closure", "priority"):
        meta = read_json(root / scenario / "experiment_meta.json")
        scale = float(meta["time_scale"])
        events = read_json(root / scenario / "MAPPO_events.json").get("events", [])
        for event in events:
            if event.get("type") not in EVENT_TYPES[scenario]:
                continue
            rows.append({
                "scenario": scenario,
                "event_type": event["type"],
                "step": int(event["step"]),
                "duration_min": round(float(event.get("duration", 0.0)) * scale / 60.0, 2),
                "affected_flights": event.get("count", "N/A"),
                "detail": event.get("edge_nodes", event.get("description", "priority boost")),
            })
    fields = ["scenario", "event_type", "step", "duration_min", "affected_flights", "detail"]
    with (out_dir / "perturbation_intensity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Scenario | Event | Step | Duration (min) | Affected flights | Detail |", "|---|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['event_type']} | {row['step']} | {row['duration_min']} | {row['affected_flights']} | {row['detail']} |")
    (out_dir / "perturbation_intensity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create operational disturbance figures.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_timeseries(root, out_dir)
    export_intensity(root, out_dir)
    plot_priority_preservation(root, out_dir)
    print(f"[OK] Operational disturbance figures written to {out_dir}")


if __name__ == "__main__":
    main()
