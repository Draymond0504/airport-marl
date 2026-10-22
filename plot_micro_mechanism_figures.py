"""Generate the three event-faithful micro-mechanism figures.

The figures are built from the latest layered rolling outputs:

* delay: actual cumulative completions and on-route-wait survival;
* closure: observed edge transitions reconstructed from raw actions;
* priority: event-local motion traces and actual completed-aircraft taxi time.

No illustrative or manually fitted values are used.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plot_final_paper_figures import (
    METHODS,
    INK,
    OTHER,
    VIP,
    draw_network,
    layout_f_geometry,
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_event(events: list[dict], event_type: str) -> dict:
    matches = [event for event in events if event.get("type") == event_type]
    if not matches:
        raise RuntimeError(f"No {event_type!r} event found.")
    return min(matches, key=lambda event: int(event["step"]))


def plot_delay(root: Path, out_dir: Path) -> None:
    """Show the actual release interruption and the on-route waiting tail."""
    event = find_event(read_json(root / "delay" / "MAPPO_events.json")["events"], "delay_interruption")
    scale = float(read_json(root / "delay" / "experiment_meta.json")["time_scale"])
    start_min = int(event["step"]) * scale / 60.0
    end_min = start_min + int(event.get("duration", 0)) * scale / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    # Use an event-local time window so that the interruption and recovery are visible.
    left = axes[0]
    window_end = end_min + 240.0
    for method, spec in METHODS.items():
        rows = read_csv(root / "delay" / f"{method}_timeseries.csv")
        summary = read_json(root / "delay" / f"{method}_summary.json")
        total = int(summary["total_count"])
        points = [
            (float(row["time_min"]), 100.0 * float(row["cumulative_success"]) / total)
            for row in rows
            if float(row["time_min"]) <= window_end
        ]
        if not points:
            continue
        x, y = zip(*points)
        left.plot(x, y, color=spec["color"], linewidth=2.0, label=spec["label"])
        left.scatter(x[-1], y[-1], color=spec["color"], s=22, zorder=3)
        left.annotate(f"{y[-1]:.1f}%", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=spec["color"])

    left.axvspan(start_min, end_min, color="#94A3B8", alpha=0.18, zorder=0)
    left.axvline(start_min, color="#475569", linestyle="--", linewidth=0.9)
    left.axvline(end_min, color="#475569", linestyle=":", linewidth=0.9)
    left.text(start_min, 101.5, "interruption", rotation=90, ha="right", va="top", fontsize=7, color="#475569")
    left.text(end_min, 101.5, "release resumes", rotation=90, ha="left", va="top", fontsize=7, color="#475569")
    left.set_xlim(0, window_end)
    left.set_ylim(0, 105)
    left.set_xlabel("Simulation time (min)")
    left.set_ylabel("Cumulative completed tasks (%)")
    left.set_title("Mid-service interruption and recovery")
    left.legend(frameon=False, loc="lower right")
    left.grid(axis="y", alpha=0.18)
    left.spines[["top", "right"]].set_visible(False)

    # Survival curve for actual on-route waiting. Queue waiting is deliberately excluded.
    right = axes[1]
    wait_values: dict[str, list[float]] = {}
    for method in METHODS:
        rows = read_csv(root / "delay" / f"{method}_per_agent.csv")
        wait_values[method] = [float(row.get("onroute_wait_sec") or 0.0) / 60.0 for row in rows]
    xmax = max((max(values, default=0.0) for values in wait_values.values()), default=1.0)
    xmax = max(10.0, float(np.ceil(xmax / 10.0) * 10.0))
    grid = np.linspace(0.0, xmax, 250)
    for method, spec in METHODS.items():
        values = np.asarray(wait_values[method], dtype=float)
        survival = [100.0 * float(np.mean(values >= threshold)) for threshold in grid]
        right.plot(grid, survival, color=spec["color"], linewidth=2.0, label=spec["label"])
        right.axvline(float(np.median(values)), color=spec["color"], linestyle=":", linewidth=0.8, alpha=0.7)
    right.set_xlim(0, xmax)
    right.set_ylim(0, 105)
    right.set_xlabel("On-route waiting threshold (min)")
    right.set_ylabel("Flights exceeding threshold (%)")
    right.set_title("Tail probability of on-route waiting")
    right.legend(frameon=False, loc="upper right")
    right.grid(axis="y", alpha=0.18)
    right.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Delay mechanism: interruption, recovery, and local waiting", fontsize=14, color=INK, y=1.01)
    fig.text(0.5, 0.01, f"Actual interruption: t={start_min:.1f}–{end_min:.1f} min; waiting curve uses on-route wait only (n=363 per method).", ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_dir / "fig_delay_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def observed_edge_transitions(path: Path) -> list[tuple[int, int, int]]:
    """Reconstruct completed edge transitions from position changes in raw actions."""
    transitions = []
    for row in read_csv(path):
        before = int(row["position_before"])
        after = int(row["position_after"])
        if int(row["on_edge_before"]) == 1 and before != after:
            transitions.append((int(row["step"]), before, after))
    return transitions


def edge_share_counts(transitions: list[tuple[int, int, int]], start: int, end: int) -> tuple[dict[tuple[int, int], float], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for step, start_node, end_node in transitions:
        if start <= step < end:
            counts[tuple(sorted((start_node, end_node)))] += 1
    total = sum(counts.values())
    if total == 0:
        return {}, 0
    return {edge: value / total for edge, value in counts.items()}, total


def plot_closure(root: Path, out_dir: Path) -> None:
    """Compare route shares before and during the first actual closure."""
    event = find_event(read_json(root / "closure" / "MAPPO_events.json")["events"], "closure")
    trigger = int(event["step"])
    duration = int(event["duration"])
    closed = tuple(sorted(int(value) for value in event["edge_nodes"]))
    windows = [("Before closure", trigger - duration, trigger), ("Closure active", trigger, trigger + duration)]
    pos, edges = layout_f_geometry()
    data = {method: observed_edge_transitions(root / "closure" / f"{method}_raw_actions.csv") for method in METHODS}
    shares: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    totals: dict[tuple[str, str], int] = {}
    for method in METHODS:
        for label, start, end in windows:
            shares[(method, label)], totals[(method, label)] = edge_share_counts(data[method], start, end)
    vmax = max((value for values in shares.values() for value in values.values()), default=1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    fig = plt.figure(figsize=(12.0, 8.3))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06], hspace=0.20, wspace=0.08)
    axes = np.array([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])], [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])
    colorbar_ax = fig.add_subplot(grid[:, 2])
    for row, method in enumerate(METHODS):
        for col, (label, start, end) in enumerate(windows):
            ax = axes[row, col]
            draw_network(ax, pos, edges)
            local = shares[(method, label)]
            for edge, share in local.items():
                if edge[0] not in pos or edge[1] not in pos:
                    continue
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                ax.plot([x0, x1], [y0, y1], color=cmap(norm(share)), linewidth=1.2 + 5.0 * share / max(vmax, 1e-9), solid_capstyle="round", zorder=3)
            if label == "Closure active":
                x0, y0 = pos[closed[0]]
                x1, y1 = pos[closed[1]]
                ax.plot([x0, x1], [y0, y1], color="#7F1D1D", linewidth=5.0, linestyle=(0, (2, 1)), zorder=5)
            ax.set_title(f"{METHODS[method]['label']} | {label}\n{totals[(method, label)]} observed transitions", fontsize=10, color=INK)
            ax.text(0.02, 0.03, f"steps {start}–{end - 1}", transform=ax.transAxes, fontsize=8, color="#4B5563", ha="left", va="bottom")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Share of observed edge transitions")
    fig.suptitle(f"Relative edge-use response to closure of link {closed[0]}–{closed[1]}", fontsize=14, color=INK, y=0.985)
    fig.text(0.5, 0.012, "Colour and width encode within-window route share; the dashed dark-red segment is unavailable only during the closure window.", ha="center", fontsize=8, color="#5F6B72")
    fig.subplots_adjust(left=0.025, right=0.95, bottom=0.08, top=0.90)
    fig.savefig(out_dir / "fig_closure_edge_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def local_distance_tracks(payload: dict) -> tuple[dict[int, list[tuple[float, float]]], int]:
    event_step = int(payload["event_step"])
    scale = float(payload.get("time_scale", 1.0))
    first = event_step - int(payload.get("window_before_steps", 100))
    last = event_step + int(payload.get("window_after_steps", 200))
    tracks: dict[int, list[tuple[float, float]]] = defaultdict(list)
    distance: dict[int, float] = defaultdict(float)
    previous_step: dict[int, int] = {}
    previous_speed: dict[int, float] = {}
    for frame in sorted(payload.get("frames", []), key=lambda item: int(item["step"])):
        step = int(frame["step"])
        if step < first or step > last:
            continue
        rel_min = (step - event_step) * scale / 60.0
        for agent in frame.get("agents", []):
            flight = int(agent["flight_index"])
            if flight in previous_step:
                distance[flight] += max(0, step - previous_step[flight]) * previous_speed[flight]
            tracks[flight].append((rel_min, distance[flight]))
            previous_step[flight] = step
            previous_speed[flight] = float(agent.get("speed", 0.0) or 0.0)
    return dict(tracks), int(payload["target_flight_index"])


def completed_taxi_values(path: Path, high_priority: bool) -> list[float]:
    values = []
    for row in read_csv(path):
        if row.get("status") != "completed":
            continue
        is_high = float(row.get("priority") or 1.0) > 1.0
        if is_high == high_priority:
            values.append(float(row.get("taxi_sec") or 0.0) / 60.0)
    return values


def plot_priority(root: Path, out_dir: Path) -> None:
    payloads = {method: read_json(root / "priority" / f"{method}_micro_priority.json") for method in METHODS}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, method in zip(axes[:2], METHODS):
        tracks, target = local_distance_tracks(payloads[method])
        for flight, points in tracks.items():
            if not points:
                continue
            x, y = zip(*points)
            if flight == target:
                ax.plot(x, y, color=VIP, linewidth=3.0, label=f"Priority aircraft {target}", zorder=4)
            else:
                ax.plot(x, y, color=OTHER, alpha=0.55, linewidth=1.0, label="Concurrent aircraft", zorder=2)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper left", fontsize=8)
        ax.axvline(0.0, color="#374151", linestyle="--", linewidth=1.0)
        ax.set_title(METHODS[method]["label"])
        ax.set_xlabel("Time relative to priority boost (min)")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Local cumulative travel distance (graph units)")

    # Empirical aggregate companion: completed-aircraft executed taxi time by class.
    ax = axes[2]
    classes = [("Priority aircraft", True), ("Normal aircraft", False)]
    x = np.arange(len(classes))
    width = 0.34
    for offset, method in enumerate(METHODS):
        values = [float(np.mean(completed_taxi_values(root / "priority" / f"{method}_per_agent.csv", high))) if completed_taxi_values(root / "priority" / f"{method}_per_agent.csv", high) else np.nan for _, high in classes]
        bars = ax.bar(x + (offset - 0.5) * width, values, width=width, color=METHODS[method]["color"], alpha=0.85, label=METHODS[method]["label"])
        for bar, value, (_, high) in zip(bars, values, classes):
            if np.isfinite(value):
                n = len(completed_taxi_values(root / "priority" / f"{method}_per_agent.csv", high))
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [item[0] for item in classes])
    ax.set_ylabel("Mean executed taxi time (min)")
    ax.set_title("Priority response by aircraft class")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Priority perturbation: local response and service-time impact", fontsize=14, color=INK, y=1.01)
    fig.text(0.5, 0.01, "The trajectory panels use the actual event-relative micro-log; the bar panel includes completed aircraft only and reports n explicitly.", ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_dir / "fig_priority_preservation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the three empirical micro-mechanism figures.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_delay(root, out_dir)
    plot_closure(root, out_dir)
    plot_priority(root, out_dir)
    print(f"[OK] Three micro-mechanism figures written to {out_dir}")


if __name__ == "__main__":
    main()
