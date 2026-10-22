"""Generate the final three empirical micro-mechanism figures."""

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
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_final_paper_figures import METHODS, INK, draw_network, layout_f_geometry


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_event(events: list[dict], event_type: str) -> dict:
    matches = [event for event in events if event.get("type") == event_type]
    if not matches:
        raise RuntimeError(f"No {event_type!r} event found")
    return min(matches, key=lambda event: int(event["step"]))


def plot_delay(root: Path, out_dir: Path) -> None:
    event = find_event(read_json(root / "delay" / "MAPPO_events.json")["events"], "delay_interruption")
    scale = float(read_json(root / "delay" / "experiment_meta.json")["time_scale"])
    start_min = int(event["step"]) * scale / 60.0
    end_min = start_min + int(event.get("duration", 0)) * scale / 60.0

    # Use an independent event panel rather than overlaying the zoomed curve
    # on the full cumulative curve.  This keeps both the endpoint and the
    # interruption response readable at paper scale.
    fig = plt.figure(figsize=(11.2, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.34, wspace=0.24)
    full_ax = fig.add_subplot(grid[0, :])
    event_ax = fig.add_subplot(grid[1, 0])
    wait_ax = fig.add_subplot(grid[1, 1])

    curves = {}
    full_end = 0.0
    for method, spec in METHODS.items():
        rows = read_csv(root / "delay" / f"{method}_timeseries.csv")
        summary = read_json(root / "delay" / f"{method}_summary.json")
        total = int(summary["total_count"])
        points = [(float(row["time_min"]), 100.0 * float(row["cumulative_success"]) / total) for row in rows]
        final_rate = 100.0 * float(summary["success_count"]) / total
        if points and final_rate > points[-1][1]:
            points.append((points[-1][0], final_rate))
        curves[method] = points
        full_end = max(full_end, points[-1][0])
        x, y = zip(*points)
        full_ax.plot(x, y, color=spec["color"], linewidth=2.0, label=spec["label"])
        full_ax.scatter(x[-1], y[-1], color=spec["color"], s=22, zorder=3)
        full_ax.annotate(f"{y[-1]:.2f}%", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=spec["color"])

    full_ax.axvspan(start_min, end_min, color="#94A3B8", alpha=0.18)
    full_ax.axvline(start_min, color="#475569", linestyle="--", linewidth=0.9)
    full_ax.axvline(end_min, color="#475569", linestyle=":", linewidth=0.9)
    full_ax.set_xlim(0, full_end)
    full_ax.set_ylim(0, 105)
    full_ax.set_xlabel("Simulation time (min)")
    full_ax.set_ylabel("Cumulative completed tasks (%)")
    full_ax.set_title("Overall completion and mid-service interruption")
    full_ax.legend(frameon=False, loc="lower right")
    full_ax.grid(axis="y", alpha=0.18)
    full_ax.spines[["top", "right"]].set_visible(False)

    local_start = max(0.0, start_min - 30.0)
    local_end = min(full_end, end_min + 45.0)
    local_values = []
    for method, spec in METHODS.items():
        points = [(x, y) for x, y in curves[method] if local_start <= x <= local_end]
        if not points:
            points = [p for p in curves[method] if p[0] <= local_end][-1:]
        x, y = zip(*points)
        local_values.extend(y)
        event_ax.plot(x, y, color=spec["color"], linewidth=2.0, marker=".", markersize=2.5, label=spec["label"])
    event_ax.axvspan(start_min, end_min, color="#94A3B8", alpha=0.22)
    event_ax.axvline(start_min, color="#475569", linestyle="--", linewidth=0.9, label="Interruption start")
    event_ax.axvline(end_min, color="#475569", linestyle=":", linewidth=0.9, label="Resumption")
    event_ax.set_xlim(local_start, local_end)
    if local_values:
        event_ax.set_ylim(max(0.0, min(local_values) - 3.0), min(105.0, max(local_values) + 5.0))
    event_ax.set_xlabel("Simulation time (min)")
    event_ax.set_ylabel("Cumulative completed tasks (%)")
    event_ax.set_title("Event-window response")
    event_ax.grid(axis="y", alpha=0.18)
    event_ax.spines[["top", "right"]].set_visible(False)
    event_ax.legend(frameon=False, fontsize=7, loc="best")

    waits = {}
    for method in METHODS:
        rows = read_csv(root / "delay" / f"{method}_per_agent.csv")
        waits[method] = [float(row.get("onroute_wait_sec") or 0.0) / 60.0 for row in rows]
    xmax = max((max(values, default=0.0) for values in waits.values()), default=1.0)
    xmax = max(10.0, float(np.ceil(xmax / 10.0) * 10.0))
    wait_grid = np.linspace(0.0, xmax, 250)
    for method, spec in METHODS.items():
        values = np.asarray(waits[method], dtype=float)
        wait_ax.plot(wait_grid, [100.0 * float(np.mean(values >= threshold)) for threshold in wait_grid], color=spec["color"], linewidth=2.0, label=spec["label"])
        wait_ax.axvline(float(np.median(values)), color=spec["color"], linestyle=":", linewidth=0.8, alpha=0.7)
    wait_ax.set_xlim(0, xmax)
    wait_ax.set_ylim(0, 105)
    wait_ax.set_xlabel("On-route waiting threshold (min)")
    wait_ax.set_ylabel("Flights exceeding threshold (%)")
    wait_ax.set_title("Tail probability of on-route waiting")
    wait_ax.legend(frameon=False, loc="upper right")
    wait_ax.grid(axis="y", alpha=0.18)
    wait_ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Delay mechanism: interruption, recovery, and local waiting", fontsize=14, color=INK, y=0.985)
    fig.text(0.5, 0.012, f"Actual interruption: t={start_min:.1f} to {end_min:.1f} min; event response is shown in a separate panel.", ha="center", fontsize=8, color="#5F6B72")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.93)
    fig.savefig(out_dir / "fig_delay_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def observed_transitions(path: Path) -> list[tuple[int, int, int]]:
    transitions = []
    for row in read_csv(path):
        before = int(row["position_before"])
        after = int(row["position_after"])
        if int(row["on_edge_before"]) == 1 and before != after:
            transitions.append((int(row["step"]), before, after))
    return transitions


def shares_in_window(transitions: list[tuple[int, int, int]], start: int, end: int) -> tuple[dict[tuple[int, int], float], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for step, before, after in transitions:
        if start <= step < end:
            counts[tuple(sorted((before, after)))] += 1
    total = sum(counts.values())
    return ({edge: count / total for edge, count in counts.items()} if total else {}, total)

def occupancy_in_window(path: Path, start: int, end: int) -> tuple[dict[tuple[int, int], float], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in read_csv(path):
        step = int(row["step"])
        if start <= step < end:
            edge = tuple(sorted((int(row["edge_start"]), int(row["edge_end"]))))
            counts[edge] += 1
    total = sum(counts.values())
    return ({edge: count / total for edge, count in counts.items()} if total else {}, total)


def plot_closure(root: Path, out_dir: Path) -> None:
    event = find_event(read_json(root / "closure" / "MAPPO_events.json")["events"], "closure")
    trigger = int(event["step"])
    duration = int(event["duration"])
    closed = tuple(sorted(int(value) for value in event["edge_nodes"]))
    windows = [("Before closure", trigger - duration, trigger), ("Closure active", trigger, trigger + duration)]
    pos, edges = layout_f_geometry()
    occupancy_available = all((root / "closure" / f"{method}_edge_occupancy.csv").exists() for method in METHODS)
    transitions = {method: observed_transitions(root / "closure" / f"{method}_raw_actions.csv") for method in METHODS}
    shares: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    totals: dict[tuple[str, str], int] = {}
    for method in METHODS:
        for label, start, end in windows:
            if occupancy_available:
                shares[(method, label)], totals[(method, label)] = occupancy_in_window(
                    root / "closure" / f"{method}_edge_occupancy.csv", start, end
                )
            else:
                shares[(method, label)], totals[(method, label)] = shares_in_window(transitions[method], start, end)
    sample_label = "edge-occupancy samples" if occupancy_available else "observed transitions"
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
                colour = cmap(0.12 + 0.88 * float(norm(share)))
                linewidth = max(1.55, 1.2 + 5.0 * share / max(vmax, 1e-9))
                ax.plot([x0, x1], [y0, y1], color=colour, linewidth=linewidth, alpha=0.96, solid_capstyle="round", zorder=3)
            if label == "Closure active":
                x0, y0 = pos[closed[0]]
                x1, y1 = pos[closed[1]]
                ax.plot([x0, x1], [y0, y1], color="#7F1D1D", linewidth=5.0, linestyle=(0, (2, 1)), zorder=5)
            ax.set_title(f"{METHODS[method]['label']} | {label}\nNormalized within-window edge occupancy", fontsize=10, color=INK)
            ax.text(0.02, 0.03, f"steps {start}–{end - 1}", transform=ax.transAxes, fontsize=8, color="#4B5563", ha="left", va="bottom")
            if not local:
                ax.text(0.5, 0.5, "No observed transitions\nin this window", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#64748B")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Share of observed edge transitions")
    fig.suptitle(f"Relative edge-occupancy response to closure of link {closed[0]}–{closed[1]}", fontsize=14, color=INK, y=0.985)
    fig.text(0.5, 0.012, "Each panel is normalized by its own edge-occupancy observations; the dashed dark-red segment is unavailable only during the closure window.", ha="center", fontsize=8, color="#5F6B72")
    fig.subplots_adjust(left=0.025, right=0.95, bottom=0.08, top=0.90)
    fig.savefig(out_dir / "fig_closure_edge_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def completed_taxi_values(path: Path, high_priority: bool) -> list[float]:
    values = []
    for row in read_csv(path):
        if row.get("status") != "completed":
            continue
        is_high = float(row.get("priority") or 1.0) > 1.0
        if is_high == high_priority:
            values.append(float(row.get("taxi_sec") or 0.0) / 60.0)
    return values


def plot_priority(root: Path, out_dir: Path, baseline_root: Path | None = None) -> None:
    # Preserve the requested two-panel layout, replacing mock-up values with
    # actual completed-aircraft measurements from the latest rolling run.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    high_values = []
    for method in METHODS:
        values = completed_taxi_values(root / "priority" / f"{method}_per_agent.csv", True)
        high_values.append(float(np.mean(values)) if values else np.nan)
    bars = axes[0].bar(np.arange(2), high_values, color=[METHODS[m]["color"] for m in METHODS], width=0.58, alpha=0.82)
    for bar, method, value in zip(bars, METHODS, high_values):
        values = completed_taxi_values(root / "priority" / f"{method}_per_agent.csv", True)
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}\n(n={len(values)})", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Priority-aircraft executed taxi time")
    axes[0].set_ylabel("Mean executed taxi time (min)")
    axes[0].set_xticks(np.arange(2), [METHODS[m]["label"] for m in METHODS])

    baseline_dir = (baseline_root or root) / "baseline"
    baseline_values, priority_values, baseline_ns, priority_ns = [], [], [], []
    for method in METHODS:
        baseline = completed_taxi_values(baseline_dir / f"{method}_per_agent.csv", False)
        priority = completed_taxi_values(root / "priority" / f"{method}_per_agent.csv", False)
        baseline_values.append(float(np.mean(baseline)) if baseline else np.nan)
        priority_values.append(float(np.mean(priority)) if priority else np.nan)
        baseline_ns.append(len(baseline))
        priority_ns.append(len(priority))
    x = np.arange(2)
    width = 0.32
    b1 = axes[1].bar(x - width / 2, baseline_values, width, color=[METHODS[m]["color"] for m in METHODS], alpha=0.30, label="Baseline")
    b2 = axes[1].bar(x + width / 2, priority_values, width, color=[METHODS[m]["color"] for m in METHODS], alpha=0.90, label="Priority perturbation")
    for bars_group, values, ns in [(b1, baseline_values, baseline_ns), (b2, priority_values, priority_ns)]:
        for bar, value, n in zip(bars_group, values, ns):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)
    axes[1].set_title("Normal-aircraft service time: baseline vs priority")
    axes[1].set_ylabel("Mean executed taxi time (min)")
    axes[1].set_xticks(x, [METHODS[m]["label"] for m in METHODS])
    axes[1].legend(handles=[Patch(facecolor="#64748B", alpha=0.30, label="Baseline"), Patch(facecolor="#64748B", alpha=0.90, label="Priority perturbation")], frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Priority preservation and normal-service impact", fontsize=14, color=INK, y=1.02)
    fig.text(0.5, -0.03, "Means are computed over completed aircraft; the priority protocol boosts three selected aircraft on entry.", ha="center", fontsize=8, color="#5F6B72")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_priority_preservation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final empirical micro-mechanism figures")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_delay(root, out_dir)
    plot_closure(root, out_dir)
    plot_priority(root, out_dir)
    print(f"[OK] Final three micro-mechanism figures written to {out_dir}")


if __name__ == "__main__":
    main()
