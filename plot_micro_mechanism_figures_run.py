"""Run the final three micro figures with an authoritative Delay endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_final_paper_figures import METHODS, INK
from plot_micro_mechanism_figures_final import find_event, plot_closure, plot_priority, read_csv, read_json


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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--delay-root", default=None, help="Optional root containing delay/.")
    parser.add_argument("--closure-root", default=None, help="Optional root containing closure/.")
    parser.add_argument("--priority-root", default=None, help="Optional root containing priority/.")
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="Root containing the protocol-matched baseline/ directory. Defaults to --result-root.",
    )
    args = parser.parse_args()
    root, out_dir = Path(args.result_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delay_root = Path(args.delay_root) if args.delay_root else root
    closure_root = Path(args.closure_root) if args.closure_root else root
    priority_root = Path(args.priority_root) if args.priority_root else root
    plot_delay(delay_root, out_dir)
    plot_closure(closure_root, out_dir)
    baseline_root = Path(args.baseline_root) if args.baseline_root else priority_root
    plot_priority(priority_root, out_dir, baseline_root=baseline_root)
    print(f"[OK] Final three micro-mechanism figures written to {out_dir}")


if __name__ == "__main__":
    main()
