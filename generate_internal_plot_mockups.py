"""Create five explicitly non-empirical plot mockups matching project plots.

These assets are for internal narrative and layout review only. They are not
derived from model runs and must never be presented as measured results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from plot_edge_heatmap import build_layout_f_pos_edges


STATUS = "ILLUSTRATIVE INTERNAL ONLY - NOT EMPIRICAL RESULTS"
MAPPO = "#2E86AB"
FCFS = "#E84855"
GUIDED = "#1B7F79"
NOGUIDE = "#C56A3D"


def stamp(fig: plt.Figure) -> None:
    fig.text(0.5, 0.012, STATUS, ha="center", va="bottom", fontsize=8,
             color="#9B3030", fontweight="bold")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ablation_reward_curve(output_dir: Path, rng: np.random.Generator) -> None:
    episodes = np.arange(0, 10001, 100)
    seed_curves: dict[str, list[np.ndarray]] = {"MAPPO-Guided": [], "MAPPO-NoGuide": []}
    for index in range(3):
        guided = -190 + 940 * (1 - np.exp(-episodes / (1550 + 100 * index)))
        no_guide = -210 + 750 * (1 - np.exp(-episodes / (3200 + 180 * index)))
        seed_curves["MAPPO-Guided"].append(guided + rng.normal(0, 28, len(episodes)))
        seed_curves["MAPPO-NoGuide"].append(no_guide + rng.normal(0, 35, len(episodes)))
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for method, color in (("MAPPO-Guided", GUIDED), ("MAPPO-NoGuide", NOGUIDE)):
        matrix = np.vstack(seed_curves[method])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        ax.plot(episodes, mean, color=color, linewidth=2.2, label=method)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.18)
        for seed, values in enumerate(matrix, start=1):
            for ep, value in zip(episodes, values):
                rows.append({"episode": int(ep), "method": method, "seed": seed, "total_reward": round(float(value), 3)})
    write_csv(output_dir / "illustrative_ablation_training_log.csv", rows)
    ax.set(title="Ablation: convergence of episode reward", xlabel="Episode", ylabel="Total reward")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    stamp(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "fig_ablation_reward_convergence_internal.png", dpi=300)
    plt.close(fig)


def baseline_bar_chart(output_dir: Path) -> None:
    rows = [
        {"metric": "Safe completion (%)", "FCFS+A*": 88.5, "MAPPO-Guided": 90.5},
        {"metric": "Full success (%)", "FCFS+A*": 78.5, "MAPPO-Guided": 83.5},
        {"metric": "Mean taxi time (min)", "FCFS+A*": 0.68, "MAPPO-Guided": 0.76},
        {"metric": "Mean waiting time (min)", "FCFS+A*": 0.11, "MAPPO-Guided": 0.08},
        {"metric": "Collision events / 100", "FCFS+A*": 0.23, "MAPPO-Guided": 0.14},
        {"metric": "Throughput (flt/hr)", "FCFS+A*": 21.8, "MAPPO-Guided": 22.4},
    ]
    write_csv(output_dir / "illustrative_baseline_metrics.csv", rows)
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.1))
    for ax, row in zip(axes.flat, rows):
        labels = ["FCFS+A*", "MAPPO-Guided"]
        vals = [row[label] for label in labels]
        bars = ax.bar(labels, vals, color=[FCFS, MAPPO], width=0.58, alpha=0.88)
        for bar, value in zip(bars, vals):
            fmt = f"{value:.1f}" if value >= 2 else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value, fmt, ha="center", va="bottom", fontsize=8)
        ax.set_title(row["metric"], fontsize=10)
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.tick_params(axis="x", labelrotation=10, labelsize=8)
    fig.suptitle("Baseline scenario: FCFS+A* versus planning-guided MAPPO", fontsize=13, y=0.985)
    stamp(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_dir / "fig_baseline_algorithm_comparison_internal.png", dpi=300)
    plt.close(fig)


def draw_heatmap(
    ax,
    pos,
    all_edges,
    edge_rates: dict[tuple[int, int], float],
    title: str,
    closed: tuple[int, int] | None,
    vmax: float,
    summary: str,
) -> None:
    """Draw a directed, exposure-normalized edge-use map.

    The map is intentionally an edge-use view rather than a reconstructed
    aircraft trajectory.  Arrowheads show movement direction, while the
    shared color scale represents traversals per 100 exposed aircraft-steps.
    """
    cmap = matplotlib.colormaps["YlOrRd"]
    norm = colors.Normalize(vmin=0, vmax=vmax)
    for u, v in all_edges:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#D8D8D8", lw=0.85, zorder=1)
    for (u, v), rate in edge_rates.items():
        if rate <= 0 or u not in pos or v not in pos:
            continue
        color = cmap(norm(min(rate, vmax)))
        width = 1.0 + 5.0 * min(rate / vmax, 1)
        start = np.asarray(pos[u], dtype=float)
        end = np.asarray(pos[v], dtype=float)
        vector = end - start
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=width, zorder=2)
        # A short arrow segment avoids covering the node labels while still
        # making the directed edge-use interpretation explicit.
        arrow_start = start + 0.56 * vector
        arrow_end = start + 0.78 * vector
        ax.add_patch(FancyArrowPatch(
            arrow_start, arrow_end, arrowstyle="-|>", mutation_scale=8,
            linewidth=0.0, color=color, zorder=3,
        ))
    if closed:
        u, v = closed
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#303030", lw=3.0, ls="--", zorder=5)
        ax.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2 + 4.6,
                "Closed", color="#303030", ha="center", fontsize=8.5, weight="bold")
    # Keep the quantitative summary away from the entry/exit labels and the
    # active lower taxiway, which are the visual evidence in this figure.
    ax.text(0.02, 0.97, summary, transform=ax.transAxes, fontsize=7.6,
            va="top", ha="left", color="#2D2D2D",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BEBEBE", "alpha": 0.92})
    for node, label, ha in ((900, "Entry", "right"), (901, "Exit", "left")):
        if node in pos:
            ax.text(pos[node][0], pos[node][1] - 2.5, label, fontsize=7.5,
                    color="#565656", ha=ha, va="top")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def closure_heatmaps(output_dir: Path) -> None:
    pos, backbone, gates = build_layout_f_pos_edges()
    all_edges = backbone + gates
    # Rates are normalized by active, closure-exposed aircraft-steps.  This
    # prevents a lightly populated post-closure window from looking like a
    # rerouting failure merely because it contains fewer absolute movements.
    base = {
        (900, 101): 1.45, (101, 102): 1.45, (102, 103): 1.38,
        (103, 104): 1.35, (104, 901): 1.34, (102, 202): 0.46,
    }
    mappo_after = {
        (900, 101): 1.31, (101, 201): 1.12, (201, 202): 1.12,
        (202, 203): 1.03, (203, 204): 1.00, (204, 104): 1.00,
        (104, 901): 0.99, (202, 304): 0.72, (304, 305): 0.68,
        (305, 306): 0.65,
    }
    fcfs_after = {
        (900, 101): 0.76, (101, 201): 0.51, (201, 202): 0.51,
        (202, 203): 0.48, (203, 204): 0.44, (204, 104): 0.43,
        (104, 901): 0.42, (202, 304): 0.24, (304, 305): 0.21,
    }
    panels = (
        ("MAPPO-Guided", "Before closure", base, None,
         "Exposed: 18 aircraft\nWindow throughput: 14.5 moves / 100 ac-steps"),
        ("MAPPO-Guided", "After closure", mappo_after, (102, 103),
         "Exposed: 18 aircraft\nWindow throughput: 13.1 moves / 100 ac-steps"),
        ("FCFS+A*", "Before closure", base, None,
         "Exposed: 18 aircraft\nWindow throughput: 14.5 moves / 100 ac-steps"),
        ("FCFS+A*", "After closure", fcfs_after, (102, 103),
         "Exposed: 18 aircraft\nWindow throughput: 7.6 moves / 100 ac-steps"),
    )
    rows = []
    for method, phase, rates, _, summary in panels:
        for (u, v), rate in rates.items():
            rows.append({
                "method": method,
                "phase": phase,
                "edge_start": u,
                "edge_end": v,
                "normalized_edge_rate_per_100_exposed_ac_steps": rate,
                "panel_summary": summary.replace("\n", "; "),
            })
    write_csv(output_dir / "illustrative_closure_edge_counts.csv", rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    draw_heatmap(axes[0, 0], pos, all_edges, base, "(a) MAPPO before closure", None, 1.5, panels[0][4])
    draw_heatmap(axes[0, 1], pos, all_edges, mappo_after, "(b) MAPPO after closure", (102, 103), 1.5, panels[1][4])
    draw_heatmap(axes[1, 0], pos, all_edges, base, "(c) FCFS+A* before closure", None, 1.5, panels[2][4])
    draw_heatmap(axes[1, 1], pos, all_edges, fcfs_after, "(d) FCFS+A* after closure", (102, 103), 1.5, panels[3][4])
    scalar = cm.ScalarMappable(cmap="YlOrRd", norm=colors.Normalize(vmin=0, vmax=1.5))
    scalar.set_array([])
    fig.subplots_adjust(left=0.03, right=0.86, bottom=0.08, top=0.90, wspace=0.04, hspace=0.12)
    color_axis = fig.add_axes([0.89, 0.22, 0.018, 0.56])
    fig.colorbar(scalar, cax=color_axis, label="Edge traversals / 100 exposed aircraft-steps")
    fig.suptitle("Taxiway-closure response: edge-use redistribution", fontsize=13, y=0.965)
    stamp(fig)
    fig.savefig(output_dir / "fig_closure_before_after_heatmaps_internal.png", dpi=300)
    plt.close(fig)


def delay_boxplot(output_dir: Path, rng: np.random.Generator) -> None:
    mappo = np.clip(rng.gamma(shape=2.0, scale=1.7, size=180), 0.1, 13)
    fcfs = np.clip(rng.gamma(shape=2.8, scale=2.1, size=180), 0.1, 19)
    rows = ([{"method": "MAPPO-Guided", "waiting_time_min": round(float(v), 4)} for v in mappo] +
            [{"method": "FCFS+A*", "waiting_time_min": round(float(v), 4)} for v in fcfs])
    write_csv(output_dir / "illustrative_delay_wait_samples.csv", rows)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    box = ax.boxplot([fcfs, mappo], tick_labels=["FCFS+A*", "MAPPO-Guided"], patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.8})
    for patch, color in zip(box["boxes"], [FCFS, MAPPO]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    means = [float(np.mean(fcfs)), float(np.mean(mappo))]
    ax.scatter([1, 2], means, marker="D", s=38, color="#202020", zorder=4, label="Mean")
    ax.set(title="Delay scenario: waiting-time distribution", ylabel="Total waiting time (min)")
    ax.grid(axis="y", alpha=0.24, linestyle="--")
    ax.legend(frameon=False)
    stamp(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "fig_delay_wait_boxplot_internal.png", dpi=300)
    plt.close(fig)


def priority_timespace(output_dir: Path) -> None:
    event = 0
    time = np.arange(-90, 151, 2)
    corridor_length_m = 220.0

    def route(points):
        xp, yp = zip(*points)
        return np.interp(time, xp, yp, left=np.nan, right=np.nan)

    # The same local interaction is rendered in both panels: a focal VIP and
    # nearby aircraft competing for one shared corridor.  A flat segment is a
    # real operational statement (waiting), unlike an arbitrary node-number
    # plateau in a network-wide plot.
    tracks = {
        "MAPPO-Guided": {
            "VIP": route([(-90, 0), (-30, 42), (0, 78), (20, 130), (52, 220), (90, 220)]),
            "Normal A": route([(-90, 0), (-30, 34), (0, 48), (42, 48), (72, 112), (118, 220)]),
            "Normal B": route([(-90, 112), (-30, 155), (0, 184), (22, 220), (88, 220)]),
        },
        "FCFS+A*": {
            "VIP": route([(-90, 0), (-30, 42), (0, 76), (78, 76), (102, 126), (142, 220)]),
            "Normal A": route([(-90, 0), (-30, 40), (0, 96), (28, 158), (60, 220), (90, 220)]),
            "Normal B": route([(-90, 0), (-30, 20), (0, 44), (68, 44), (98, 104), (145, 220)]),
        },
    }
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.1), sharex=True, sharey=True)
    for ax, (method, method_tracks) in zip(axes, tracks.items()):
        for label, values in method_tracks.items():
            valid = ~np.isnan(values)
            color = "#FF982B" if label == "VIP" else ("#4E83B6" if label == "Normal A" else "#A9C5E8")
            lw = 2.8 if label == "VIP" else 1.55
            ax.plot(time[valid], values[valid], color=color, lw=lw,
                    label="VIP flight" if label == "VIP" else "Nearby normal flight")
            for t, value in zip(time[valid], values[valid]):
                rows.append({
                    "method": method,
                    "aircraft_group": label,
                    "relative_time_s": int(t),
                    "corridor_position_m": round(float(value), 2),
                })
        ax.axvline(event, color="#F39C12", lw=1.3, ls="--")
        ax.text(5, corridor_length_m - 10, "Priority activated", color="#E68A00", fontsize=9, weight="bold", va="top")
        if method == "MAPPO-Guided":
            ax.annotate("Normal A yields", xy=(22, 48), xytext=(42, 22), fontsize=8,
                        color="#3E6F9B", arrowprops={"arrowstyle": "->", "color": "#3E6F9B", "lw": 1})
            ax.annotate("VIP released", xy=(22, 134), xytext=(40, 165), fontsize=8,
                        color="#D16C00", arrowprops={"arrowstyle": "->", "color": "#D16C00", "lw": 1})
        else:
            ax.annotate("VIP waits in queue", xy=(42, 76), xytext=(15, 118), fontsize=8,
                        color="#D16C00", arrowprops={"arrowstyle": "->", "color": "#D16C00", "lw": 1})
            ax.annotate("Normal A proceeds first", xy=(18, 136), xytext=(38, 184), fontsize=8,
                        color="#3E6F9B", arrowprops={"arrowstyle": "->", "color": "#3E6F9B", "lw": 1})
        ax.set_ylabel("Position along shared corridor (m)")
        ax.set_title(f"({ 'a' if method == 'MAPPO-Guided' else 'b' }) {method} evolution")
        ax.set_ylim(-8, corridor_length_m + 10)
        ax.set_yticks([0, 55, 110, 165, 220])
        ax.grid(alpha=0.25, linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=True, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time relative to priority activation (s)")
    fig.suptitle("Priority-change scenario: time-space trajectory evolution", fontsize=13, y=0.985)
    stamp(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_dir / "fig_priority_timespace_evolution_internal.png", dpi=300)
    plt.close(fig)
    write_csv(output_dir / "illustrative_priority_timespace.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate internal plot-type mockups")
    parser.add_argument("--output-dir", default="results/internal_reference_plot_mockups")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ablation_reward_curve(output_dir, rng)
    baseline_bar_chart(output_dir)
    closure_heatmaps(output_dir)
    delay_boxplot(output_dir, rng)
    priority_timespace(output_dir)
    manifest = {
        "status": STATUS,
        "purpose": "Internal review of five requested paper plot types.",
        "source": "Synthetic illustrative values, not model execution output.",
        "files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "README_INTERNAL_ONLY.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote internal plot mockups to: {output_dir}")


if __name__ == "__main__":
    main()
