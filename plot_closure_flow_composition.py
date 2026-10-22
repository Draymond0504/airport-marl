"""Compare closure-window edge-flow composition, not absolute movement volume."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plot_final_paper_figures import METHODS, draw_network, layout_f_geometry, read_csv, read_json


def count_edges(rows: list[dict], start: int, end: int) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        if start <= int(float(row["step"])) < end:
            edge = tuple(sorted((int(row["edge_start"]), int(row["edge_end"]))))
            counts[edge] += 1
    return dict(counts)


def hhi(counts: dict[tuple[int, int], int]) -> float:
    total = sum(counts.values())
    return sum((value / total) ** 2 for value in counts.values()) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot closure edge-flow composition.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--window-steps", type=int, default=400)
    args = parser.parse_args()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = read_json(root / "closure" / "MAPPO_events.json")["events"]
    event = min((item for item in events if item.get("type") == "closure"), key=lambda item: int(item["step"]))
    trigger = int(event["step"])
    closed = tuple(sorted(int(item) for item in event["edge_nodes"]))
    width = min(int(args.window_steps), trigger)
    windows = [("Pre-event", trigger - width, trigger), ("Closure active", trigger, trigger + width)]
    raw = {method: read_csv(root / "closure" / f"{method}_edge_traversals.csv") for method in METHODS}
    counts = {(method, label): count_edges(raw[method], start, end) for method in METHODS for label, start, end in windows}
    shares = {
        key: {edge: value * 100.0 / max(sum(local.values()), 1) for edge, value in local.items()}
        for key, local in counts.items()
    }
    vmax = max((value for values in shares.values() for value in values.values()), default=1.0)
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    cmap = plt.get_cmap("YlOrRd")
    pos, edges = layout_f_geometry()

    fig = plt.figure(figsize=(12.2, 8.25))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.065], hspace=0.20, wspace=0.08)
    axes = [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)]
    colorbar_ax = fig.add_subplot(grid[:, 2])
    for row, method in enumerate(METHODS):
        for col, (label, start, end) in enumerate(windows):
            ax = axes[row][col]
            draw_network(ax, pos, edges)
            local_shares = shares[(method, label)]
            local_counts = counts[(method, label)]
            for edge, share in local_shares.items():
                if edge[0] not in pos or edge[1] not in pos:
                    continue
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                ax.plot([x0, x1], [y0, y1], color=cmap(norm(share)),
                        linewidth=1.05 + 4.5 * share / max(vmax, 1.0), solid_capstyle="round", zorder=3)
            if label == "Closure active":
                x0, y0 = pos[closed[0]]
                x1, y1 = pos[closed[1]]
                ax.plot([x0, x1], [y0, y1], color="#7F1D1D", linewidth=5.2,
                        linestyle=(0, (2, 1)), zorder=5)
            total = sum(local_counts.values())
            ax.set_title(f"{METHODS[method]['label']} | {label}\n{total} traversals | {len(local_counts)} used edges | HHI={hhi(local_counts):.4f}",
                         fontsize=9.4, color="#263238")
            ax.text(0.02, 0.035, f"steps {start}-{end - 1}", transform=ax.transAxes,
                    fontsize=7.8, color="#4B5563", ha="left", va="bottom")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("Share of observed edge traversals within each panel (%)", fontsize=9)
    fig.suptitle(f"Edge-flow composition around closure of link {closed[0]}-{closed[1]}", fontsize=14, color="#263238", y=0.985)
    fig.text(0.5, 0.012,
             "All panels use equal 400-step windows. Colours show each edge's share of panel-level movement, isolating flow concentration from total movement volume; the dashed segment is unavailable in the right column.",
             ha="center", fontsize=7.8, color="#5F6B72")
    fig.subplots_adjust(left=0.025, right=0.95, bottom=0.08, top=0.90)
    fig.savefig(out_dir / "fig_closure_edge_flow.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_closure_edge_flow_composition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote closure flow-composition figure at event step {trigger}.")


if __name__ == "__main__":
    main()
