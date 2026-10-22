"""Draw the Layout F airport topology and apron operating rules.

The topology is read from TaxiGraphEnv itself so the figure stays aligned with
the simulation implementation.  Both PNG and SVG are written for paper use.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

from env.taxi_graph_env import NodeType, TaxiGraphEnv


ROAD = "#B7C4D1"
ROAD_DARK = "#536575"
APRON = "#008C95"
APRON_LIGHT = "#BFE7E4"
GATE = "#5667A8"
SHADOW = "#8064A2"
RUNWAY = "#263746"
AIRCRAFT = "#D85C4A"
INK = "#24323D"
MUTED = "#6B7780"


def unique_edges(env):
    seen = set()
    for edge in env.edges:
        key = tuple(sorted((edge.start, edge.end)))
        if key not in seen:
            seen.add(key)
            yield key


def draw_aircraft(ax, x, y, angle=0.0, color=AIRCRAFT, scale=1.4):
    """Small aircraft silhouette; the only pictorial icon in the figure."""
    shape = [
        (0.95, 0.0), (0.12, 0.17), (-0.12, 0.62), (-0.28, 0.62),
        (-0.18, 0.15), (-0.70, 0.37), (-0.84, 0.26), (-0.34, -0.02),
        (-0.84, -0.26), (-0.70, -0.37), (-0.18, -0.15),
        (-0.28, -0.62), (-0.12, -0.62), (0.12, -0.17),
    ]
    ca, sa = math.cos(angle), math.sin(angle)
    points = []
    for px, py in shape:
        px *= scale
        py *= scale
        points.append((x + px * ca - py * sa, y + px * sa + py * ca))
    ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="white",
                         linewidth=0.7, zorder=12))


def arrow_on_segment(ax, p0, p1, color=APRON, alpha=1.0, lw=1.6, ms=8):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return
    start = (x0 + dx * 0.30, y0 + dy * 0.30)
    end = (x0 + dx * 0.70, y0 + dy * 0.70)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, alpha=alpha,
                                 shrinkA=0, shrinkB=0, zorder=8))


def draw_rule_inset(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.96, "APRON RULES", ha="left", va="top", fontsize=11,
            fontweight="bold", color=INK)

    # Rule 1: one-direction corridor reservation.
    y = 0.78
    ax.plot([0.08, 0.40], [y, y], color=APRON, linewidth=4, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((0.16, y), (0.35, y), arrowstyle="-|>",
                                 mutation_scale=12, color="white", linewidth=1.2))
    ax.plot([0.16, 0.35], [y - 0.10, y - 0.10], color="#C7CFD4", linewidth=2,
            linestyle="--")
    ax.text(0.44, y + 0.02, "Corridor lock", fontsize=9, color=INK, va="center")
    ax.text(0.44, y - 0.09, "one direction at a time", fontsize=8, color=MUTED, va="center")

    # Rule 2: node and edge occupancy.
    y = 0.51
    ax.plot([0.10, 0.38], [y, y], color=ROAD_DARK, linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.Circle((0.24, y), 0.055, facecolor="white", edgecolor=ROAD_DARK,
                            linewidth=1.8, zorder=3))
    ax.add_patch(plt.Circle((0.24, y), 0.018, facecolor=AIRCRAFT, edgecolor="none", zorder=4))
    ax.text(0.44, y + 0.02, "Occupancy", fontsize=9, color=INK, va="center")
    ax.text(0.44, y - 0.09, "reserve node / edge", fontsize=8, color=MUTED, va="center")

    # Rule 3: gate shadow clearance.
    y = 0.24
    ax.add_patch(Ellipse((0.24, y), 0.34, 0.17, facecolor=SHADOW, alpha=0.15,
                         edgecolor=SHADOW, linewidth=1.3, linestyle="--"))
    ax.add_patch(Rectangle((0.12, y - 0.035), 0.07, 0.07, facecolor=GATE,
                           edgecolor="white", linewidth=0.7))
    ax.add_patch(Rectangle((0.29, y - 0.035), 0.07, 0.07, facecolor=GATE,
                           edgecolor="white", linewidth=0.7))
    ax.text(0.44, y + 0.02, "Gate shadow", fontsize=9, color=INK, va="center")
    ax.text(0.44, y - 0.09, "clear adjacent stands", fontsize=8, color=MUTED, va="center")


def draw_figure(output: Path):
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
    pos = {node_id: node.position for node_id, node in env.nodes.items()}
    gate_ids = [node_id for node_id, node in env.nodes.items()
                if node.type == NodeType.GATE]
    backbone = [edge for edge in unique_edges(env)
                if edge[0] < 2000 and edge[1] < 2000]
    gate_edges = [edge for edge in unique_edges(env)
                  if edge[0] >= 2000 or edge[1] >= 2000]
    corridor_map = {c.id: c.nodes_seq for c in env.corridor_mgr.corridors}
    apron_corridors = {11, 12, 13, 14}

    fig = plt.figure(figsize=(13.0, 8.4), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)

    # Base network.
    for u, v in backbone:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=ROAD, linewidth=2.0, solid_capstyle="round", zorder=1)

    # Highlight the four central apron corridors defined by the environment.
    for cid in sorted(apron_corridors):
        seq = corridor_map[cid]
        for u, v in zip(seq[:-1], seq[1:]):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=APRON, linewidth=5.2, alpha=0.85,
                    solid_capstyle="round", zorder=3)
            arrow_on_segment(ax, pos[u], pos[v], color=APRON, alpha=0.95, lw=1.5, ms=8)

    # Gate stubs and stands.
    for u, v in gate_edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=GATE, linewidth=1.1, alpha=0.65, zorder=2)
    for gate_id in gate_ids:
        x, y = pos[gate_id]
        ax.add_patch(Rectangle((x - 0.8, y - 0.55), 1.6, 1.1,
                               facecolor=GATE, edgecolor="white", linewidth=0.5,
                               zorder=7))

    # Label every modeled node so the figure can be checked against the graph.
    for node_id, node in env.nodes.items():
        x, y = node.position
        if node_id < 30:
            ax.text(x, y, str(node_id), ha="center", va="center",
                    fontsize=5.8, color="white", fontweight="bold", zorder=10)
        elif 2000 <= node_id < 2030:
            ax.plot(x, y, "o", markersize=3.4, color=GATE, zorder=7)
            ax.text(x + 0.65, y + 0.45, str(node_id), ha="left", va="bottom",
                    fontsize=5.1, color=GATE, zorder=10)
        else:
            ax.plot(x, y, "o", markersize=3.0, color=ROAD_DARK, zorder=6)
            ax.text(x, y + 0.85, str(node_id), ha="center", va="bottom",
                    fontsize=5.4, color=INK, zorder=10)


    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    # Context labels kept intentionally sparse.
    ax.text(0, max(all_y) - 1.5, "APRON / GATE AREA", ha="center", va="bottom",
            fontsize=10, color=APRON, fontweight="bold")
    ax.text(0, min(all_y) + 1.0, "RUNWAY ACCESS", ha="center", va="top",
            fontsize=9, color=RUNWAY, fontweight="bold")
    ax.text(max(all_x) - 1.0, 31, "GATE\nSTANDS", ha="left", va="center",
            fontsize=8.5, color=GATE, fontweight="bold", linespacing=1.2)

    ax.set_aspect("equal")
    ax.set_xlim(min(all_x) - 6, max(all_x) + 10)
    ax.set_ylim(min(all_y) - 7, max(all_y) + 8)
    ax.axis("off")

    fig.suptitle("Layout F Airport Taxi Network", fontsize=15,
                 fontweight="bold", color=INK, y=0.97)
    fig.text(0.5, 0.018, "Layout F | green: harbor apron corridors | node labels: modeled graph IDs",
             ha="center", fontsize=8.5, color=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.svg')}")


def main():
    parser = argparse.ArgumentParser(description="Draw the Layout F topology and apron rules")
    parser.add_argument("--output", default="paper_figure/fig_airport_topology_apron_rules.png")
    args = parser.parse_args()
    draw_figure(Path(args.output))


if __name__ == "__main__":
    main()
