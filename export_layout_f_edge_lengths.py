"""Export the modeled Layout F edge lengths.

The graph stores both directions for every physical edge. This utility reports
each physical edge once, using the Euclidean distance between its node
coordinates and the model's metre-per-coordinate-unit scale.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from env.taxi_graph_env import TaxiGraphEnv


def edge_class(u: int, v: int) -> str:
    """Classify stand connectors separately from the taxiway network."""
    endpoints = {u, v}
    if any(0 <= node < 30 or 2000 <= node < 2030 for node in endpoints):
        return "stand_connector"
    if 900 in endpoints or 901 in endpoints:
        return "runway_connection"
    return "taxiway_network"


def collect_edges(env: TaxiGraphEnv) -> list[dict[str, object]]:
    unique: dict[tuple[int, int], object] = {}
    for edge in env.edges:
        key = (min(edge.start, edge.end), max(edge.start, edge.end))
        unique.setdefault(key, edge)

    rows: list[dict[str, object]] = []
    for (u, v), edge in sorted(unique.items()):
        rows.append(
            {
                "start_node": u,
                "end_node": v,
                "edge_class": edge_class(u, v),
                "length_model_units": round(float(edge.length), 6),
                "length_m": round(float(edge.length) * env.L_SCALE, 3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Layout F topology edge lengths")
    parser.add_argument(
        "--out-dir",
        default="results/topology_layout_f",
        help="Output directory for CSV and Markdown files",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override the environment metre-per-unit scale; default uses env.L_SCALE",
    )
    args = parser.parse_args()

    env = TaxiGraphEnv(layout="F", num_agents=8, max_steps=1)
    scale = float(args.scale) if args.scale is not None else float(env.L_SCALE)
    rows = collect_edges(env)
    if args.scale is not None:
        for row in rows:
            row["length_m"] = round(float(row["length_model_units"]) * scale, 3)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "layout_f_edge_lengths.csv"
    md_path = out_dir / "layout_f_edge_lengths.md"

    fields = [
        "start_node",
        "end_node",
        "edge_class",
        "length_model_units",
        "length_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Layout F Edge Lengths\n\n")
        handle.write(f"- Physical edges: **{len(rows)}**\n")
        handle.write(f"- Length scale: **{scale:.3f} m/model unit**\n\n")
        handle.write("| Start | End | Class | Model units | Metres |\n")
        handle.write("|---:|---:|---|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['start_node']} | {row['end_node']} | "
                f"{row['edge_class']} | {row['length_model_units']:.3f} | "
                f"{row['length_m']:.3f} |\n"
            )

    print(f"Exported {len(rows)} physical Layout F edges")
    print(f"Scale: {scale:.3f} m/model unit")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
