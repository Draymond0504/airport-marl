"""Export the primary and executor-intervention metrics from layered runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(result_dir: Path, method: str) -> dict:
    path = result_dir / f"{method}_summary.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    mappo = load_summary(result_dir, "MAPPO")
    fcfs = load_summary(result_dir, "FCFS")

    rows = [
        ("Target completion rate", fmt(mappo.get("target_completion_rate_pct"), "%"), fmt(fcfs.get("target_completion_rate_pct"), "%")),
        ("Target completed flights", mappo.get("target_success_count", 0), fcfs.get("target_success_count", 0)),
        ("Timeout count", mappo.get("timeout_count", 0), fcfs.get("timeout_count", 0)),
        ("Collision/safety violations", mappo.get("safety_violation_count", 0), fcfs.get("safety_violation_count", 0)),
        ("Conflict-protection interventions", mappo.get("conflict_protection_events", 0), fcfs.get("conflict_protection_events", 0)),
        ("Stall-triggered replanning", mappo.get("stall_replanning_events", 0), fcfs.get("stall_replanning_events", 0)),
        ("Target-equivalent throughput", fmt(mappo.get("target_throughput_per_hour"), " flights/h"), fmt(fcfs.get("target_throughput_per_hour"), " flights/h")),
        ("Average on-route waiting", fmt(mappo.get("avg_onroute_wait_min"), " min"), fmt(fcfs.get("avg_onroute_wait_min"), " min")),
    ]
    lines = [
        "| Metric | MAPPO | FCFS+A* |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {label} | {left} | {right} |" for label, left, right in rows)
    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else result_dir / "layered_execution_metrics.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
