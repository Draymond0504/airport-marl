"""Export unambiguous paper tables from operational rolling experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = ("baseline", "delay", "closure", "priority")
METHODS = (("MAPPO", "MAPPO-Guided"), ("FCFS", "FCFS+A*"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in SCENARIOS:
        for prefix, label in METHODS:
            summary = json.loads((root / scenario / f"{prefix}_summary.json").read_text(encoding="utf-8"))
            rows.append({
                "scenario": scenario,
                "method": label,
                "flights": summary["total_count"],
                "window_safe_service_pct": summary["safe_completion_rate_pct"],
                "conflict_events": summary["collision_count"],
                "stall_recovery_triggers": summary["deadlock_count"],
                "moving_time_min": summary["avg_moving_min"],
                "onroute_wait_min": summary["avg_onroute_wait_min"],
                "entry_queue_min": summary["avg_delay_min"],
                "total_mission_min": summary["avg_total_time_min"],
                "throughput_flights_hr": summary["throughput_per_hour"],
            })

    fields = list(rows[0])
    with (out_dir / "table_4_4_operational_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Scenario", "Method", "N", "Window-safe service (%)", "Conflict events",
        "Stall-recovery triggers", "Moving (min)", "On-route wait (min)",
        "Entry queue (min)", "Total mission (min)", "Throughput (flt/hr)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {flights} | {window_safe_service_pct:.2f} | "
            "{conflict_events} | {stall_recovery_triggers} | {moving_time_min:.2f} | "
            "{onroute_wait_min:.2f} | {entry_queue_min:.2f} | {total_mission_min:.2f} | "
            "{throughput_flights_hr:.2f} |".format(**row)
        )
    (out_dir / "table_4_4_operational_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Operational paper table written to {out_dir}")


if __name__ == "__main__":
    main()
