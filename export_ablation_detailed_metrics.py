"""Export task-time decomposition, PDI sample sizes, and calibrated apron time.

The calibrated value is a reporting sensitivity analysis. It does not alter
the learned policy or relabel the current graph-only time as airport-wide taxi
time. Assumptions are exported alongside every value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calibrated_minutes(rows: list[dict], meters_per_unit: float, cruise_speed_mps: float, terminal_service_sec: float) -> float:
    values = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        travel_sec = number(row.get("path_dist")) * meters_per_unit / cruise_speed_mps
        onroute_wait_sec = number(row.get("onroute_wait_sec"))
        entry_wait_sec = number(row.get("wait_time_min")) * 60.0
        values.append((travel_sec + onroute_wait_sec + entry_wait_sec + terminal_service_sec) / 60.0)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detailed ablation metrics.")
    parser.add_argument("--guided-dir", default="results/ablation_seed42/guided_baseline")
    parser.add_argument("--noguide-dir", default="results/ablation_seed42/noguide_baseline")
    parser.add_argument("--pdi-csv", default="results/ablation_seed42/figures/ablation_path_deviation.csv")
    parser.add_argument("--out-dir", default="results/ablation_seed42/figures")
    parser.add_argument("--meters-per-unit", type=float, default=5.833)
    parser.add_argument("--cruise-speed-mps", type=float, default=5.5)
    parser.add_argument("--terminal-service-sec", type=float, default=90.0)
    args = parser.parse_args()

    pdi = {row["method"]: row for row in read_csv(Path(args.pdi_csv))}
    methods = [
        ("MAPPO-Guided", Path(args.guided_dir)),
        ("MAPPO-NoGuide", Path(args.noguide_dir)),
    ]
    output = []
    for label, directory in methods:
        summary = read_json(directory / "MAPPO_summary.json")
        agents = read_csv(directory / "MAPPO_per_agent.csv")
        output.append({
            "method": label,
            "safe_completion_rate_pct": number(summary.get("safe_completion_rate_pct")),
            "completed_flights": int(number(summary.get("success_count"))),
            "network_moving_min": number(summary.get("avg_moving_min")),
            "network_onroute_wait_min": number(summary.get("avg_onroute_wait_min")),
            "entry_queue_wait_min": number(summary.get("avg_wait_min")),
            "network_total_mission_min": number(summary.get("avg_total_time_min")),
            "calibrated_apron_mission_min": calibrated_minutes(
                agents, args.meters_per_unit, args.cruise_speed_mps, args.terminal_service_sec
            ),
            "pdi_sample_n": int(number(pdi[label].get("n"))),
            "pdi_median": number(pdi[label].get("pdi_median")),
            "pdi_iqr": number(pdi[label].get("pdi_iqr")),
            "conflict_protection_events": int(number(summary.get("collision_count"))),
            "stall_recovery_events": int(number(summary.get("deadlock_count"))),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with (out_dir / "ablation_detailed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    guided, noguide = output
    assumptions = (
        f"Calibration assumptions: {args.meters_per_unit:.3f} m per graph unit, "
        f"{args.cruise_speed_mps:.1f} m/s nominal apron cruise speed, and "
        f"{args.terminal_service_sec:.0f} s terminal transition service per mission."
    )
    markdown = "\n".join([
        "| Metric | MAPPO-Guided | MAPPO-NoGuide |",
        "|---|---:|---:|",
        f"| Window-safe service completion | {guided['safe_completion_rate_pct']:.2f}% ({guided['completed_flights']}/363) | {noguide['safe_completion_rate_pct']:.2f}% ({noguide['completed_flights']}/363) |",
        f"| Network moving time | {guided['network_moving_min']:.2f} min | {noguide['network_moving_min']:.2f} min |",
        f"| En-route waiting time | {guided['network_onroute_wait_min']:.2f} min | {noguide['network_onroute_wait_min']:.2f} min |",
        f"| Entry queue waiting time | {guided['entry_queue_wait_min']:.2f} min | {noguide['entry_queue_wait_min']:.2f} min |",
        f"| Network mission time | {guided['network_total_mission_min']:.2f} min | {noguide['network_total_mission_min']:.2f} min |",
        f"| Calibrated apron mission time | {guided['calibrated_apron_mission_min']:.2f} min | {noguide['calibrated_apron_mission_min']:.2f} min |",
        f"| PDI sample size | {guided['pdi_sample_n']} | {noguide['pdi_sample_n']} |",
        f"| Median PDI (IQR) | {guided['pdi_median']:.3f} ({guided['pdi_iqr']:.3f}) | {noguide['pdi_median']:.3f} ({noguide['pdi_iqr']:.3f}) |",
        f"| Conflict-protection triggers | {guided['conflict_protection_events']} | {noguide['conflict_protection_events']} |",
        f"| Stall-recovery triggers | {guided['stall_recovery_events']} | {noguide['stall_recovery_events']} |",
        "",
        assumptions,
        "The calibrated value is a sensitivity-analysis reporting metric; it is not a replacement for an airport-specific empirical calibration.",
    ])
    (out_dir / "ablation_detailed_metrics.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
