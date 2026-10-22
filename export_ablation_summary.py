"""Export a reproducible Guided vs NoGuide ablation summary table."""

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


def first_threshold_episode(path: Path, threshold: float) -> int | None:
    for row in read_csv(path):
        if float(row["eval_safe_completion_rate"]) >= threshold:
            return int(float(row["episode"]))
    return None


def pdi_by_method(path: Path) -> dict[str, dict]:
    return {row["method"]: row for row in read_csv(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export filled ablation metrics for Chapter 4.")
    parser.add_argument("--guided-dir", default="results/ablation_seed42/guided_baseline")
    parser.add_argument("--noguide-dir", default="results/ablation_seed42/noguide_baseline")
    parser.add_argument("--guided-eval", default="checkpoints/ablation_guided_seed42/evaluation_log.csv")
    parser.add_argument("--noguide-eval", default="checkpoints/ablation_noguide_seed42/evaluation_log.csv")
    parser.add_argument("--pdi-csv", default="results/ablation_seed42/figures/ablation_path_deviation.csv")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--out-dir", default="results/ablation_seed42/figures")
    args = parser.parse_args()

    guided = read_json(Path(args.guided_dir) / "MAPPO_summary.json")
    noguide = read_json(Path(args.noguide_dir) / "MAPPO_summary.json")
    pdi = pdi_by_method(Path(args.pdi_csv))
    guided_episode = first_threshold_episode(Path(args.guided_eval), args.threshold)
    noguide_episode = first_threshold_episode(Path(args.noguide_eval), args.threshold)

    methods = [
        ("MAPPO-Guided", guided, guided_episode),
        ("MAPPO-NoGuide", noguide, noguide_episode),
    ]
    rows = []
    for label, summary, episode in methods:
        pdi_row = pdi[label]
        rows.append({
            "method": label,
            "safe_completion_rate_pct": round(float(summary["safe_completion_rate_pct"]), 2),
            "average_mission_time_min": round(float(summary["avg_total_time_min"]), 2),
            "path_deviation_median": round(float(pdi_row["pdi_median"]), 5),
            "path_deviation_iqr": round(float(pdi_row["pdi_iqr"]), 5),
            "conflict_protection_events": int(summary["collision_count"]),
            "stall_recovery_events": int(summary["deadlock_count"]),
            "convergence_episode": episode if episode is not None else f"> {max(int(float(row['episode'])) for row in read_csv(Path(args.noguide_eval if label.endswith('NoGuide') else args.guided_eval))}",
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    guided_row, noguide_row = rows
    markdown = "\n".join([
        "| Metric | MAPPO-Guided | MAPPO-NoGuide |",
        "|---|---:|---:|",
        f"| Safe completion rate | {guided_row['safe_completion_rate_pct']:.2f}% | {noguide_row['safe_completion_rate_pct']:.2f}% |",
        f"| Average mission time | {guided_row['average_mission_time_min']:.2f} min | {noguide_row['average_mission_time_min']:.2f} min |",
        f"| Path deviation rate (median PDI) | {guided_row['path_deviation_median']:.3f} | {noguide_row['path_deviation_median']:.3f} |",
        f"| Conflict-protection events | {guided_row['conflict_protection_events']} | {noguide_row['conflict_protection_events']} |",
        f"| Training convergence episode ({args.threshold:.0f}% threshold) | {guided_row['convergence_episode']} | {noguide_row['convergence_episode']} |",
        "",
        f"PDI IQR: Guided={guided_row['path_deviation_iqr']:.3f}; NoGuide={noguide_row['path_deviation_iqr']:.3f}.",
        f"Stall-recovery triggers: Guided={guided_row['stall_recovery_events']}; NoGuide={noguide_row['stall_recovery_events']}.",
    ])
    md_path = out_dir / "ablation_summary.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\n[OK] {csv_path}")
    print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
