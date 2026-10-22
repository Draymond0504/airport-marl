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


def convergence_episode(rows: list[dict], threshold: float) -> int | None:
    for row in rows:
        if float(row["eval_safe_completion_rate"]) >= threshold:
            return int(float(row["episode"]))
    return None


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

    summaries = {
        "MAPPO-Guided": read_json(Path(args.guided_dir) / "MAPPO_summary.json"),
        "MAPPO-NoGuide": read_json(Path(args.noguide_dir) / "MAPPO_summary.json"),
    }
    pdi = {row["method"]: row for row in read_csv(Path(args.pdi_csv))}
    eval_rows = {
        "MAPPO-Guided": read_csv(Path(args.guided_eval)),
        "MAPPO-NoGuide": read_csv(Path(args.noguide_eval)),
    }

    rows = []
    for label in ("MAPPO-Guided", "MAPPO-NoGuide"):
        summary = summaries[label]
        eval_log = eval_rows[label]
        episode = convergence_episode(eval_log, args.threshold)
        last_episode = max(int(float(row["episode"])) for row in eval_log)
        pdi_row = pdi[label]
        rows.append({
            "method": label,
            "safe_completion_rate_pct": round(float(summary["safe_completion_rate_pct"]), 2),
            "average_mission_time_min": round(float(summary["avg_total_time_min"]), 2),
            "path_deviation_median": round(float(pdi_row["pdi_median"]), 5),
            "path_deviation_iqr": round(float(pdi_row["pdi_iqr"]), 5),
            "conflict_protection_events": int(summary["collision_count"]),
            "stall_recovery_events": int(summary["deadlock_count"]),
            "convergence_episode": episode if episode is not None else f"> {last_episode}",
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    guided, noguide = rows
    markdown = "\n".join([
        "| Metric | MAPPO-Guided | MAPPO-NoGuide |",
        "|---|---:|---:|",
        f"| Safe completion rate | {guided['safe_completion_rate_pct']:.2f}% | {noguide['safe_completion_rate_pct']:.2f}% |",
        f"| Average mission time | {guided['average_mission_time_min']:.2f} min | {noguide['average_mission_time_min']:.2f} min |",
        f"| Path deviation rate (median PDI) | {guided['path_deviation_median']:.3f} | {noguide['path_deviation_median']:.3f} |",
        f"| Conflict-protection events | {guided['conflict_protection_events']} | {noguide['conflict_protection_events']} |",
        f"| Training convergence episode ({args.threshold:.0f}% threshold) | {guided['convergence_episode']} | {noguide['convergence_episode']} |",
        "",
        f"PDI IQR: Guided={guided['path_deviation_iqr']:.3f}; NoGuide={noguide['path_deviation_iqr']:.3f}.",
        f"Stall-recovery triggers: Guided={guided['stall_recovery_events']}; NoGuide={noguide['stall_recovery_events']}.",
    ])
    (out_dir / "ablation_summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
