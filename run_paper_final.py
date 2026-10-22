"""Run and validate the final TRB experiment suite.

The simulation core remains in run_paper_exp_4_2.py. This wrapper only fixes
the paper-level protocol: one model, one full schedule, one seed, four
scenarios, one result tree, then one plotting pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCENARIOS = ("baseline", "delay", "closure", "priority")
REQUIRED_FILES = (
    "MAPPO_summary.json",
    "FCFS_summary.json",
    "MAPPO_timeseries.csv",
    "FCFS_timeseries.csv",
    "MAPPO_per_agent.csv",
    "FCFS_per_agent.csv",
    "MAPPO_events.json",
    "FCFS_events.json",
)


def validate_result_dir(directory: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (directory / name).exists()]


def run_scenario(args, scenario: str, result_dir: Path) -> None:
    command = [
        sys.executable,
        "run_paper_exp_4_2.py",
        "--config", args.config,
        "--model", args.model,
        "--excel", args.excel,
        "--scenario", scenario,
        "--time-scale", str(args.time_scale),
        "--schedule-compression", str(args.schedule_compression),
        "--start-hour", str(args.start_hour),
        "--end-hour", str(args.end_hour),
        "--active-limit", str(args.active_limit),
        "--entry-headway-steps", str(args.entry_headway_steps),
        "--deadlock-timeout", str(args.deadlock_timeout),
        "--max-steps", str(args.max_steps),
        "--seed", str(args.seed),
        "--result-dir", str(result_dir),
    ]
    print("\n[RUN] " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final TRB experiment suite")
    parser.add_argument("--config", default="configs/fine_tune_8agents.yaml")
    parser.add_argument(
        "--model",
        default="checkpoints/stage3_criticv2_timeout_v3/stage3_8agents_best_eval.pt",
    )
    parser.add_argument("--excel", default="data.xlsx")
    parser.add_argument("--result-root", default="results/paper_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-scale", type=float, default=1.127)
    parser.add_argument("--schedule-compression", type=float, default=0.10)
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--end-hour", type=float, default=24.0)
    parser.add_argument("--active-limit", type=int, default=8)
    parser.add_argument("--entry-headway-steps", type=int, default=5)
    parser.add_argument("--deadlock-timeout", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model checkpoint not found: {model}")

    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "TRB final experiment suite",
        "model": str(model),
        "config": args.config,
        "excel": args.excel,
        "seed": args.seed,
        "time_scale": args.time_scale,
        "schedule_compression": args.schedule_compression,
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "active_limit": args.active_limit,
        "entry_headway_steps": args.entry_headway_steps,
        "deadlock_timeout": args.deadlock_timeout,
        "max_steps": args.max_steps,
        "scenarios": list(SCENARIOS),
    }
    with (root / "paper_final_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.plot_only:
        for scenario in SCENARIOS:
            directory = root / scenario
            missing = validate_result_dir(directory)
            if args.skip_existing and not missing:
                print(f"[SKIP] {scenario}: result contract already complete")
                continue
            run_scenario(args, scenario, directory)

    incomplete = {}
    for scenario in SCENARIOS:
        missing = validate_result_dir(root / scenario)
        if missing:
            incomplete[scenario] = missing
    if incomplete:
        print("[WARN] Incomplete result folders:")
        for scenario, missing in incomplete.items():
            print(f"  {scenario}: {', '.join(missing)}")

    plot_command = [
        sys.executable,
        "plot_trb_results.py",
        "--baseline-dir", str(root / "baseline"),
        "--delay-dir", str(root / "delay"),
        "--closure-dir", str(root / "closure"),
        "--priority-dir", str(root / "priority"),
        "--out-dir", str(root / "figures"),
    ]
    print("\n[PLOT] " + " ".join(plot_command))
    subprocess.run(plot_command, check=True)

    empirical_plot_command = [
        sys.executable,
        "plot_final_paper_figures.py",
        "--result-root", str(root),
    ]
    print("\n[PLOT] " + " ".join(empirical_plot_command))
    subprocess.run(empirical_plot_command, check=True)


if __name__ == "__main__":
    main()
