"""Train/evaluate the paper's A* guidance ablation.

The guided baseline is expected at results/paper_final/baseline. The script
trains only the NoGuide curriculum and evaluates its 8-agent best checkpoint
on the same full baseline schedule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command):
    print("\n[RUN] " + " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the NoGuide paper ablation")
    parser.add_argument("--guided-result", default="results/paper_final/baseline")
    parser.add_argument("--save-dir", default="checkpoints/paper_ablation_no_guide")
    parser.add_argument("--result-dir", default="results/paper_final/ablation/no_guide/baseline")
    parser.add_argument("--config", default="configs/ablation_no_guide_stable.yaml")
    parser.add_argument("--excel", default="data.xlsx")
    parser.add_argument("--episodes-per-stage", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-scale", type=float, default=1.127)
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--end-hour", type=float, default=24.0)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--active-limit", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    guided_summary = Path(args.guided_result) / "MAPPO_summary.json"
    if not guided_summary.exists():
        raise SystemExit(f"Guided baseline is missing: {guided_summary}")

    save_dir = Path(args.save_dir)
    if not args.skip_training:
        run([
            sys.executable, "train_curriculum.py",
            "--config", args.config,
            "--stages", "2,4,8",
            "--episodes-per-stage", str(args.episodes_per_stage),
            "--save-dir", str(save_dir),
            "--seed", str(args.seed),
            "--final-safe-rate", "80",
            "--final-full-success-rate", "80",
        ])

    model = save_dir / "stage3_8agents_best_eval.pt"
    if not model.exists():
        model = save_dir / "stage3_8agents_latest.pt"
    if not model.exists():
        raise SystemExit(f"No NoGuide 8-agent checkpoint found in {save_dir}")

    result_dir = Path(args.result_dir)
    run([
        sys.executable, "run_paper_exp_4_2.py",
        "--config", args.config,
        "--model", str(model),
        "--excel", args.excel,
        "--scenario", "baseline",
        "--time-scale", str(args.time_scale),
        "--start-hour", str(args.start_hour),
        "--end-hour", str(args.end_hour),
        "--active-limit", str(args.active_limit),
        "--max-steps", str(args.max_steps),
        "--seed", str(args.seed),
        "--result-dir", str(result_dir),
        "--skip-fcfs",
    ])

    no_guide_summary = result_dir / "MAPPO_summary.json"
    with (result_dir.parent / "ablation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "guided_result": str(Path(args.guided_result)),
            "no_guide_result": str(result_dir),
            "no_guide_model": str(model),
            "seed": args.seed,
            "comparison": "MAPPO-Guided vs MAPPO-NoGuide",
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] NoGuide result: {no_guide_summary}")
    print(f"[OK] Manifest: {result_dir.parent / 'ablation_manifest.json'}")


if __name__ == "__main__":
    main()
