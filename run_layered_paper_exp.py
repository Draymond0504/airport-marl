"""Run MAPPO and FCFS+A* through the shared layered execution protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from cbs_baseline import build_edge_length_map
from configs.config import Config
from env.marl_env import MARLEnv
from layered_rolling_executor import run_layered_fcfs, run_layered_mappo
from simulate_continuous import load_full_flight_schedule


def _configure_rolling_env(config: Config, args) -> Config:
    """Align policy-time feasibility with the layered execution shield."""
    if not args.disable_execution_mask_alignment:
        config.env.pre_entry_clearance_guard = True
    if args.enable_clearance_feature_alignment:
        config.env.clearance_risk_observation = True
        config.env.clearance_aware_reference_progress = True
    if args.rolling_approach_reservation_progress is not None:
        config.env.approach_reservation_progress = max(
            0.0, min(float(args.rolling_approach_reservation_progress), 0.95)
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--excel", default="data.xlsx")
    parser.add_argument(
        "--allow-noncanonical-schedule", action="store_true",
        help="Allow debug/demo schedules whose final counts are not 271 TARGET + 92 TRANSIT.",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--scenario", choices=["baseline", "delay", "closure", "priority"], default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--schedule-compression", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--active-limit", type=int, default=8)
    parser.add_argument(
        "--operational-timeout-steps", type=int, default=1800,
        help="Active service window before an unfinished flight is marked timeout and its slot is released.",
    )
    parser.add_argument("--entry-headway-steps", type=int, default=5)
    parser.add_argument(
        "--disable-execution-mask-alignment", action="store_true",
        help="Keep the checkpoint/config action mask unchanged for ablations.",
    )
    parser.add_argument(
        "--enable-clearance-feature-alignment", action="store_true",
        help="Expose clearance features that require matching fine-tuning; off by default for frozen actors.",
    )
    parser.add_argument(
        "--rolling-approach-reservation-progress", type=float, default=None,
        help="Override the progress at which a downstream node is reserved.",
    )
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--transit-ratio", type=float, default=0.35)
    parser.add_argument("--start-hour", type=float, default=8.0)
    parser.add_argument("--end-hour", type=float, default=24.0)
    parser.add_argument("--stall-threshold", type=int, default=180)
    parser.add_argument(
        "--stall-recovery-trigger", type=int, default=45,
        help="Consecutive no-progress steps before safe route-follow recovery."
    )
    parser.add_argument(
        "--stall-recovery-cooldown", type=int, default=15,
        help="Minimum steps between route-follow recovery proposals."
    )
    parser.add_argument("--disable-stall-path-recovery", action="store_true")
    parser.add_argument(
        "--disable-vacating-node-entry", action="store_true",
        help="Disable same-cycle entry into a node whose occupant is leaving."
    )
    parser.add_argument("--disable-stall-replanning", action="store_true")
    parser.add_argument("--skip-mappo", action="store_true")
    parser.add_argument("--skip-fcfs", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = _configure_rolling_env(Config.from_yaml(args.config), args)
    config.env.num_agents = args.active_limit
    config.env.enable_deadlock_recovery = False
    config.env.auto_speed = True
    env = MARLEnv(config)
    edge_len_map = build_edge_length_map(env.graph_env)
    flights = load_full_flight_schedule(
        args.excel,
        time_scale=args.time_scale,
        transit_ratio=args.transit_ratio,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )
    if not flights:
        raise RuntimeError("No usable flights were loaded from the schedule.")
    if not all(f.get("start") is not None and f.get("goal") is not None for f in flights):
        raise RuntimeError("Schedule contains an invalid start/goal pair.")

    target_count = sum(
        str(f.get("agent_role", "TARGET")).upper() == "TARGET" for f in flights
    )
    transit_count = sum(
        str(f.get("agent_role", "TARGET")).upper() == "TRANSIT" for f in flights
    )
    if not args.allow_noncanonical_schedule and (target_count, transit_count) != (271, 92):
        raise RuntimeError(
            "Rolling evaluation requires the complete 08:00-24:00 schedule "
            "(271 TARGET + 92 TRANSIT = 363 movements). "
            f"Loaded {target_count} TARGET + {transit_count} TRANSIT "
            f"from {args.excel}. Use data.xlsx or explicitly pass "
            "--allow-noncanonical-schedule for a debug/demo run."
        )

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = vars(args).copy()
    meta.update({
        "flight_count": len(flights),
        "execution_protocol": "layered_proposal_protection_replanning",
        "execution_mask_alignment": not args.disable_execution_mask_alignment,
        "target_movement_count": target_count,
        "transit_movement_count": transit_count,
        "complete_schedule_contract": {
            "expected_target_movement_count": 271,
            "expected_transit_movement_count": 92,
            "expected_total_movement_count": 363,
            "validated": (target_count, transit_count) == (271, 92),
        },
        "pre_entry_clearance_guard": bool(config.env.pre_entry_clearance_guard),
        "clearance_feature_alignment": bool(args.enable_clearance_feature_alignment),
        "clearance_risk_observation": bool(getattr(config.env, "clearance_risk_observation", False)),
        "clearance_aware_reference_progress": bool(
            getattr(config.env, "clearance_aware_reference_progress", False)
        ),
        "rolling_approach_reservation_progress": config.env.approach_reservation_progress,
    })
    (out / "experiment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries = {}
    if not args.skip_mappo:
        summaries["MAPPO"] = run_layered_mappo(
            args, flights, _configure_rolling_env(Config.from_yaml(args.config), args),
            edge_len_map, str(out)
        )
    if not args.skip_fcfs:
        summaries["FCFS+A*"] = run_layered_fcfs(
            args, flights, _configure_rolling_env(Config.from_yaml(args.config), args),
            edge_len_map, str(out)
        )

    # The paper matrix uses event counts for intervention frequency.  In
    # particular, R_stall is not a percentage and excludes initial A* route
    # construction and closure-triggered replanning.
    matrix_fields = [
        "method",
        "overall_task_completion_rate_pct",
        "target_aircraft_completion_rate_pct",
        "target_aircraft_throughput_per_hour",
        "mean_executed_taxi_time_min",
        "conflict_protection_intervention_frequency",
        "R_stall",
        "released_task_count",
        "released_target_task_count",
        "timeout_count",
        "safety_violation_count",
    ]
    matrix_rows = [
        {"method": method, **{field: summary.get(field) for field in matrix_fields if field != "method"}}
        for method, summary in summaries.items()
    ]
    (out / "paper_metrics_matrix.json").write_text(
        json.dumps({"definitions": {
            "overall_task_completion_rate": "completed released tasks / released tasks",
            "target_aircraft_completion_rate": "completed released target tasks / released target tasks",
            "target_aircraft_throughput": "completed target aircraft / simulated hour",
            "mean_executed_taxi_time": "activation to target arrival for completed aircraft",
            "conflict_protection_intervention_frequency": "count of nominal actions replaced by the execution filter",
            "R_stall": "count of replanning events triggered by persistent no-progress stall",
        }, "rows": matrix_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out / "paper_metrics_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_fields)
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"Layered experiment completed: {out}")


if __name__ == "__main__":
    main()
