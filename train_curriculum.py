"""Curriculum training for airport MAPPO.

Default schedule: 2 agents -> 4 agents -> 8 agents.
Use the stable no-perturbation config first to obtain a clean convergence curve.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from configs.config import Config
from train_improved import train_improved


DEFAULT_STAGES = [
    {
        "agents": 2, "max_steps": 300, "entropy": 0.10,
        "lr_actor": 5.0e-5, "lr_critic": 2.0e-5,
        "handoff_safe_rate": 90.0, "handoff_full_success_rate": 60.0,
    },
    {
        "agents": 4, "max_steps": 400, "entropy": 0.08,
        "lr_actor": 4.0e-5, "lr_critic": 2.0e-5,
        "handoff_safe_rate": 80.0, "handoff_full_success_rate": 40.0,
    },
    {
        "agents": 8, "max_steps": 500, "entropy": 0.08,
        "lr_actor": 3.0e-5, "lr_critic": 1.0e-5,
        "handoff_safe_rate": 0.0, "handoff_full_success_rate": 0.0,
    },
]


def parse_stages(raw: str) -> list[int]:
    stages = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not stages:
        raise ValueError("At least one curriculum stage is required.")
    return stages


def make_stage_config(
    base: Config,
    stage: dict,
    episodes: int,
    use_config_training: bool = False,
) -> Config:
    cfg = deepcopy(base)
    cfg.env.num_agents = stage["agents"]
    cfg.env.max_steps = stage["max_steps"]
    cfg.env.taxiway_close_prob = 0.0
    cfg.env.priority_change_prob = 0.0
    cfg.env.delay_prob = 0.0
    # Preserve the base configuration so the NoGuide ablation remains a true
    # isolation of the A* observation features across all curriculum stages.
    cfg.env.use_astar_guide = getattr(base.env, "use_astar_guide", True)
    cfg.env.midroute_start_prob = 0.0
    cfg.env.random_flow_mix = False
    cfg.env.entry_stagger_steps = 40

    cfg.training.total_episodes = episodes
    if not use_config_training:
        cfg.training.entropy_coef = stage["entropy"]
        cfg.training.lr_actor = stage["lr_actor"]
        cfg.training.lr_critic = stage["lr_critic"]
        cfg.training.enable_lr_decay = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stable curriculum training.")
    parser.add_argument("--config", default="configs/improved_stable.yaml", help="Base config path.")
    parser.add_argument("--episodes-per-stage", type=int, default=3000, help="Episodes per stage.")
    parser.add_argument("--save-dir", default="checkpoints/curriculum_stable", help="Checkpoint/log directory.")
    parser.add_argument("--stages", default="2,4,8", help="Comma-separated agent counts.")
    parser.add_argument("--start-stage", type=int, default=0, help="0-based stage index to start from.")
    parser.add_argument("--resume", default=None, help="Checkpoint to load for the first active stage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-config-training",
        action="store_true",
        help="Use optimizer/PPO settings from --config instead of curriculum defaults; intended for same-stage fine-tuning.",
    )
    parser.add_argument(
        "--strict-handoff",
        action="store_true",
        help="Stop before the next stage if deterministic evaluation misses the handoff thresholds.",
    )
    parser.add_argument(
        "--final-safe-rate",
        type=float,
        default=80.0,
        help="Final-stage diagnostic threshold for safe completion rate (percent).",
    )
    parser.add_argument(
        "--final-full-success-rate",
        type=float,
        default=80.0,
        help="Final-stage diagnostic threshold for full-episode success rate (percent).",
    )
    parser.add_argument(
        "--strict-final",
        action="store_true",
        help="Return failure status when the final-stage best evaluation misses its thresholds.",
    )
    args = parser.parse_args()

    requested_agents = parse_stages(args.stages)
    stage_defs_by_agents = {stage["agents"]: stage for stage in DEFAULT_STAGES}
    stages = []
    for agents in requested_agents:
        if agents not in stage_defs_by_agents:
            raise ValueError(f"Unsupported stage agent count: {agents}. Supported: 2,4,8")
        stages.append(stage_defs_by_agents[agents])

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_config = Config.from_yaml(args.config)

    previous_checkpoint = args.resume
    if args.start_stage > 0 and previous_checkpoint is None:
        prev_stage = stages[args.start_stage - 1]
        prev_prefix = f"stage{args.start_stage}_{prev_stage['agents']}agents"
        prev_best = save_dir / f"{prev_prefix}_best_eval.pt"
        previous_checkpoint = str(
            prev_best if prev_best.exists() else save_dir / f"{prev_prefix}_latest.pt"
        )

    print("=" * 70)
    print("Stable Curriculum Training")
    print(f"Stages: {[stage['agents'] for stage in stages]} agents")
    print(f"Episodes per stage: {args.episodes_per_stage}")
    print(f"Save dir: {save_dir}")
    print("=" * 70)

    for idx in range(args.start_stage, len(stages)):
        stage = stages[idx]
        prefix = f"stage{idx + 1}_{stage['agents']}agents"
        cfg = make_stage_config(
            base_config,
            stage,
            args.episodes_per_stage,
            use_config_training=args.use_config_training,
        )

        print("\n" + "=" * 70)
        print(
            f"Stage {idx + 1}/{len(stages)}: {stage['agents']} agents, "
            f"max_steps={stage['max_steps']}, entropy={cfg.training.entropy_coef}"
        )
        if previous_checkpoint:
            print(f"Loading actor from: {previous_checkpoint}")
        print("=" * 70)

        # The centralized critic dimensions change between curriculum stages,
        # so a handoff loads only the actor. A same-stage continuation should
        # preserve critic and optimizer state to avoid a fresh value-function
        # warm-up and the resulting reward oscillation.
        same_stage_resume = bool(
            previous_checkpoint
            and prefix in Path(previous_checkpoint).stem
        )
        if previous_checkpoint:
            print(
                "Resume mode: "
                + (
                    "same-stage (load actor, critic, optimizer)"
                    if same_stage_resume
                    else "curriculum handoff (load actor only)"
                )
            )

        train_result = train_improved(
            config_path=None,
            resume=previous_checkpoint,
            override_config=cfg,
            load_critic=same_stage_resume,
            load_optimizer=same_stage_resume,
            save_prefix=prefix,
            seed=args.seed,
            save_dir_override=str(save_dir),
        )

        best_checkpoint = train_result.get("best_eval_path")
        previous_checkpoint = str(best_checkpoint or (save_dir / f"{prefix}_latest.pt"))

        best_eval = train_result.get("best_eval") or {}
        if idx < len(stages) - 1:
            safe_rate = float(best_eval.get("eval_safe_completion_rate", 0.0))
            full_rate = float(best_eval.get("eval_full_success_rate", 0.0))
            min_safe = float(stage.get("handoff_safe_rate", 0.0))
            min_full = float(stage.get("handoff_full_success_rate", 0.0))
            passed = safe_rate >= min_safe and full_rate >= min_full
            print(
                f"[Handoff] Stage {stage['agents']} agents: "
                f"safe={safe_rate:.1f}% (min {min_safe:.1f}%), "
                f"full={full_rate:.1f}% (min {min_full:.1f}%) -> "
                f"{'PASS' if passed else 'WARN'}"
            )
            if not passed and args.strict_handoff:
                raise SystemExit(
                    f"Stage {stage['agents']} did not meet the handoff criteria; "
                    "strict curriculum stopped."
                )
        else:
            safe_rate = float(best_eval.get("eval_safe_completion_rate", 0.0))
            full_rate = float(best_eval.get("eval_full_success_rate", 0.0))
            min_safe = args.final_safe_rate
            min_full = args.final_full_success_rate
            passed = safe_rate >= min_safe and full_rate >= min_full
            print(
                f"[Final diagnostic] Stage {stage['agents']} agents: "
                f"safe={safe_rate:.1f}% (target {min_safe:.1f}%), "
                f"full={full_rate:.1f}% (target {min_full:.1f}%) -> "
                f"{'PASS' if passed else 'FAIL'}"
            )
            if not passed and args.strict_final:
                raise SystemExit(
                    f"Final stage did not meet the diagnostic targets after "
                    f"{args.episodes_per_stage} episodes."
                )

    print("\nCurriculum training completed.")


if __name__ == "__main__":
    main()
