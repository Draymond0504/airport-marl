"""Generate explicitly non-empirical figures for internal paper planning.

The outputs are visual storyboards only. They are deliberately isolated from
the empirical result folders and watermarked so they cannot be confused with
measured MAPPO or FCFS results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATUS = "ILLUSTRATIVE INTERNAL REFERENCE ONLY - NOT EMPIRICAL RESULTS"
COLORS = {"MAPPO-Guided": "#2d7f74", "FCFS+A*": "#b05b3b", "MAPPO-NoGuide": "#6c7a89"}


def mark_internal(fig: plt.Figure) -> None:
    fig.text(
        0.5, 0.012, STATUS, ha="center", va="bottom", fontsize=8,
        color="#8d3030", weight="bold",
    )


def save_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def convergence(output_dir: Path, rng: np.random.Generator) -> None:
    episodes = np.arange(0, 10001, 100)
    safe = 42 + 47 * (1 - np.exp(-episodes / 2200))
    full = 5 + 79 * (1 - np.exp(-episodes / 2850))
    safe += rng.normal(0, 1.4, len(episodes))
    full += rng.normal(0, 1.8, len(episodes))
    safe = np.clip(safe, 0, 91)
    full = np.clip(full, 0, 85)
    rows = [
        {"episode": int(ep), "safe_completion_pct": round(float(s), 3), "full_success_pct": round(float(f), 3)}
        for ep, s, f in zip(episodes, safe, full)
    ]
    save_csv(output_dir / "illustrative_convergence.csv", rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(episodes, safe, lw=2.2, color=COLORS["MAPPO-Guided"], label="Safe completion")
    ax.plot(episodes, full, lw=2.2, color="#496f9e", label="Full success")
    ax.axhline(80, color="#7d7d7d", ls="--", lw=1, label="Internal target")
    ax.set(xlabel="Training episodes", ylabel="Rate (%)", ylim=(0, 100), title="Illustrative deterministic evaluation trajectory")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    mark_internal(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "fig_4_1_illustrative_convergence.png", dpi=220)
    plt.close(fig)


def baseline_completion(output_dir: Path) -> None:
    minutes = np.arange(0, 121, 2)
    fcfs = np.minimum(44, np.maximum(0, 0.46 * (minutes - 12)))
    mappo = np.minimum(46, np.maximum(0, 0.38 * (minutes - 12) + 0.09 * np.maximum(minutes - 55, 0)))
    rows = []
    for minute, m, f in zip(minutes, mappo, fcfs):
        rows.extend([
            {"minute": int(minute), "method": "MAPPO-Guided", "cumulative_safe_completed": round(float(m), 3)},
            {"minute": int(minute), "method": "FCFS+A*", "cumulative_safe_completed": round(float(f), 3)},
        ])
    save_csv(output_dir / "illustrative_baseline_timeseries.csv", rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(minutes, mappo, lw=2.4, color=COLORS["MAPPO-Guided"], label="MAPPO-Guided")
    ax.plot(minutes, fcfs, lw=2.2, color=COLORS["FCFS+A*"], label="FCFS+A*")
    ax.set(xlabel="Simulation time (min)", ylabel="Cumulative safe completions", title="Illustrative baseline completion process")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    mark_internal(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "fig_4_2_illustrative_baseline_completion.png", dpi=220)
    plt.close(fig)


def perturbation_outcomes(output_dir: Path) -> None:
    rows = [
        {"scenario": "Baseline", "method": "MAPPO-Guided", "safe_completion": 90.5, "full_success": 83.5, "recovery_time": 0.0},
        {"scenario": "Baseline", "method": "FCFS+A*", "safe_completion": 88.5, "full_success": 78.5, "recovery_time": 0.0},
        {"scenario": "Delay", "method": "MAPPO-Guided", "safe_completion": 87.5, "full_success": 77.0, "recovery_time": 13.0},
        {"scenario": "Delay", "method": "FCFS+A*", "safe_completion": 81.5, "full_success": 64.0, "recovery_time": 23.0},
        {"scenario": "Closure", "method": "MAPPO-Guided", "safe_completion": 84.5, "full_success": 71.0, "recovery_time": 18.0},
        {"scenario": "Closure", "method": "FCFS+A*", "safe_completion": 77.0, "full_success": 55.0, "recovery_time": 32.0},
        {"scenario": "Priority", "method": "MAPPO-Guided", "safe_completion": 86.5, "full_success": 75.0, "recovery_time": 15.0},
        {"scenario": "Priority", "method": "FCFS+A*", "safe_completion": 80.0, "full_success": 59.0, "recovery_time": 26.0},
    ]
    save_csv(output_dir / "illustrative_perturbation_metrics.csv", rows)
    scenarios = ["Delay", "Closure", "Priority"]
    x = np.arange(len(scenarios))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0))
    for ax, key, title, unit in zip(
        axes,
        ["safe_completion", "full_success", "recovery_time"],
        ["Safe completion", "Full success", "Recovery time"],
        ["%", "%", "min"],
    ):
        mappo = [next(row[key] for row in rows if row["scenario"] == s and row["method"] == "MAPPO-Guided") for s in scenarios]
        fcfs = [next(row[key] for row in rows if row["scenario"] == s and row["method"] == "FCFS+A*") for s in scenarios]
        ax.bar(x - width / 2, mappo, width, color=COLORS["MAPPO-Guided"], label="MAPPO-Guided")
        ax.bar(x + width / 2, fcfs, width, color=COLORS["FCFS+A*"], label="FCFS+A*")
        ax.set_xticks(x, scenarios)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    mark_internal(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "fig_4_3_illustrative_perturbation_outcomes.png", dpi=220)
    plt.close(fig)


def delay_mechanism(output_dir: Path, rng: np.random.Generator) -> None:
    time = np.arange(-40, 81)
    base = 18 + 0.34 * (time + 40)
    mappo = np.where(time < 0, base, base + 0.52 * time - 0.08 * np.maximum(time - 45, 0))
    fcfs = np.where(time < 0, base, base + 0.25 * time)
    wait_mappo = np.maximum(0.2, rng.gamma(1.8, 1.7, 160))
    wait_fcfs = np.maximum(0.2, rng.gamma(2.6, 2.3, 160))
    rows = [{"relative_time": int(t), "method": "MAPPO-Guided", "cumulative_safe_completed": round(float(v), 3)} for t, v in zip(time, mappo)]
    rows += [{"relative_time": int(t), "method": "FCFS+A*", "cumulative_safe_completed": round(float(v), 3)} for t, v in zip(time, fcfs)]
    save_csv(output_dir / "illustrative_delay_response.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].plot(time, mappo, lw=2.3, color=COLORS["MAPPO-Guided"], label="MAPPO-Guided")
    axes[0].plot(time, fcfs, lw=2.2, color=COLORS["FCFS+A*"], label="FCFS+A*")
    axes[0].axvline(0, color="#555555", ls="--", lw=1)
    axes[0].set(xlabel="Minutes from delay release", ylabel="Cumulative safe completions", title="Release recovery")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].boxplot([wait_mappo, wait_fcfs], tick_labels=["MAPPO", "FCFS"], patch_artist=True,
                    boxprops={"facecolor": "#9bc8c0"}, medianprops={"color": "#203030"})
    axes[1].set(ylabel="En-route waiting time (min)", title="Waiting distribution")
    axes[1].grid(axis="y", alpha=0.2)
    mark_internal(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "fig_4_4_illustrative_delay_recovery.png", dpi=220)
    plt.close(fig)


def closure_mechanism(output_dir: Path) -> None:
    time = np.arange(-30, 61)
    closed_mappo = np.where(time < 0, 4.0, np.maximum(0, 4.0 - 0.55 * time))
    bypass_mappo = np.where(time < 0, 1.0, np.minimum(4.0, 1.0 + 0.12 * time))
    closed_fcfs = np.where(time < 0, 4.0, np.maximum(0, 4.0 - 0.18 * time))
    bypass_fcfs = np.where(time < 0, 1.0, np.minimum(3.0, 1.0 + 0.05 * time))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, closed, bypass, name in [
        (axes[0], closed_mappo, bypass_mappo, "MAPPO-Guided"),
        (axes[1], closed_fcfs, bypass_fcfs, "FCFS+A*"),
    ]:
        ax.plot(time, closed, lw=2.2, color="#b05b3b", label="Closed-corridor flow")
        ax.plot(time, bypass, lw=2.2, color="#2d7f74", label="Bypass-corridor flow")
        ax.axvline(0, color="#555555", ls="--", lw=1)
        ax.set(title=name, xlabel="Minutes from closure", ylabel="Active aircraft")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    mark_internal(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "fig_4_5_illustrative_closure_response.png", dpi=220)
    plt.close(fig)


def priority_mechanism(output_dir: Path) -> None:
    time = np.arange(-20, 61)
    mappo_vip = np.where(time < 0, 0.72, 0.72 + 0.42 * (1 - np.exp(-time / 7)))
    fcfs_vip = np.where(time < 0, 0.72, 0.72 + 0.28 * (1 - np.exp(-time / 20)))
    mappo_regular = np.where(time < 0, 0.78, 0.78 - 0.10 * np.exp(-time / 15))
    fcfs_regular = np.where(time < 0, 0.78, 0.78 - 0.20 * np.exp(-time / 20))
    rows = []
    for method, vip, regular in [("MAPPO-Guided", mappo_vip, mappo_regular), ("FCFS+A*", fcfs_vip, fcfs_regular)]:
        for t, v, r in zip(time, vip, regular):
            rows.append({"relative_time": int(t), "method": method, "vip_speed": round(float(v), 3), "regular_speed": round(float(r), 3)})
    save_csv(output_dir / "illustrative_priority_response.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, vip, regular, name in [
        (axes[0], mappo_vip, mappo_regular, "MAPPO-Guided"),
        (axes[1], fcfs_vip, fcfs_regular, "FCFS+A*"),
    ]:
        ax.plot(time, vip, color="#b05b3b", lw=2.3, label="High-priority aircraft")
        ax.plot(time, regular, color="#496f9e", lw=2.1, label="Regular aircraft mean")
        ax.axvline(0, color="#555555", ls="--", lw=1)
        ax.set(title=name, xlabel="Minutes from priority boost", ylabel="Normalized taxi speed", ylim=(0.45, 1.25))
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    mark_internal(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "fig_4_6_illustrative_priority_response.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create internal illustrative paper figures")
    parser.add_argument("--output-dir", default="results/internal_reference_mockup")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    convergence(output_dir, rng)
    baseline_completion(output_dir)
    perturbation_outcomes(output_dir)
    delay_mechanism(output_dir, rng)
    closure_mechanism(output_dir)
    priority_mechanism(output_dir)
    manifest = {
        "status": STATUS,
        "purpose": "Internal planning and figure-layout review only.",
        "prohibited_use": "Do not report these values as measured experiments or include them in a submission.",
        "seed": args.seed,
        "figures": sorted(path.name for path in output_dir.glob("fig_*.png")),
    }
    (output_dir / "README_INTERNAL_ONLY.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Internal illustrative figures written to: {output_dir}")


if __name__ == "__main__":
    main()
