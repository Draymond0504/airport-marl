"""Paper delay mechanism figure using an on-network waiting tail curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"MAPPO": "#007C91", "FCFS": "#C25B35"}
LABELS = {"MAPPO": "MAPPO-Guided", "FCFS": "FCFS+A*"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def waits(result_dir: Path, method: str) -> np.ndarray:
    return np.asarray([
        float(row["onroute_wait_sec"]) / 60.0
        for row in read_csv(result_dir / f"{method}_per_agent.csv")
        if row["status"] == "completed" and row.get("onroute_wait_sec")
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result_dir, out_dir = Path(args.result_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(result_dir / "experiment_meta.json")
    event = next(item for item in read_json(result_dir / "MAPPO_events.json")["events"] if item.get("type") == "delay_interruption")
    start_min = float(event["step"]) * float(meta["time_scale"]) / 60.0
    end_min = start_min + float(event["duration"]) * float(meta["time_scale"]) / 60.0
    values = {method: waits(result_dir, method) for method in ("MAPPO", "FCFS")}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1.12, 1.0]})
    ax = axes[0]
    ax.axvspan(start_min, end_min, color="#7B8794", alpha=0.18, zorder=0)
    ax.axvline(start_min, color="#475569", linestyle="--", linewidth=0.9)
    ax.text(start_min, 99, "Admission interruption", rotation=90, va="top", ha="right", fontsize=8, color="#475569")
    for method in ("MAPPO", "FCFS"):
        summary = read_json(result_dir / f"{method}_summary.json")
        rows = read_csv(result_dir / f"{method}_timeseries.csv")
        x = [float(row["time_min"]) for row in rows]
        y = [100.0 * float(row["cumulative_success"]) / float(summary["total_count"]) for row in rows]
        ax.plot(x, y, color=COLORS[method], linewidth=2.0, label=LABELS[method])
        ax.scatter(x[-1], y[-1], color=COLORS[method], s=20, zorder=3)
    ax.set_title("Mid-service interruption and recovery")
    ax.set_xlabel("Simulation time (min)")
    ax.set_ylabel("Cumulative window-safe service (%)")
    ax.set_ylim(0, 105)
    ax.grid(color="#DDE3E8", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    max_wait = max(float(series.max()) for series in values.values())
    thresholds = np.linspace(0.0, max_wait, 401)
    for method, series in values.items():
        exceedance = np.asarray([(series > threshold).mean() * 100.0 for threshold in thresholds])
        ax.step(thresholds, exceedance, where="post", color=COLORS[method], linewidth=2.0, label=LABELS[method])
    for threshold in (1.0, 5.0):
        ax.axvline(threshold, color="#94A3B8", linestyle=":", linewidth=0.9)
        labels = []
        for method, series in values.items():
            share = (series > threshold).mean() * 100.0
            labels.append(f"{LABELS[method].split('-')[0]} {share:.1f}%")
        ax.text(threshold + 0.18, 93 if threshold == 1.0 else 83, "\n".join(labels), fontsize=7.5, color="#475569", va="top")
    ax.set_title("Tail probability of on-network waiting")
    ax.set_xlabel("Waiting-time threshold (min)")
    ax.set_ylabel("Flights exceeding threshold (%)")
    ax.set_xlim(0, max_wait)
    ax.set_ylim(0, 102)
    ax.grid(color="#DDE3E8", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Delay mechanism: interruption, recovery, and local waiting", fontsize=14, y=1.01)
    fig.tight_layout()
    for filename in ("fig_delay_mechanism.png", "fig_delay_onroute_wait_tail.png"):
        fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for method, series in values.items():
        summary_rows.append({
            "method": LABELS[method],
            "completed_sample_n": len(series),
            "share_wait_gt_1min_pct": round(float((series > 1.0).mean() * 100.0), 2),
            "share_wait_gt_5min_pct": round(float((series > 5.0).mean() * 100.0), 2),
            "p90_wait_min": round(float(np.percentile(series, 90)), 3),
            "p95_wait_min": round(float(np.percentile(series, 95)), 3),
        })
    with (out_dir / "delay_wait_tail_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] Wrote delay waiting-tail figure to {out_dir}")


if __name__ == "__main__":
    main()
