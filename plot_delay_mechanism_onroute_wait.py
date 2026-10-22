"""Replace delay queue-delay visualisation with on-network waiting response."""

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


def completed_onroute_waits(result_dir: Path, method: str) -> np.ndarray:
    waits = []
    for row in read_csv(result_dir / f"{method}_per_agent.csv"):
        if row["status"] != "completed" or not row.get("onroute_wait_sec"):
            continue
        waits.append(float(row["onroute_wait_sec"]) / 60.0)
    return np.asarray(waits, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(result_dir / "experiment_meta.json")
    scale = float(meta["time_scale"])
    event_data = read_json(result_dir / "MAPPO_events.json")["events"]
    event = next(item for item in event_data if item.get("type") == "delay_interruption")
    start_min = float(event["step"]) * scale / 60.0
    end_min = (float(event["step"]) + float(event["duration"])) * scale / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.12, 1.0]})
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
    ax.grid(axis="both", color="#DDE3E8", linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    series = [completed_onroute_waits(result_dir, method) for method in ("MAPPO", "FCFS")]
    box = ax.boxplot(series, patch_artist=True, widths=0.52, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "#1F2937", "markeredgecolor": "#1F2937", "markersize": 4})
    for patch, method in zip(box["boxes"], ("MAPPO", "FCFS")):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.66)
    ax.set_xticklabels([LABELS[method] for method in ("MAPPO", "FCFS")])
    ax.set_title("On-network waiting during delay scenario")
    ax.set_ylabel("On-network waiting time (min)")
    ax.grid(axis="y", color="#DDE3E8", linestyle="--", linewidth=0.6)
    ax.scatter([], [], marker="D", color="#1F2937", s=18, label="Mean")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle("Delay mechanism: interruption, recovery, and local waiting", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_delay_mechanism.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_delay_onroute_wait.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for method, values in zip(("MAPPO", "FCFS"), series):
        rows.append({
            "method": LABELS[method],
            "completed_sample_n": len(values),
            "mean_onroute_wait_min": round(float(np.mean(values)), 3),
            "median_onroute_wait_min": round(float(np.median(values)), 3),
            "p75_onroute_wait_min": round(float(np.percentile(values, 75)), 3),
        })
    with (out_dir / "delay_onroute_wait_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote on-network delay mechanism figure to {out_dir}")


if __name__ == "__main__":
    main()
