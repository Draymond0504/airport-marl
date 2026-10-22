"""Publication-oriented plots for the airport MARL experiments.

The script consumes the result contract written by run_paper_exp_4_2.py:
  *_summary.json, *_timeseries.csv, *_per_agent.csv
and optionally *_events.json and *_micro_priority.json.

It deliberately derives the denominator from each summary instead of using a
hard-coded flight count.  The main completion metric is safe completion
(completion_rate_pct); eventual service is shown separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))


METHODS = {
    "MAPPO": {"color": "#1F77B4", "label": "MAPPO"},
    "FCFS": {"color": "#D95F02", "label": "FCFS+A*"},
}
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "delay": "Delay",
    "closure": "Closure",
    "priority": "Priority",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return None


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
        return []


def number(value, default=None):
    if value in (None, "", "None", "N/A"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value, default=None):
    result = number(value, default)
    return int(result) if result is not None else default


def is_true(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def derive_summary(summary: dict | None, agents: list[dict]) -> dict:
    """Normalize current and legacy tracker output to one metric schema."""
    summary = dict(summary or {})
    total = integer(summary.get("total_count"), None)
    if total is None or total <= 0:
        total = len(agents)

    finished = [r for r in agents if integer(r.get("finish_step"), None) is not None]
    safe = [r for r in finished if not is_true(r.get("crashed", False))]
    if "success_count" not in summary:
        summary["success_count"] = len(safe)
    if "eventual_success_count" not in summary:
        summary["eventual_success_count"] = len(safe)
    if "completion_rate_pct" not in summary:
        summary["completion_rate_pct"] = 100.0 * summary["success_count"] / total if total else 0.0
    if "eventual_completion_rate_pct" not in summary:
        summary["eventual_completion_rate_pct"] = (
            100.0 * summary["eventual_success_count"] / total if total else 0.0
        )
    for key in ("collision_count", "deadlock_count"):
        summary[key] = number(summary.get(key), 0.0)
    summary["total_count"] = total
    summary["safe_rate"] = number(summary.get("completion_rate_pct"), 0.0)
    summary["eventual_rate"] = number(summary.get("eventual_completion_rate_pct"), 0.0)
    summary["deadlock_per_100"] = 100.0 * summary["deadlock_count"] / total if total else 0.0
    summary["collision_per_100"] = 100.0 * summary["collision_count"] / total if total else 0.0
    summary["timeout_rate"] = 100.0 * number(summary.get("timeout_count"), 0.0) / total if total else 0.0
    total_time = number(summary.get("avg_total_time_min"), None)
    if total_time is None:
        taxi = number(summary.get("avg_taxi_min"), 0.0)
        wait = number(summary.get("avg_wait_min"), 0.0)
        total_time = taxi + wait
    summary["avg_total_time"] = total_time
    summary["throughput"] = number(summary.get("throughput_per_hour"), 0.0)
    return summary


def load_record(directory: str, scenario: str) -> dict:
    root = Path(directory)
    record = {"scenario": scenario, "label": SCENARIO_LABELS.get(scenario, scenario.title()),
              "dir": str(root), "summary": {}, "agents": {}, "timeseries": {}, "events": {},
              "meta": read_json(root / "experiment_meta.json") or {}}
    for method, stem in (("MAPPO", "MAPPO"), ("FCFS", "FCFS")):
        summary = read_json(root / f"{stem}_summary.json")
        agents = read_csv(root / f"{stem}_per_agent.csv")
        record["summary"][method] = derive_summary(summary, agents) if (summary or agents) else None
        record["agents"][method] = agents
        record["timeseries"][method] = read_csv(root / f"{stem}_timeseries.csv")
        record["events"][method] = read_json(root / f"{stem}_events.json") or {}
    record["micro_priority"] = {
        method: read_json(root / f"{stem}_micro_priority.json") or {}
        for method, stem in (("MAPPO", "MAPPO"), ("FCFS", "FCFS"))
    }
    record["micro_corridor"] = {
        method: read_json(root / f"{stem}_micro_corridor.json") or {}
        for method, stem in (("MAPPO", "MAPPO"), ("FCFS", "FCFS"))
    }
    return record


def available_records(records: list[dict]) -> list[dict]:
    return [r for r in records if any(r["summary"].get(m) for m in METHODS)]


def save_figure(fig, out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_dir / filename}")


def add_bar_labels(ax, bars, values, labels, fmt="{:.1f}"):
    for bar, value, label in zip(bars, values, labels):
        if value is None:
            continue
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1.2,
                fmt.format(value) + (f"\n{label}" if label else ""),
                ha="center", va="bottom", fontsize=7)


def plot_scenario_outcomes(records: list[dict], out_dir: Path):
    records = available_records(records)
    if not records:
        return
    metrics = [
        ("safe_rate", "Safe completion rate (%)", "Safe completion"),
        ("timeout_rate", "Timeout rate (%)", "Timeouts"),
        ("collision_per_100", "Collision events per 100 flights", "Collisions"),
        ("throughput", "Throughput (flights/h)", "System throughput"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    x = np.arange(len(records))
    width = 0.36
    for ax, (key, ylabel, title) in zip(axes.flat, metrics):
        for offset, method in ((-width / 2, "FCFS"), (width / 2, "MAPPO")):
            values = []
            labels = []
            for record in records:
                summary = record["summary"].get(method)
                values.append(summary.get(key) if summary else 0.0)
                if not summary:
                    labels.append("n/a")
                elif key == "safe_rate":
                    labels.append(f"{int(summary.get('success_count', 0))}/{int(summary['total_count'])}")
                elif key == "timeout_rate":
                    labels.append(f"{int(summary.get('timeout_count', 0))}/{int(summary['total_count'])}")
                else:
                    labels.append(f"N={int(summary['total_count'])}")
            bars = ax.bar(x + offset, values, width, color=METHODS[method]["color"],
                          alpha=0.85, label=METHODS[method]["label"])
            if key in {"safe_rate", "timeout_rate"}:
                add_bar_labels(ax, bars, values, labels)
        ax.set_title(title, pad=14)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in records])
        if key in {"safe_rate", "timeout_rate"}:
            ax.set_ylim(0, 118)
        ax.legend(loc="best", frameon=False)
    fig.suptitle("Performance under baseline and dynamic perturbations", fontsize=14, y=0.995)
    fig.text(0.5, 0.01, "Labels show completed or timed-out flights / total flights; no retry is applied after collision or deadlock.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.90, wspace=0.15, hspace=0.55)
    save_figure(fig, out_dir, "fig_scenario_outcomes.png")


def series_xy(rows: list[dict], total: int):
    points = []
    for row in rows:
        x = number(row.get("time_min"), None)
        if x is None:
            x = number(row.get("step"), None)
        y = number(row.get("cumulative_success"), 0.0)
        if x is not None:
            points.append((x, 100.0 * y / total if total else 0.0))
    points.sort(key=lambda p: p[0])
    return points


def event_intervals(event_payload: dict, time_scale: float, total_steps: float):
    """Return event spans in simulation minutes when the new event export exists."""
    intervals = []
    events = event_payload.get("events", []) if isinstance(event_payload, dict) else []
    for event in events:
        step = number(event.get("step"), None)
        if step is None or step < 0:
            continue
        duration = number(event.get("duration"), 0.0)
        start = step * time_scale / 60.0
        end = (step + duration) * time_scale / 60.0
        intervals.append((start, max(start, end), event.get("type", "event")))
    return intervals


def plot_perturbation_timeseries(records: list[dict], out_dir: Path):
    perturbations = [r for r in records if r["scenario"] in {"delay", "closure", "priority"}]
    if not perturbations:
        return
    fig, axes = plt.subplots(len(perturbations), 1, figsize=(9, 3.0 * len(perturbations)), squeeze=False)
    for ax, record in zip(axes[:, 0], perturbations):
        event_types = {
            "delay": {"delay_release_plan"},
            "closure": {"closure"},
            "priority": {"priority_boost"},
        }.get(record["scenario"], set())
        for method in ("MAPPO", "FCFS"):
            summary = record["summary"].get(method)
            if not summary:
                continue
            points = series_xy(record["timeseries"].get(method, []), int(summary["total_count"]))
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=METHODS[method]["color"], linewidth=2.0,
                    label=METHODS[method]["label"])
            ax.scatter(xs[-1], ys[-1], color=METHODS[method]["color"], s=22, zorder=3)
            ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=METHODS[method]["color"])
            # The new runner exports events per algorithm; old result folders simply have no shading.
            payload = record["events"].get(method, {})
            time_scale = number(record["meta"].get("time_scale"), 3.0) or 3.0
            for start, end, event_type in event_intervals(payload, time_scale, xs[-1] if xs else 0):
                if event_type not in event_types:
                    continue
                ax.axvspan(start, end, color="#999999", alpha=0.12, zorder=0)
                ax.axvline(start, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_title(record["label"])
        ax.set_ylabel("Cumulative safe completion (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", frameon=False)
    axes[-1, 0].set_xlabel("Simulation time (min)")
    fig.suptitle("Cumulative safe completion under perturbations", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, out_dir, "fig_perturbation_timeseries.png")


def wait_values(rows: list[dict]):
    values = []
    for row in rows:
        if is_true(row.get("crashed", False)):
            continue
        onroute = number(row.get("onroute_wait_sec"), 0.0) or 0.0
        queue = number(row.get("wait_time_min"), 0.0) or 0.0
        if row.get("onroute_wait_sec") not in (None, "", "None") or row.get("wait_time_min") not in (None, "", "None"):
            values.append(onroute / 60.0 + queue)
    return values


def plot_delay_wait(records: list[dict], out_dir: Path):
    record = next((r for r in records if r["scenario"] == "delay"), None)
    if not record:
        return
    data = []
    labels = []
    colors = []
    for method in ("MAPPO", "FCFS"):
        vals = wait_values(record["agents"].get(method, []))
        if vals:
            data.append(vals)
            labels.append(METHODS[method]["label"])
            colors.append(METHODS[method]["color"])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    means = [float(np.mean(values)) for values in data]
    ax.scatter(np.arange(1, len(data) + 1), means, marker="D", color="#222222", s=32,
               label="Mean", zorder=4)
    for i, (mean, values) in enumerate(zip(means, data), start=1):
        ax.text(i, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, 0.98, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#555555")
    ax.set_title("Total waiting time in the delay scenario")
    ax.set_ylabel("Waiting time (min)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_delay_wait_distribution.png")


def draw_event_completion(ax, record: dict, event_type: str):
    """Draw cumulative completion and shade the actual logged event windows."""
    for method in ("MAPPO", "FCFS"):
        summary = record["summary"].get(method)
        if not summary:
            continue
        points = series_xy(record["timeseries"].get(method, []), int(summary["total_count"]))
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=METHODS[method]["color"], linewidth=2.0,
                label=METHODS[method]["label"])
        payload = record["events"].get(method, {})
        time_scale = number(record["meta"].get("time_scale"), 3.0) or 3.0
        for start, end, logged_type in event_intervals(payload, time_scale, xs[-1]):
            if logged_type != event_type and not str(logged_type).startswith(f"{event_type}_"):
                continue
            ax.axvspan(start, end, color="#888888", alpha=0.15, zorder=0)
            ax.axvline(start, color="#555555", linestyle="--", linewidth=0.9)
            ax.text(start, 101, event_type.title(), fontsize=7, color="#555555",
                    rotation=90, va="top", ha="right")
        ax.scatter(xs[-1], ys[-1], color=METHODS[method]["color"], s=20, zorder=3)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Cumulative safe completion (%)")
    ax.set_xlabel("Simulation time (min)")
    ax.legend(frameon=False, loc="lower right")


def plot_delay_mechanism(records: list[dict], out_dir: Path):
    """Combine the delay event response with the waiting-time distribution."""
    record = next((r for r in records if r["scenario"] == "delay"), None)
    if not record:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_event_completion(axes[0], record, "delay")
    axes[0].set_title("Delay release and cumulative completion")

    data, labels, colors = [], [], []
    for method in ("MAPPO", "FCFS"):
        values = wait_values(record["agents"].get(method, []))
        if values:
            data.append(values)
            labels.append(METHODS[method]["label"])
            colors.append(METHODS[method]["color"])
    if data:
        box = axes[1].boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        means = [float(np.mean(values)) for values in data]
        axes[1].scatter(np.arange(1, len(data) + 1), means, marker="D", color="#222222",
                        s=28, label="Mean", zorder=4)
        axes[1].set_ylabel("Total waiting time (min)")
        axes[1].legend(frameon=False)
    axes[1].set_title("Waiting-time response")
    fig.suptitle("Delay mechanism: release, waiting, and recovery", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_delay_mechanism.png")


def path_distance_values(record: dict, method: str):
    values = []
    for row in record["agents"].get(method, []):
        if is_true(row.get("crashed", False)):
            continue
        value = number(row.get("path_dist"), None)
        if value is not None and value > 0:
            values.append(value)
    return values


def plot_closure_mechanism(records: list[dict], out_dir: Path):
    """Show closure response plus the route-distance change caused by rerouting."""
    closure = next((r for r in records if r["scenario"] == "closure"), None)
    baseline = next((r for r in records if r["scenario"] == "baseline"), None)
    if not closure:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_event_completion(axes[0], closure, "closure")
    axes[0].set_title("Closure event and cumulative completion")

    data, labels, colors = [], [], []
    for method in ("MAPPO", "FCFS"):
        for record, suffix in ((baseline, "Baseline"), (closure, "Closure")):
            if not record:
                continue
            values = path_distance_values(record, method)
            if values:
                data.append(values)
                labels.append(f"{METHODS[method]['label']}\n{suffix}")
                colors.append(METHODS[method]["color"])
    if data:
        box = axes[1].boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
        axes[1].set_ylabel("Completed path distance (graph units)")
        axes[1].tick_params(axis="x", labelsize=8)
    corridor_counts = []
    for method in ("MAPPO", "FCFS"):
        payload = closure["micro_corridor"].get(method, {})
        corridor_counts.append(len(payload.get("events", [])) if payload else 0)
    suffix = f" | corridor micro-events: MAPPO={corridor_counts[0]}, FCFS={corridor_counts[1]}"
    fig.suptitle("Closure mechanism: rerouting and route-distance change" + suffix,
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_closure_mechanism.png")


def extract_priority_series(payload: dict):
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    if not frames:
        return None
    event_step = number(payload.get("event_step"), 0.0) or 0.0
    vip_ids = set()
    for frame in frames:
        for agent in frame.get("agents", []):
            if is_true(agent.get("is_vip", False)):
                vip_ids.add(agent.get("id"))
        for agent_id in frame.get("boosted", []):
            vip_ids.add(agent_id)
    vip_speed = {}
    normal_speed = {}
    for frame in frames:
        rel = number(frame.get("step"), 0.0) - event_step
        vip = [number(a.get("speed"), 0.0) or 0.0 for a in frame.get("agents", []) if a.get("id") in vip_ids]
        normal = [number(a.get("speed"), 0.0) or 0.0 for a in frame.get("agents", []) if a.get("id") not in vip_ids]
        if vip:
            vip_speed.setdefault(rel, []).extend(vip)
        if normal:
            normal_speed.setdefault(rel, []).extend(normal)
    def average(mapping):
        return sorted((x, float(np.mean(values))) for x, values in mapping.items())
    return event_step, average(vip_speed), average(normal_speed)


def plot_priority_response(records: list[dict], out_dir: Path):
    record = next((r for r in records if r["scenario"] == "priority"), None)
    if not record:
        return
    series = {method: extract_priority_series(record["micro_priority"].get(method, {}))
              for method in METHODS}
    if not any(series.values()):
        # Rolling execution records actual scheduled-flight IDs.  When a
        # trajectory micro-log is unavailable, report the aggregate priority
        # outcome rather than silently omitting the priority figure.
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        labels, values, colors, notes = [], [], [], []
        for method in ("MAPPO", "FCFS"):
            rows = record["agents"].get(method, [])
            high = [row for row in rows if number(row.get("priority"), 1) > 1]
            normal = [row for row in rows if number(row.get("priority"), 1) <= 1]
            for group, group_rows, color in (("Priority", high, "#E69F00"), ("Normal", normal, "#777777")):
                group_values = [number(row.get("taxi_sec"), None) for row in group_rows]
                clean = [value for value in group_values if value is not None]
                if group_rows:
                    labels.append(f"{METHODS[method]['label']}\n{group}")
                    values.append(float(np.mean(clean)) / 60.0 if clean else 0.0)
                    colors.append(color if method == "MAPPO" else ("#F2C14E" if group == "Priority" else "#B0B0B0"))
                    notes.append(f"{len(clean)}/{len(group_rows)} completed")
        if values:
            bars = ax.bar(np.arange(len(values)), values, color=colors, alpha=0.85)
            for bar, value, note in zip(bars, values, notes):
                y = value if value > 0 else 0.015
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.2f}\n{note}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Average taxi time (min)")
            ax.set_title("Priority scenario: aggregate service outcome")
            ax.text(0.5, -0.22, "Aggregate fallback: no event-window trajectory log was exported.",
                    transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
            fig.tight_layout()
            save_figure(fig, out_dir, "fig_priority_event_response.png")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, method in zip(axes, ("MAPPO", "FCFS")):
        parsed = series.get(method)
        if not parsed:
            ax.text(0.5, 0.5, "No micro-log", transform=ax.transAxes, ha="center")
            ax.set_title(METHODS[method]["label"])
            continue
        _, vip, normal = parsed
        if vip:
            x, y = zip(*vip)
            ax.plot(x, y, color="#E69F00", linewidth=2.2, label="Priority flight")
        if normal:
            x, y = zip(*normal)
            ax.plot(x, y, color="#777777", linewidth=1.6, label="Normal flights")
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(METHODS[method]["label"])
        ax.set_xlabel("Relative step (event = 0)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean speed (normalized graph units / step)")
    fig.suptitle("Local response to a dynamic priority change", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_priority_event_response.png")


def export_metrics(records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "method", "total_count", "safe_completion_pct", "eventual_service_pct",
              "deadlocks", "deadlocks_per_100", "collisions", "avg_total_time_min", "throughput_per_hour"]
    rows = []
    for record in records:
        for method in METHODS:
            summary = record["summary"].get(method)
            if not summary:
                continue
            rows.append({
                "scenario": record["scenario"],
                "method": METHODS[method]["label"],
                "total_count": int(summary["total_count"]),
                "safe_completion_pct": round(summary["safe_rate"], 3),
                "eventual_service_pct": round(summary["eventual_rate"], 3),
                "deadlocks": int(summary["deadlock_count"]),
                "deadlocks_per_100": round(summary["deadlock_per_100"], 3),
                "collisions": int(summary["collision_count"]),
                "avg_total_time_min": round(summary["avg_total_time"], 3),
                "throughput_per_hour": round(summary["throughput"], 3),
            })
    with (out_dir / "scenario_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "scenario_metrics.md").open("w", encoding="utf-8") as f:
        f.write("| Scenario | Method | N | Safe completion (%) | Eventual service (%) | Deadlocks/100 | Avg total taxi (min) | Throughput (flt/hr) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write("| {scenario} | {method} | {total_count} | {safe_completion_pct:.2f} | {eventual_service_pct:.2f} | {deadlocks_per_100:.2f} | {avg_total_time_min:.2f} | {throughput_per_hour:.2f} |\n".format(**row))
    print(f"[OK] {out_dir / 'scenario_metrics.csv'}")
    print(f"[OK] {out_dir / 'scenario_metrics.md'}")


def main():
    parser = argparse.ArgumentParser(description="Generate TRB-ready airport MARL result figures")
    parser.add_argument("--baseline-dir", default="results/exp_4_2")
    parser.add_argument("--delay-dir", default="results/exp_4_3_delay")
    parser.add_argument("--closure-dir", default="results/exp_4_3_closure")
    parser.add_argument("--priority-dir", default="results/exp_4_3_priority")
    parser.add_argument("--out-dir", default="results/trb_figures")
    args = parser.parse_args()

    pairs = [("baseline", args.baseline_dir), ("delay", args.delay_dir),
             ("closure", args.closure_dir), ("priority", args.priority_dir)]
    records = [load_record(directory, scenario) for scenario, directory in pairs]
    records = available_records(records)
    if not records:
        raise SystemExit("No valid result summaries found.")
    out_dir = Path(args.out_dir)
    plot_scenario_outcomes(records, out_dir)
    plot_perturbation_timeseries(records, out_dir)
    plot_delay_wait(records, out_dir)
    plot_delay_mechanism(records, out_dir)
    plot_closure_mechanism(records, out_dir)
    plot_priority_response(records, out_dir)
    export_metrics(records, out_dir)
    print(f"Generated figures from {len(records)} scenario result folders.")


if __name__ == "__main__":
    main()
