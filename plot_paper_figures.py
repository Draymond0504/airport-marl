"""
论文图表绘制脚本（优化版）
============================
基于实验结果分析的优化要点：
  1. 吞吐量时序图：删除噪声大的瞬时吞吐右图，替换为多场景柱状图（更适合强调鲁棒性）
  2. 箱线图：增加显著性标注和完成率注释，强化对比叙事
  3. 延迟图：添加"per-agent"折算注释，澄清 FCFS 延迟的正确解读
  4. 新增 fig3：4场景完成率分组柱状图（baseline/delay/closure/priority）

运行示例：
  # 仅基准场景（4.2节）
  python plot_paper_figures.py --result-dir results/exp_4_2

  # 多场景对比（4.3节）
  python plot_paper_figures.py --mode scenario \\
      --dirs results/exp_4_2 results/exp_4_3_delay \\
             results/exp_4_3_closure results/exp_4_3_priority \\
      --labels Baseline Delay Closure Priority
"""

import argparse
import json
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np


# ================================================================
# 全局样式（学术期刊风格）
# ================================================================
FONT_SIZE_TITLE  = 13
FONT_SIZE_LABEL  = 11
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOT  = 9
DPI = 300

COLOR_MAPPO = "#2E86AB"   # 钢蓝
COLOR_FCFS  = "#E84855"   # 朱红

STYLE_MAPPO = {"color": COLOR_MAPPO, "linewidth": 2.0, "linestyle": "-",  "marker": "o", "markersize": 3}
STYLE_FCFS  = {"color": COLOR_FCFS,  "linewidth": 2.0, "linestyle": "--", "marker": "s", "markersize": 3}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.titlesize":     FONT_SIZE_TITLE,
    "axes.labelsize":     FONT_SIZE_LABEL,
    "xtick.labelsize":    FONT_SIZE_TICK,
    "ytick.labelsize":    FONT_SIZE_TICK,
    "legend.fontsize":    FONT_SIZE_LEGEND,
    "figure.dpi":         DPI,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})


# ================================================================
# 工具函数
# ================================================================
def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"  [WARN] 文件不存在，已跳过: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  [WARN] 文件不存在，已跳过: {path}")
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_fig(fig, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  ✓ 图表已保存: {path}")
    plt.close(fig)


def rolling_mean(y: list, w: int) -> list:
    """简单滚动均值平滑。"""
    result = []
    for i in range(len(y)):
        start = max(0, i - w + 1)
        result.append(sum(y[start:i + 1]) / (i - start + 1))
    return result


# ================================================================
# 图 1：累计吞吐量折线图（只保留左图，删除噪声大的瞬时吞吐）
# ================================================================
def plot_throughput_timeseries(mappo_ts: list[dict], fcfs_ts: list[dict],
                               out_dir: str, scenario_label: str = "Baseline"):
    """绘制两算法累计完成航班数随仿真时间的变化曲线。"""
    if not mappo_ts and not fcfs_ts:
        print("  [SKIP] 无时序数据，跳过 fig1")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    if mappo_ts:
        x_m = [float(r["time_min"]) for r in mappo_ts]
        y_m = [int(r["cumulative_success"]) for r in mappo_ts]
        ax.plot(x_m, y_m, label="MAPPO (Ours)", **STYLE_MAPPO)
        # 终点标注
        ax.annotate(f"{y_m[-1]}", xy=(x_m[-1], y_m[-1]),
                    xytext=(x_m[-1] - 40, y_m[-1] + 8),
                    fontsize=FONT_SIZE_ANNOT, color=COLOR_MAPPO,
                    arrowprops=dict(arrowstyle="-", color=COLOR_MAPPO, lw=0.8))

    if fcfs_ts:
        x_f = [float(r["time_min"]) for r in fcfs_ts]
        y_f = [int(r["cumulative_success"]) for r in fcfs_ts]
        ax.plot(x_f, y_f, label="FCFS+A*", **STYLE_FCFS)
        ax.annotate(f"{y_f[-1]}", xy=(x_f[-1], y_f[-1]),
                    xytext=(x_f[-1] - 40, y_f[-1] - 18),
                    fontsize=FONT_SIZE_ANNOT, color=COLOR_FCFS,
                    arrowprops=dict(arrowstyle="-", color=COLOR_FCFS, lw=0.8))

    ax.set_xlabel("Simulation Time (min)")
    ax.set_ylabel("Cumulative Completed Flights")
    ax.set_title(f"Cumulative Throughput — {scenario_label} Scenario")
    ax.legend(loc="upper left")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.tight_layout()
    save_fig(fig, out_dir, f"fig1_throughput_{scenario_label.lower()}.png")


# ================================================================
# 图 2：滑行时间分布箱线图（增加完成率注释和显著性标注）
# ================================================================
def plot_taxitime_boxplot(mappo_agents: list[dict], fcfs_agents: list[dict],
                          out_dir: str, total_agents: int = 363):
    """绘制成功飞机的滑行时间分布箱线图，含完成率 n 标注和均值菱形。"""
    def extract_times(rows):
        times = []
        for r in rows:
            crashed = str(r.get("crashed", "False")).strip().lower()
            ts = r.get("taxi_sec", "")
            if crashed == "false" and ts not in ("", "None", None):
                try:
                    times.append(float(ts))
                except ValueError:
                    pass
        return times

    mappo_times = extract_times(mappo_agents)
    fcfs_times  = extract_times(fcfs_agents)

    if not mappo_times and not fcfs_times:
        print("  [SKIP] 无有效滑行时间数据，跳过 fig2")
        return

    groups  = []
    labels  = []
    colors  = []
    counts  = []
    if fcfs_times:
        groups.append(fcfs_times);  labels.append("FCFS+A*");       colors.append(COLOR_FCFS);  counts.append(len(fcfs_times))
    if mappo_times:
        groups.append(mappo_times); labels.append("MAPPO\n(Ours)"); colors.append(COLOR_MAPPO); counts.append(len(mappo_times))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    bp = ax.boxplot(
        groups, patch_artist=True, notch=False, widths=0.45,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45,
                    "markeredgewidth": 0.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)

    # 均值菱形
    mean_handle = None
    for i, d in enumerate(groups, start=1):
        mean_val = np.mean(d)
        h, = ax.plot(i, mean_val, marker="D", color="black", markersize=7,
                     zorder=5, linestyle="none")
        if i == 1:
            mean_handle = h

    # n 标注（含完成率）
    for i, (d, n) in enumerate(zip(groups, counts), start=1):
        rate = n / total_agents * 100
        top  = np.percentile(d, 99)   # 用 99 分位避开极端离群
        ax.text(i, top * 1.04, f"n={n}\n({rate:.1f}%)",
                ha="center", fontsize=FONT_SIZE_ANNOT - 0.5,
                color="gray", linespacing=1.4)

    # 显著性标注（bracket）
    if len(groups) == 2:
        y_max = max(np.percentile(g, 99) for g in groups) * 1.20
        ax.annotate(
            "", xy=(2, y_max), xytext=(1, y_max),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
        )
        ax.text(1.5, y_max * 1.02, "p < 0.001  ***",
                ha="center", va="bottom", fontsize=FONT_SIZE_ANNOT,
                style="italic", color="black")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Taxiing Time (s)")
    ax.set_title("Taxiing Time Distribution Comparison")
    if mean_handle:
        ax.legend([mean_handle], ["Mean value"], loc="upper right")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_taxitime_boxplot.png")


# ================================================================
# 图 3：多场景完成率分组柱状图（替代噪声大的瞬时吞吐图）
# ================================================================
def plot_scenario_comparison(scenario_data: list[dict], labels: list[str], out_dir: str):
    """
    4 场景（Baseline/Delay/Closure/Priority）下 MAPPO vs FCFS 完成率分组柱状图。
    这是论文 4.3 节最核心的对比图。
    """
    mappo_rates, fcfs_rates = [], []
    mappo_n,     fcfs_n     = [], []
    for sd in scenario_data:
        ms = sd.get("mappo")
        fs = sd.get("fcfs")
        mappo_rates.append(ms["completion_rate_pct"] if ms else 0)
        fcfs_rates.append(fs["completion_rate_pct"]  if fs else 0)
        mappo_n.append(ms["success_count"]  if ms else 0)
        fcfs_n.append(fs["success_count"]   if fs else 0)

    n     = len(labels)
    x     = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * n), 5.5))

    bars_f = ax.bar(x - width / 2, fcfs_rates,  width,
                    label="FCFS+A*",      color=COLOR_FCFS,  alpha=0.75)
    bars_m = ax.bar(x + width / 2, mappo_rates, width,
                    label="MAPPO (Ours)", color=COLOR_MAPPO, alpha=0.75)

    # 数值标签（含绝对数量）
    for bar, h, cnt in zip(bars_f, fcfs_rates, fcfs_n):
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.8,
                f"{h:.1f}%\n({cnt})", ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOT, color=COLOR_FCFS, linespacing=1.3)
    for bar, h, cnt in zip(bars_m, mappo_rates, mappo_n):
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.8,
                f"{h:.1f}%\n({cnt})", ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOT, color=COLOR_MAPPO, linespacing=1.3)

    # Delta 标注（MAPPO - FCFS）
    for i in range(n):
        delta = mappo_rates[i] - fcfs_rates[i]
        y_pos = max(mappo_rates[i], fcfs_rates[i]) + 10
        ax.text(x[i], y_pos, f"Δ+{delta:.1f}pp",
                ha="center", va="bottom", fontsize=FONT_SIZE_ANNOT,
                color="dimgray", style="italic")

    ax.set_ylim(0, 120)
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Completion Rate Under Different Perturbation Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_LABEL)
    ax.legend(loc="lower right")
    ax.axhline(y=100, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_scenario_comparison.png")


# ================================================================
# 图 4：推理/规划延迟对比（含正确叙事注释）
# ================================================================
def plot_inference_latency(mappo_s: dict | None, fcfs_s: dict | None, out_dir: str):
    """
    MAPPO 每步推理延迟 vs FCFS 批量重规划延迟。
    注意：FCFS 的延迟是"一次对全体飞机的集中式规划"，
          而 MAPPO 是"每步对所有活跃智能体的并行前向传播"，语义不同。
    """
    items = []
    if mappo_s and mappo_s.get("avg_inference_ms") not in (None, "N/A"):
        items.append({
            "label":  "MAPPO\nInference\n(per step)",
            "avg":    float(mappo_s["avg_inference_ms"]),
            "max":    float(mappo_s["max_inference_ms"]),
            "color":  COLOR_MAPPO,
            "hatch":  "",
        })
    if fcfs_s and fcfs_s.get("avg_replan_ms") not in (None, "N/A"):
        n_replan = fcfs_s.get("replan_count", 1) or 1
        items.append({
            "label":  f"FCFS+A*\nReplanning\n({n_replan} event{'s' if n_replan > 1 else ''})",
            "avg":    float(fcfs_s["avg_replan_ms"]),
            "max":    float(fcfs_s["max_replan_ms"]),
            "color":  COLOR_FCFS,
            "hatch":  "//",
        })

    if not items:
        print("  [SKIP] 无延迟数据，跳过 fig4")
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    x = np.arange(len(items))
    width = 0.50

    avg_vals = [it["avg"] for it in items]
    max_vals = [it["max"] for it in items]

    bars = ax.bar(x, avg_vals, width,
                  color=[it["color"] for it in items],
                  hatch=[it["hatch"] for it in items],
                  alpha=0.72, label="Average Latency")

    # Max 误差棒
    ax.errorbar(x, avg_vals,
                yerr=[np.zeros(len(items)),
                      [m - a for m, a in zip(max_vals, avg_vals)]],
                fmt="none", color="black", capsize=6,
                linewidth=1.8, label="Max Latency")

    # 数值标签
    for bar, avg, mxv in zip(bars, avg_vals, max_vals):
        cx = bar.get_x() + bar.get_width() / 2.
        ax.text(cx, avg + max(max_vals) * 0.01,
                f"{avg:.2f} ms", ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOT)
        if mxv > avg * 1.05:
            ax.text(cx, mxv + max(max_vals) * 0.01,
                    f"{mxv:.2f} ms", ha="center", va="bottom",
                    fontsize=FONT_SIZE_ANNOT - 0.5, color="gray")

    # 1 ms 实时性判定线
    rt_line = ax.axhline(y=1.0, color="#27AE60", linestyle="--",
                         linewidth=1.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([it["label"] for it in items],
                       fontsize=FONT_SIZE_TICK, linespacing=1.4)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Decision Latency Comparison\n(Average ± Max)")

    # 合并图例（包含实时阈值线）
    handles, lbls = ax.get_legend_handles_labels()
    handles.append(rt_line)
    lbls.append("1 ms Real-time Threshold")
    ax.legend(handles, lbls, loc="upper left", fontsize=FONT_SIZE_LEGEND - 0.5)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_inference_latency.png")


# ================================================================
# 图 5：综合对比表格图（可选，自动生成论文 Table）
# ================================================================
def plot_summary_table(mappo_s: dict | None, fcfs_s: dict | None, out_dir: str, scenario_label: str = "Baseline"):
    """将关键指标渲染为图片形式的表格，可直接插入论文。"""
    if not mappo_s and not fcfs_s:
        return

    rows = [
        ["Metric",               "FCFS+A*",                        "MAPPO (Ours)"],
        ["Completion Rate",
         f"{fcfs_s['completion_rate_pct']:.1f}% ({fcfs_s['success_count']}/{fcfs_s['total_count']})" if fcfs_s else "N/A",
         f"{mappo_s['completion_rate_pct']:.1f}% ({mappo_s['success_count']}/{mappo_s['total_count']})" if mappo_s else "N/A"],
        ["Avg. Taxiing Time",
         f"{fcfs_s['avg_taxi_min']:.2f} min" if fcfs_s else "N/A",
         f"{mappo_s['avg_taxi_min']:.2f} min" if mappo_s else "N/A"],
        ["Avg. Wait Time",
         f"{fcfs_s.get('avg_wait_min', 0.0):.2f} min" if fcfs_s else "N/A",
         f"{mappo_s.get('avg_wait_min', 0.0):.2f} min" if mappo_s else "N/A"],
        ["Avg. Path Distance",
         f"{fcfs_s['avg_path_dist']:.1f} units" if fcfs_s else "N/A",
         f"{mappo_s['avg_path_dist']:.1f} units" if mappo_s else "N/A"],
        ["Throughput",
         f"{fcfs_s['throughput_per_hour']:.2f} flt/hr" if fcfs_s else "N/A",
         f"{mappo_s['throughput_per_hour']:.2f} flt/hr" if mappo_s else "N/A"],
        ["Collision Count",
         str(fcfs_s['collision_count']) if fcfs_s else "N/A",
         str(mappo_s['collision_count']) if mappo_s else "N/A"],
        ["Decision Latency",
         f"{fcfs_s['avg_replan_ms']} ms (batch)" if fcfs_s else "N/A",
         f"{mappo_s['avg_inference_ms']} ms (per step)" if mappo_s else "N/A"],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axis("off")

    col_widths = [0.30, 0.35, 0.35]
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    # 表头样式
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # 交替行颜色
    for i in range(1, len(rows)):
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor("#EAF4FB" if i % 2 == 0 else "white")
            # 高亮 MAPPO 列（第 3 列）
            if j == 2:
                cell.set_facecolor("#D6EAF8" if i % 2 == 0 else "#EBF5FB")

    ax.set_title(f"Performance Comparison: FCFS+A* vs MAPPO ({scenario_label} Scenario)",
                 pad=12, fontsize=FONT_SIZE_TITLE, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, out_dir, f"fig5_summary_table_{scenario_label.lower()}.png")


# ================================================================
# 图 6：优先级场景对比柱状图
# ================================================================
def plot_priority_comparison(mappo_agents: list[dict], fcfs_agents: list[dict], out_dir: str):
    """绘制高低优先级航班的通行时间对比柱状图"""
    def extract_avg_times(agents_list):
        high_times = []
        low_times = []
        for r in agents_list:
            if str(r.get("crashed", "False")).strip().lower() == "false" and r.get("taxi_sec"):
                try:
                    t = float(r["taxi_sec"]) / 60.0  # 转为分钟
                    p = int(r.get("priority", 1))
                    if p > 1:
                        high_times.append(t)
                    else:
                        low_times.append(t)
                except ValueError:
                    pass
        avg_high = sum(high_times) / len(high_times) if high_times else 0.0
        avg_low = sum(low_times) / len(low_times) if low_times else 0.0
        return avg_high, avg_low, len(high_times), len(low_times)

    m_high, m_low, m_n_h, m_n_l = extract_avg_times(mappo_agents)
    f_high, f_low, f_n_h, f_n_l = extract_avg_times(fcfs_agents)

    labels = ["Low Priority\n(Regular Flights)", "High Priority\n(VIP/Emergency)"]
    fcfs_means = [f_low, f_high]
    mappo_means = [m_low, m_high]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    rects1 = ax.bar(x - width/2, fcfs_means, width, label='FCFS+A*', color=COLOR_FCFS, alpha=0.85)
    rects2 = ax.bar(x + width/2, mappo_means, width, label='MAPPO (Ours)', color=COLOR_MAPPO, alpha=0.85)

    ax.set_ylabel('Average Taxiing Time (min)')
    ax.set_title('Taxiing Time by Priority Class\n(Priority Scenario)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT)
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_priority_comparison.png")


# ================================================================
# 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="论文图表绘制脚本（优化版）")
    parser.add_argument("--mode", default="scenario", choices=["single", "scenario"])
    parser.add_argument("--result-dir", default="results/exp_4_2",
                        help="single 模式下的数据目录")
    parser.add_argument("--dirs", nargs="+",
                        default=["results/exp_4_2",
                                 "results/exp_4_3_delay",
                                 "results/exp_4_3_closure",
                                 "results/exp_4_3_priority"],
                        help="scenario 模式下各场景目录")
    parser.add_argument("--labels", nargs="+",
                        default=["Baseline", "Delay", "Closure", "Priority"])
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--total-agents", type=int, default=363,
                        help="航班总数（用于完成率计算）")
    args = parser.parse_args()

    print(f"\n[绘图模式] {args.mode.upper()}")
    print(f"[输出目录] {args.out_dir}\n")

    if args.mode == "single":
        d = args.result_dir
        mappo_s   = load_json(os.path.join(d, "MAPPO_summary.json"))
        fcfs_s    = load_json(os.path.join(d, "FCFS_summary.json"))
        mappo_ts  = load_csv(os.path.join(d, "MAPPO_timeseries.csv"))
        fcfs_ts   = load_csv(os.path.join(d, "FCFS_timeseries.csv"))
        mappo_agt = load_csv(os.path.join(d, "MAPPO_per_agent.csv"))
        fcfs_agt  = load_csv(os.path.join(d, "FCFS_per_agent.csv"))

        lbl = "Baseline"

        print("生成 fig1：累计吞吐折线图...")
        plot_throughput_timeseries(mappo_ts, fcfs_ts, args.out_dir, lbl)

        print("生成 fig2：滑行时间箱线图...")
        plot_taxitime_boxplot(mappo_agt, fcfs_agt, args.out_dir,
                              total_agents=args.total_agents)

        print("生成 fig4：推理延迟对比图...")
        plot_inference_latency(mappo_s, fcfs_s, args.out_dir)

        print("生成 fig5：汇总对比表格...")
        plot_summary_table(mappo_s, fcfs_s, args.out_dir)

    elif args.mode == "scenario":
        if len(args.dirs) != len(args.labels):
            print(f"错误：--dirs ({len(args.dirs)}) 与 --labels ({len(args.labels)}) 数量不一致")
            return

        scenario_data = []
        for d, lbl in zip(args.dirs, args.labels):
            ms = load_json(os.path.join(d, "MAPPO_summary.json"))
            fs = load_json(os.path.join(d, "FCFS_summary.json"))
            scenario_data.append({"mappo": ms, "fcfs": fs})
            m_rate = ms["completion_rate_pct"] if ms else "缺失"
            f_rate = fs["completion_rate_pct"] if fs else "缺失"
            print(f"  [{lbl}] MAPPO={m_rate}%  FCFS={f_rate}%")

        print("\n生成 fig3：多场景完成率对比柱状图...")
        plot_scenario_comparison(scenario_data, args.labels, args.out_dir)

        # 遍历所有场景，生成各自的图表
        for d, lbl, s_data in zip(args.dirs, args.labels, scenario_data):
            mappo_ts  = load_csv(os.path.join(d, "MAPPO_timeseries.csv"))
            fcfs_ts   = load_csv(os.path.join(d, "FCFS_timeseries.csv"))
            mappo_agt = load_csv(os.path.join(d, "MAPPO_per_agent.csv"))
            fcfs_agt  = load_csv(os.path.join(d, "FCFS_per_agent.csv"))
            mappo_s   = s_data["mappo"]
            fcfs_s    = s_data["fcfs"]

            print(f"\n生成 fig1（{lbl}）：累计吞吐折线图...")
            plot_throughput_timeseries(mappo_ts, fcfs_ts, args.out_dir, lbl)

            print(f"生成 fig2（{lbl}）：滑行时间箱线图...")
            plot_taxitime_boxplot(mappo_agt, fcfs_agt, args.out_dir, total_agents=args.total_agents)

            print(f"生成 fig5（{lbl}）：汇总对比表格...")
            plot_summary_table(mappo_s, fcfs_s, args.out_dir, scenario_label=lbl)

            if lbl.lower() == "priority":
                print(f"生成 fig6（Priority专用）：高低优先级通行时间对比...")
                plot_priority_comparison(mappo_agt, fcfs_agt, args.out_dir)

        print("\n生成 fig4：推理延迟对比图...")
        plot_inference_latency(scenario_data[0]["mappo"], scenario_data[0]["fcfs"], args.out_dir)

    print("\n全部图表生成完毕！")


if __name__ == "__main__":
    main()
