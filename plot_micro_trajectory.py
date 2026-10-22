"""
微观时空轨迹图可视化脚本 (对比版)

输入: 
  - results/exp_4_3_priority/MAPPO_micro_priority.json
  - results/exp_4_3_priority/FCFS_micro_priority.json
输出:
  1. 时空路径图 (Time-Space Diagram): X轴=仿真步, Y轴=拓扑节点, 每架飞机一条折线
  2. 速度剖面散点图: X轴=仿真步, Y轴=速度值, VIP用特殊标记
  支持同时展示 MAPPO 和 FCFS 的上下对比。

使用方法:
  python plot_micro_trajectory.py
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import defaultdict

# 解决中文显示问题，增加多个常见中文字体 fallback，确保跨平台可用
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'PingFang SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(json_path: str):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_agent_series(frames: list, target_agents: list = None):
    """将帧数据重组为 per-agent 的时间序列。"""
    series = defaultdict(lambda: {
        "steps": [], "pos": [], "speed": [], "action": [],
        "on_edge": [], "progress": [], "is_vip": [], "priority": []
    })
    for frame in frames:
        step = frame["step"]
        for a in frame["agents"]:
            aid = a["id"]
            if target_agents and aid not in target_agents:
                continue
            s = series[aid]
            s["steps"].append(step)
            s["pos"].append(a["pos"])
            s["speed"].append(a["speed"])
            s["action"].append(a["action"])
            s["on_edge"].append(a["on_edge"])
            s["progress"].append(a["progress"])
            s["is_vip"].append(a["is_vip"])
            s["priority"].append(a["priority"])
    return series

def get_algo_target_agents(data, top_n=8):
    """从单个数据源中提取要绘制的飞机 ID"""
    vip_ids = set()
    active_non_vip = defaultdict(int)
    pos_history = defaultdict(set)

    frames = data["frames"]
    event_step = data["event_step"]
    
    for frame in frames:
        if frame["step"] >= event_step:
            for a in frame["agents"]:
                if a["is_vip"] == 1:
                    vip_ids.add(a["id"])
            break
            
    for frame in frames:
        for a in frame["agents"]:
            aid = a["id"]
            if aid not in vip_ids:
                active_non_vip[aid] += 1
                pos_history[aid].add(a["pos"])

    n_ordinary = max(1, top_n - len(vip_ids))
    # 优先挑选在窗口内移动过（经过节点数最多）的普通飞机，而不是一直死锁的飞机
    ordinary_ids = sorted(active_non_vip.keys(), key=lambda x: (len(pos_history[x]), active_non_vip[x]), reverse=True)[:n_ordinary]
    
    return list(vip_ids), ordinary_ids


def plot_single_algo(ax1, ax2, data: dict, title_prefix="", pre_vip=None, pre_ord=None):
    """在指定的两个子图上绘制单一算法的微观轨迹"""
    if pre_vip is None or pre_ord is None:
        vip_ids, ordinary_ids = get_algo_target_agents(data)
    else:
        vip_ids, ordinary_ids = pre_vip, pre_ord

    frames = data["frames"]
    event_step = data["event_step"]
    time_scale = data.get("time_scale", 3.0)

    target_ids = list(vip_ids) + ordinary_ids
    series = build_agent_series(frames, target_agents=target_ids)

    # 配色
    vip_color = "#ff4444"
    ordinary_colors = plt.cm.tab20(np.linspace(0, 1, len(ordinary_ids)))
    color_map = {}
    for i, aid in enumerate(vip_ids):
        color_map[aid] = vip_color
    for i, aid in enumerate(ordinary_ids):
        color_map[aid] = ordinary_colors[i]

    all_nodes = sorted(set(pos for s in series.values() for pos in s["pos"]))
    node_to_y = {n: i for i, n in enumerate(all_nodes)}

    for ax in [ax1, ax2]:
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="#aaaacc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaaacc")
        ax.axvline(event_step, color="#ffff00", linewidth=1.5, linestyle="--", alpha=0.8, zorder=10)
        ax.axvspan(event_step, event_step + 200, alpha=0.06, color="#ffff00", zorder=0)

    # ── 子图1: 时空路径图
    for aid, s in series.items():
        if not s["steps"]: continue
        y_vals = [node_to_y.get(p, 0) for p in s["pos"]]
        lw = 3.0 if aid in vip_ids else 1.2
        alpha = 1.0 if aid in vip_ids else 0.55
        z = 6 if aid in vip_ids else 3
        ax1.plot(s["steps"], y_vals, color=color_map[aid], linewidth=lw, alpha=alpha, zorder=z)

        if aid not in vip_ids:
            for si, (st, act) in enumerate(zip(s["steps"], s["action"])):
                if act == 7:
                    ax1.scatter(st, y_vals[si], color=color_map[aid], marker="x",
                                s=18, linewidths=1.2, alpha=0.6, zorder=5)

    for aid in vip_ids:
        s = series.get(aid)
        if s and s["steps"]:
            for si, st in enumerate(s["steps"]):
                if st >= event_step:
                    ax1.annotate(
                        f"VIP #{aid} 提升",
                        xy=(st, node_to_y.get(s["pos"][si], 0)),
                        xytext=(st + 15, node_to_y.get(s["pos"][si], 0) + 1),
                        arrowprops=dict(arrowstyle="->", color=vip_color, lw=1.5),
                        color=vip_color, fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec=vip_color, alpha=0.9),
                        zorder=15,
                    )
                    break

    ax1.set_xlabel(f"仿真步 (Step, 1步 = {time_scale}s)", color="#aaaacc", fontsize=9)
    ax1.set_ylabel("拓扑节点", color="#aaaacc", fontsize=9)
    ax1.set_title(f"{title_prefix} - 时空轨迹图 (× = Safety Shield 拦截 / 强制等待)", color="white", fontsize=11, pad=5)
    ax1.set_yticks(range(len(all_nodes)))
    ax1.set_yticklabels([str(n) for n in all_nodes], fontsize=6, color="#aaaacc")

    # ── 子图2: 速度剖面散点图
    for aid, s in series.items():
        if not s["steps"]: continue
        is_vip_arr = s["is_vip"]
        vip_mask = [v == 1 for v in is_vip_arr]
        non_vip_mask = [v == 0 for v in is_vip_arr]
        steps_arr = np.array(s["steps"])
        speed_arr = np.array(s["speed"])

        if aid in vip_ids:
            ax2.scatter(steps_arr[vip_mask], speed_arr[vip_mask],
                        color=vip_color, s=20, marker="*", zorder=6, label=f"VIP #{aid}" if aid == list(vip_ids)[0] else "")
            ax2.scatter(steps_arr[non_vip_mask], speed_arr[non_vip_mask],
                        color=vip_color, s=8, alpha=0.3, zorder=4)
        else:
            ax2.scatter(steps_arr, speed_arr,
                        color=color_map[aid], s=5, alpha=0.35, zorder=3)

    for aid in vip_ids:
        s = series.get(aid)
        if s and s["steps"]:
            pre_spd = [sp for st, sp in zip(s["steps"], s["speed"]) if st < event_step]
            if pre_spd:
                ax2.axhline(np.mean(pre_spd), color=vip_color, linestyle=":", alpha=0.5, linewidth=1)
                ax2.text(frames[0]["step"], np.mean(pre_spd) + 0.05,
                         f"均速: {np.mean(pre_spd):.2f}", color=vip_color, fontsize=7, alpha=0.8)

    ax2.set_xlabel(f"仿真步 (Step)", color="#aaaacc", fontsize=9)
    ax2.set_ylabel("速度", color="#aaaacc", fontsize=9)
    ax2.set_title(f"{title_prefix} - 速度剖面图", color="white", fontsize=11, pad=5)
    
    return color_map, vip_ids, ordinary_ids

def save_single_figure(data, title_prefix, out_path, pre_vip=None, pre_ord=None):
    """单独为某个算法生成并保存一张图片"""
    if not data:
        return None, None

    fig = plt.figure(figsize=(20, 8), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    cmap, v_ids, o_ids = plot_single_algo(ax1, ax2, data, title_prefix, pre_vip, pre_ord)
    event_step = data["event_step"]

    # 图例
    legend_patches = [mpatches.Patch(color="#ff4444", label=f"VIP 飞机 (#{v_ids})")]
    for aid in o_ids[:5]:
        legend_patches.append(mpatches.Patch(color=cmap.get(aid, "white"), label=f"普通飞机 #{aid}"))
    legend_patches.append(mpatches.Patch(color="gray", label="× = 强制拦截/等待 (action=7)"))
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=8, facecolor="#1a1a2e", edgecolor="#aaaacc", labelcolor="white")

    plt.suptitle(
        f"微观时空轨迹剖面 ({title_prefix}) — 优先级突变 @ Step {event_step}\n"
        f"（MAPPO：VIP畅行，普通航班微调避让；FCFS：大面积死锁卡死）",
        color="white", fontsize=15, y=0.98
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"[微观图] 已保存图片: {out_path}")
    plt.close(fig)
    return v_ids, o_ids


def plot_micro_comparison(mappo_data, fcfs_data, base_out_path, top_n=8):
    out_dir = os.path.dirname(base_out_path)
    if not out_dir: out_dir = "."
    
    # 优先画 MAPPO，提取它的 VIP 和普通飞机
    mappo_out = os.path.join(out_dir, "MAPPO_micro_trajectory.png")
    v_m, o_m = save_single_figure(mappo_data, "MAPPO (本文方法)", mappo_out)
    
    # 如果有 FCFS，用 MAPPO 提取的飞机去画 FCFS（保证对比的飞机是同一批）
    if fcfs_data:
        fcfs_out = os.path.join(out_dir, "FCFS_micro_trajectory.png")
        save_single_figure(fcfs_data, "FCFS+A* (传统算法)", fcfs_out, pre_vip=v_m, pre_ord=o_m)


def main():
    parser = argparse.ArgumentParser(description="Priority 微观时空轨迹对比图")
    parser.add_argument("--mappo", type=str, default="results/exp_4_3_priority/MAPPO_micro_priority.json")
    parser.add_argument("--fcfs", type=str, default="results/exp_4_3_priority/FCFS_micro_priority.json")
    parser.add_argument("--output", type=str, default="results/exp_4_3_priority/Micro_Comparison_Timespace.png")
    parser.add_argument("--top-n", type=int, default=8, help="显示的飞机总数")
    args = parser.parse_args()

    mappo_data = load_data(args.mappo)
    fcfs_data = load_data(args.fcfs)

    if not mappo_data and not fcfs_data:
        print("[错误] 找不到任何微观数据文件。请先运行: python run_paper_exp_4_2.py --scenario priority")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    plot_micro_comparison(mappo_data, fcfs_data, args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()
