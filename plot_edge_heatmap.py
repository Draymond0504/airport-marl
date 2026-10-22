"""
热力图脚本：基于Baseline场景结果，绘制Layout F各边使用频率热力图。

使用方法：
  python plot_edge_heatmap.py --csv results/exp_4_2/MAPPO_per_agent.csv --algo MAPPO
  python plot_edge_heatmap.py --csv results/exp_4_2/FCFS_per_agent.csv --algo FCFS

原理：
  1. 从 per_agent.csv 中读取每架飞机的 path_nodes 序列
  2. 统计每段相邻节点对 (u->v) 的出现频次
  3. 以 Layout F 拓扑结构为底图，将边的粗细/颜色映射到使用频次
"""

import argparse
import math
import json
import os
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from collections import defaultdict

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 从 taxi_graph_env.py 提取的 Layout F 节点坐标（静态复制避免环境依赖）
# =====================================================================
def build_layout_f_pos_edges():
    """返回 Layout F 的节点坐标字典和边列表。"""
    y_b1 = 0; y_b2 = 5; y_mid_bot = 10; y_apron_top = 34
    x_fl = -30; x_al = -12; x_ac = 0; x_ar = 12; x_fr = 30

    pos = {}
    # Bypass 1
    pos[100] = (x_fl - 5, y_b1); pos[101] = (x_fl, y_b1); pos[102] = (x_al, y_b1)
    pos[103] = (x_ar, y_b1); pos[104] = (x_fr, y_b1); pos[105] = (x_fr + 5, y_b1)
    # Bypass 2
    pos[200] = (x_fl - 5, y_b2); pos[201] = (x_fl, y_b2); pos[202] = (x_al, y_b2)
    pos[203] = (x_ar, y_b2); pos[204] = (x_fr, y_b2); pos[205] = (x_fr + 5, y_b2)
    # Y=10 line
    pos[300] = (x_fl, y_mid_bot)
    pos[301] = (-27, y_mid_bot); pos[302] = (-20, y_mid_bot); pos[303] = (-16, y_mid_bot)
    pos[304] = (x_al, y_mid_bot); pos[305] = (x_ac, y_mid_bot); pos[306] = (x_ar, y_mid_bot)
    pos[307] = (18, y_mid_bot); pos[308] = (24, y_mid_bot); pos[309] = (x_fr, y_mid_bot)
    # Far Left
    pos[401] = (x_fl, 16); pos[402] = (x_fl, 22); pos[403] = (x_fl, 28); pos[404] = (x_fl, 34)
    pos[405] = (-26, 38); pos[406] = (-22, 42); pos[407] = (-18, 46); pos[408] = (-14, 50); pos[409] = (-10, 54)
    # Far Right
    pos[501] = (x_fr, 16); pos[502] = (x_fr, 22); pos[503] = (x_fr, 28); pos[504] = (x_fr, 34)
    pos[505] = (x_fr, 40); pos[506] = (x_fr, 46); pos[507] = (x_fr, 52)
    # Central Apron
    pos[601] = (x_al, 16); pos[602] = (x_al, 22); pos[603] = (x_al, 28); pos[604] = (x_al, 34); pos[605] = (x_al, 40)
    pos[611] = (x_ac, 25); pos[612] = (x_ac, 40)
    pos[621] = (x_ar, 16); pos[622] = (x_ar, 22); pos[623] = (x_ar, 28); pos[624] = (x_ar, 34); pos[625] = (x_ar, 40)
    pos[631] = (-8, 40); pos[632] = (-4, 40); pos[633] = (4, 40); pos[634] = (8, 40)
    # Runway entry/exit
    pos[900] = (-25, -2); pos[901] = (25, -2)

    # Gate stubs (2000系列) 和机位 (0-29)
    gate_id = 0
    stub_pos = {}
    gate_pos = {}

    def add_gate(parent_node, dx, dy):
        nonlocal gate_id
        stub_id = 2000 + gate_id
        stub_pos[stub_id] = (pos[parent_node][0] + dx * 0.5, pos[parent_node][1] + dy * 0.5)
        gate_pos[gate_id] = (pos[parent_node][0] + dx, pos[parent_node][1] + dy)
        gate_id += 1

    for n in [405, 406, 407, 408]: add_gate(n, -4, 4)
    for n in [401, 402, 403]: add_gate(n, -5, 0)
    for n in [301, 302, 303]: add_gate(n, 0, 5)
    for n in [601, 602, 603, 604]: add_gate(n, -5, 0)
    for n in [631, 632, 633, 634]: add_gate(n, 0, 5)
    for n in [621, 622, 623, 624]: add_gate(n, 5, 0)
    for n in [307, 308]: add_gate(n, 0, 5)
    for n in [501, 502, 503, 504, 505, 506]: add_gate(n, 5, 0)

    pos.update(stub_pos)
    pos.update(gate_pos)

    # 主骨架边列表（双向，热力图只统计无向频次）
    backbone_edges = [
        (100, 101), (101, 102), (102, 103), (103, 104), (104, 105),
        (200, 201), (201, 202), (202, 203), (203, 204), (204, 205),
        (101, 201), (102, 202), (103, 203), (104, 204),
        (201, 300), (202, 304), (203, 306), (204, 309),
        (300, 301), (301, 302), (302, 303), (303, 304),
        (304, 305), (305, 306),
        (306, 307), (307, 308), (308, 309),
        (300, 401), (401, 402), (402, 403), (403, 404),
        (404, 405), (405, 406), (406, 407), (407, 408), (408, 409),
        (309, 501), (501, 502), (502, 503), (503, 504), (504, 505), (505, 506), (506, 507),
        (304, 601), (601, 602), (602, 603), (603, 604), (604, 605),
        (305, 611), (611, 612),
        (306, 621), (621, 622), (622, 623), (623, 624), (624, 625),
        (605, 631), (631, 632), (632, 612), (612, 633), (633, 634), (634, 625),
        (900, 101), (104, 901),
    ]

    # Gate stubs 边
    gate_edges = []
    gid = 0
    def add_gate_edge(parent_node, dx, dy):
        nonlocal gid
        stub_id = 2000 + gid
        gate_edges.append((parent_node, stub_id))
        gate_edges.append((stub_id, gid))
        gid += 1

    for n in [405, 406, 407, 408]: add_gate_edge(n, -4, 4)
    for n in [401, 402, 403]: add_gate_edge(n, -5, 0)
    for n in [301, 302, 303]: add_gate_edge(n, 0, 5)
    for n in [601, 602, 603, 604]: add_gate_edge(n, -5, 0)
    for n in [631, 632, 633, 634]: add_gate_edge(n, 0, 5)
    for n in [621, 622, 623, 624]: add_gate_edge(n, 5, 0)
    for n in [307, 308]: add_gate_edge(n, 0, 5)
    for n in [501, 502, 503, 504, 505, 506]: add_gate_edge(n, 5, 0)

    return pos, backbone_edges, gate_edges


# =====================================================================
# 读取 per_agent CSV，提取 path_nodes 序列
# =====================================================================
def load_edge_counts_from_csv(csv_path: str):
    """从 per_agent.csv 统计各条边的使用次数（无向）。"""
    import csv
    import sys
    # 增加 CSV 字段大小限制，防止 path_nodes 过长导致报错
    # sys.maxsize 在部分平台超出 C long 范围，故用 try/except 兜底
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10 * 1024 * 1024)  # 10 MB 兜底
    
    edge_count = defaultdict(int)
    total_flights = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_flights += 1
            raw = row.get("path_nodes", "")
            if not raw or raw == "[]":
                continue
            try:
                # path_nodes 可能是字符串列表，如 "[100, 101, 102]"
                nodes = ast.literal_eval(raw)
                if not isinstance(nodes, list):
                    continue
                # 去重连续相同节点
                deduped = [nodes[0]] if nodes else []
                for n in nodes[1:]:
                    if n != deduped[-1]:
                        deduped.append(n)
                # 统计相邻节点对（无向：取排序后的pair）
                for i in range(len(deduped) - 1):
                    u, v = deduped[i], deduped[i + 1]
                    key = (min(u, v), max(u, v))
                    edge_count[key] += 1
            except Exception:
                continue

    print(f"[热力图] 读取 {total_flights} 架飞机轨迹，统计到 {len(edge_count)} 条有效边使用记录。")
    return edge_count


# =====================================================================
# 绘图
# =====================================================================
def plot_heatmap(edge_count, pos, backbone_edges, gate_edges, algo_name, out_path):
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # 归一化颜色映射
    max_count = max(edge_count.values()) if edge_count else 1
    cmap = plt.get_cmap("YlOrRd")

    def get_color_lw(u, v, is_gate=False):
        key = (min(u, v), max(u, v))
        cnt = edge_count.get(key, 0)
        norm = cnt / max_count
        if is_gate:
            if cnt == 0:
                return "#2a2a4a", 0.8
            return cmap(norm * 0.7), 1.2
        if cnt == 0:
            return "#2a2a4a", 0.6
        lw = 1.0 + norm * 8.0  # 最细1，最粗9
        return cmap(norm), lw

    # 绘制骨架边
    for u, v in backbone_edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color, lw = get_color_lw(u, v)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle='round', zorder=2)

        # 若使用频次 > 5% 最大值，标注次数
        key = (min(u, v), max(u, v))
        cnt = edge_count.get(key, 0)
        if cnt > max_count * 0.05:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, str(cnt), color="white", fontsize=5.5,
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.1', fc='#1a1a2e', alpha=0.5, ec='none'))

    # 绘制 Gate stubs
    for u, v in gate_edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color, lw = get_color_lw(u, v, is_gate=True)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle='--', zorder=1, alpha=0.7)

    # 绘制节点（只绘制骨架节点，不画 gate stubs）
    skeleton_nodes = set()
    for u, v in backbone_edges:
        skeleton_nodes.add(u)
        skeleton_nodes.add(v)

    for nid in skeleton_nodes:
        if nid not in pos:
            continue
        x, y = pos[nid]
        key_count = sum(
            edge_count.get((min(nid, nb), max(nid, nb)), 0)
            for nb in skeleton_nodes
        )
        if nid in (100, 105):  # 跑道入出口
            ax.plot(x, y, 'o', color='#00e5ff', markersize=9, zorder=5)
        elif nid in (900, 901):
            ax.plot(x, y, 's', color='#76ff03', markersize=7, zorder=5)
        else:
            ax.plot(x, y, 'o', color='#aaaacc', markersize=4, zorder=4, alpha=0.8)
        if nid >= 100:
            ax.text(x, y + 1.0, str(nid), color='#ccccee', fontsize=5,
                    ha='center', va='bottom', zorder=7)

    # 色条
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_count))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("边通过次数 (Edge Traversal Count)", color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title(f"Layout F 滑行道使用热力图 — {algo_name} (Baseline 场景)",
                 color='white', fontsize=16, pad=15)
    ax.set_xlabel("X 坐标 (拓扑单位)", color='#aaaacc')
    ax.set_ylabel("Y 坐标 (拓扑单位)", color='#aaaacc')
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#aaaacc')

    # 标注最热的前5条边
    top5 = sorted(edge_count.items(), key=lambda x: -x[1])[:5]
    for rank, ((u, v), cnt) in enumerate(top5):
        if u in pos and v in pos:
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            ax.annotate(f"#{rank+1} 最热\n{u}↔{v}\n({cnt}次)",
                        xy=(mx, my), xytext=(mx + 5, my + 3),
                        arrowprops=dict(arrowstyle='->', color='#ffff00'),
                        color='#ffff00', fontsize=7.5, zorder=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', ec='#ffff00', alpha=0.85))

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"[热力图] 已保存至: {out_path}")
    plt.show()


# =====================================================================
# 主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Layout F 滑行道使用热力图")
    parser.add_argument("--csv",    type=str, default="results/exp_4_2/MAPPO_per_agent.csv",
                        help="per_agent.csv 路径")
    parser.add_argument("--algo",   type=str, default="MAPPO",
                        help="算法名称（用于标题）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径（默认自动生成）")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[错误] 找不到文件: {args.csv}")
        print("请先运行 python run_paper_exp_4_2.py --scenario baseline 生成 per_agent.csv")
        sys.exit(1)

    out_path = args.output or f"results/heatmap_{args.algo}_baseline.png"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    pos, backbone_edges, gate_edges = build_layout_f_pos_edges()
    edge_count = load_edge_counts_from_csv(args.csv)
    plot_heatmap(edge_count, pos, backbone_edges, gate_edges, args.algo, out_path)


if __name__ == "__main__":
    main()
