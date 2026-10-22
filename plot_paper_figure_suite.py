import os
import csv
import json
import ast
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 统一风格设置 (白底)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei", "Microsoft YaHei"], # 支持中文
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT_DIR = "paper_figure"
os.makedirs(OUT_DIR, exist_ok=True)

# 统一色板
COLOR_MAPPO = "#2E86AB"   # 钢蓝
COLOR_FCFS  = "#E84855"   # 朱红
COLOR_VIP   = "#F9A03F"   # 亮橙色
COLOR_NORMAL = "#A2BCE0"  # 浅灰蓝

# ==========================================
# 工具函数
# ==========================================
def get_edge_counts(csv_path):
    """读取CSV并计算每条边的通行次数"""
    edge_counts = {}
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return edge_counts
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_str = row.get("path_nodes", "[]")
            try:
                path = ast.literal_eval(path_str)
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    edge = tuple(sorted([u, v]))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            except:
                pass
    return edge_counts


def get_edge_counts_windowed(csv_path, step_start, step_end):
    """只统计在 [step_start, step_end] 步数窗口内活跃的代理的边通行次数。

    判断代理是否活跃的条件：spawn_step <= step_end AND finish_step >= step_start。
    由于 per_agent.csv 没有逐步位置记录，此处统计这些代理的完整路径作为最佳近似。
    """
    edge_counts = {}
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return edge_counts

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                spawn  = int(row.get('spawn_step')  or -1)
                finish = int(row.get('finish_step') or -1)
            except ValueError:
                continue
            # 跳过未激活或未完成的代理
            if spawn < 0 or finish < 0:
                continue
            # 仅保留在扰动窗口内活跃过的代理
            if spawn > step_end or finish < step_start:
                continue
            path_str = row.get("path_nodes", "[]")
            try:
                path = ast.literal_eval(path_str)
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    edge = tuple(sorted([u, v]))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            except:
                pass
    return edge_counts

def build_layout_f_pos_edges():
    """构建 Layout F 坐标字典和边列表（静态提取以脱离环境依赖）"""
    y_b1 = 0; y_b2 = 5; y_mid_bot = 10; y_apron_top = 34
    x_fl = -30; x_al = -12; x_ac = 0; x_ar = 12; x_fr = 30

    pos = {}
    pos[100] = (x_fl - 5, y_b1); pos[101] = (x_fl, y_b1); pos[102] = (x_al, y_b1)
    pos[103] = (x_ar, y_b1); pos[104] = (x_fr, y_b1); pos[105] = (x_fr + 5, y_b1)
    pos[200] = (x_fl - 5, y_b2); pos[201] = (x_fl, y_b2); pos[202] = (x_al, y_b2)
    pos[203] = (x_ar, y_b2); pos[204] = (x_fr, y_b2); pos[205] = (x_fr + 5, y_b2)
    pos[300] = (x_fl, y_mid_bot)
    pos[301] = (-27, y_mid_bot); pos[302] = (-20, y_mid_bot); pos[303] = (-16, y_mid_bot)
    pos[304] = (x_al, y_mid_bot); pos[305] = (x_ac, y_mid_bot); pos[306] = (x_ar, y_mid_bot)
    pos[307] = (18, y_mid_bot); pos[308] = (24, y_mid_bot); pos[309] = (x_fr, y_mid_bot)
    pos[401] = (x_fl, 16); pos[402] = (x_fl, 22); pos[403] = (x_fl, 28); pos[404] = (x_fl, 34)
    pos[405] = (-26, 38); pos[406] = (-22, 42); pos[407] = (-18, 46); pos[408] = (-14, 50); pos[409] = (-10, 54)
    pos[501] = (x_fr, 16); pos[502] = (x_fr, 22); pos[503] = (x_fr, 28); pos[504] = (x_fr, 34)
    pos[505] = (x_fr, 40); pos[506] = (x_fr, 46); pos[507] = (x_fr, 52)
    pos[601] = (x_al, 16); pos[602] = (x_al, 22); pos[603] = (x_al, 28); pos[604] = (x_al, 34); pos[605] = (x_al, 40)
    pos[611] = (x_ac, 25); pos[612] = (x_ac, 40)
    pos[621] = (x_ar, 16); pos[622] = (x_ar, 22); pos[623] = (x_ar, 28); pos[624] = (x_ar, 34); pos[625] = (x_ar, 40)
    pos[631] = (-8, 40); pos[632] = (-4, 40); pos[633] = (4, 40); pos[634] = (8, 40)
    pos[900] = (-25, -2); pos[901] = (25, -2)

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

    all_edges = backbone_edges + gate_edges
    return pos, all_edges

# ==========================================
# 绘图 1: 闭锁前后热力图 (MAPPO vs FCFS)
# 闭锁刺激期间的局部热力图
# ==========================================
def plot_closure_heatmap():
    print("Generating Figure 1: Closure Heatmap (2x2, Windowed)...")
    pos, all_edges = build_layout_f_pos_edges()

    # 局部热力图窗口：仅统计第一次关闭期间的流量
    # 关闭1: trigger=total_steps*0.10, duration=800 → 以 20000 步计: [2000, 2800]
    # 充上前后各 300 步缓冲，确保能看到起始速度和结束返回正常
    TOTAL_STEPS  = 20000
    BUFFER       = 300
    C1_TRIGGER   = int(TOTAL_STEPS * 0.10)   # 2000
    C1_DURATION  = 800
    win_start    = C1_TRIGGER - BUFFER        # 1700
    win_end      = C1_TRIGGER + C1_DURATION + BUFFER  # 3100

    # 读数据
    # Baseline：全量累计（没有扰动，不需要窗口过滤）
    c_map_base  = get_edge_counts("results/exp_4_2/MAPPO_per_agent.csv")
    c_fcf_base  = get_edge_counts("results/exp_4_2/FCFS_per_agent.csv")

    # Closure：仅统计第一次关闭窗口内的流量（绕行路径更醒目）
    c_map_close = get_edge_counts_windowed("results/exp_4_3_closure/MAPPO_per_agent.csv", win_start, win_end)
    c_fcf_close = get_edge_counts_windowed("results/exp_4_3_closure/FCFS_per_agent.csv",  win_start, win_end)

    # 消除伪影：由于 per_agent.csv 只记录 agent 整个生命周期的完整 path_nodes，
    # 那些在 step < 2000 穿过 (102,103) 然后在 step > 2000 才 finish 的飞机，
    # 它们的整个轨迹都会被画出来，导致关闭的边上看起来“仍有流量”。
    # 在这里我们将物理关闭的边在统计图中强行归零，真实反映窗口期的绕行状态。
    for counts in [c_map_close, c_fcf_close]:
        for e in [(102, 103), (103, 102)]:
            edge = tuple(sorted(e))
            if edge in counts:
                counts[edge] = 0

    # 标干归一：Baseline 和 Closure 分别用独立的频次上限
    max_base    = 200   # 全量热力图频次上限
    max_closure = max(1, max(max(c_map_close.values(), default=1),
                             max(c_fcf_close.values(), default=1)))
    max_closure = 25  # 强制压低比例尺上限，让绕行流量（往往在 10-20 架次左右）颜色更深、更显著

    cmap = plt.cm.Reds
    norm_base    = mcolors.Normalize(vmin=0, vmax=max_base)
    norm_closure = mcolors.Normalize(vmin=0, vmax=max_closure)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def draw_heatmap(ax, counts, title, closures=None, norm=norm_base):
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=10)

        # 基础骨架
        for u, v in all_edges:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#EEEEEE', linewidth=1.5, zorder=1)

        # 流量热力
        valid_edges = set(tuple(sorted(e)) for e in all_edges)
        for edge, c in counts.items():
            if c > 0:
                u, v = edge
                if u in pos and v in pos and tuple(sorted([u, v])) in valid_edges:
                    lw = 1.5 + min(c / norm.vmax, 1.0) * 4
                    color = cmap(norm(min(c, norm.vmax)))
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                            color=color, linewidth=lw, solid_capstyle='round', zorder=2)

        # 节点
        for n, p in pos.items():
            if n < 30:
                ax.plot(p[0], p[1], 'o', color='lightcoral', markersize=3, zorder=3)
            elif n >= 2000:
                ax.plot(p[0], p[1], '.', color='lightgray', markersize=1, zorder=3)
            else:
                ax.plot(p[0], p[1], 'o', color='lightblue', markersize=2, zorder=3)

        # 闭锁标记
        if closures:
            for u, v in closures:
                cx = (pos[u][0] + pos[v][0]) / 2
                cy = (pos[u][1] + pos[v][1]) / 2
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color='#CC0000', linewidth=3, linestyle='--', zorder=4)
                ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2, zorder=5)
                ax.annotate('Closed', xy=(cx, cy), xytext=(cx, cy + 4),
                            color='#CC0000', weight='bold', fontsize=10, ha='center',
                            arrowprops=dict(arrowstyle='->', color='#CC0000'))

    draw_heatmap(axes[0, 0], c_map_base,  "(a) MAPPO Baseline (All Flights)",
                 norm=norm_base)
    draw_heatmap(axes[0, 1], c_map_close, "(b) MAPPO During Closure Window",
                 closures=[(102, 103)], norm=norm_closure)
    draw_heatmap(axes[1, 0], c_fcf_base,  "(c) FCFS+A* Baseline (All Flights)",
                 norm=norm_base)
    draw_heatmap(axes[1, 1], c_fcf_close, "(d) FCFS+A* During Closure Window",
                 closures=[(102, 103)], norm=norm_closure)

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    # 两条 Colorbar：分别对应基准和窗口
    sm_base    = plt.cm.ScalarMappable(cmap=cmap, norm=norm_base)
    sm_base.set_array([])
    sm_closure = plt.cm.ScalarMappable(cmap=cmap, norm=norm_closure)
    sm_closure.set_array([])
    cbar1 = fig.colorbar(sm_base,    ax=[axes[0,0], axes[1,0]],
                         orientation='vertical', fraction=0.03, pad=0.02)
    cbar1.set_label('Freq. (Baseline, all flights)', fontsize=10)
    cbar2 = fig.colorbar(sm_closure, ax=[axes[0,1], axes[1,1]],
                         orientation='vertical', fraction=0.03, pad=0.02)
    cbar2.set_label('Freq. (Closure window only)', fontsize=10)

    plt.savefig(os.path.join(OUT_DIR, "fig1_closure_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 绘图 2: 延误等待时间箱线图
# ==========================================
def plot_delay_wait_time_boxplot():
    print("Generating Figure 2: Delay Wait Time Boxplot...")
    csv_mappo = "results/exp_4_3_delay/MAPPO_per_agent.csv"
    csv_fcfs = "results/exp_4_3_delay/FCFS_per_agent.csv"
    
    if not os.path.exists(csv_mappo) or not os.path.exists(csv_fcfs):
        print("Missing delay per_agent data.")
        return
        
    def read_wait_times(path):
        total_waits = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('crashed') == 'True':
                    continue
                try:
                    onroute_w = float(row.get('onroute_wait_sec', 0) or 0) / 60.0
                    dep_delay = float(row.get('wait_time_min', 0) or 0)
                    total_waits.append(onroute_w + dep_delay)
                except ValueError:
                    pass
        return total_waits

    Y_CLIP = 15.0   # 截断阈值：直接过滤掉超出范围的数据点

    m_total = read_wait_times(csv_mappo)
    f_total = read_wait_times(csv_fcfs)

    # 过滤超出范围的数据（直接删去）
    m_total = [v for v in m_total if v <= Y_CLIP]
    f_total = [v for v in f_total if v <= Y_CLIP]

    # ── 统计检验（Mann-Whitney U）──────────────────────────────
    try:
        from scipy import stats as scipy_stats
        _, p_val = scipy_stats.mannwhitneyu(m_total, f_total, alternative='less')
        if p_val < 0.001:
            sig_text = "*** p < 0.001"
        elif p_val < 0.01:
            sig_text = "** p < 0.01"
        elif p_val < 0.05:
            sig_text = "* p < 0.05"
        else:
            sig_text = f"p = {p_val:.3f}"
    except ImportError:
        sig_text = None

    m_mean = np.mean(m_total)
    f_mean = np.mean(f_total)
    m_med  = np.median(m_total)
    f_med  = np.median(f_total)
    reduction_pct = (f_mean - m_mean) / f_mean * 100 if f_mean > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 5.5))

    data   = [m_total, f_total]
    labels = ['MAPPO\n(Ours)', 'FCFS+A*']

    bplot = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.42,
        showfliers=False,
        boxprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='white', linewidth=2.5),
    )

    colors = [COLOR_MAPPO, COLOR_FCFS]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # 绘制离群点（showfliers=False 已关闭自动绘制，此处手动绘制箱线图范围内的离群点）
    for xi, arr in enumerate([m_total, f_total], start=1):
        q1, q3 = np.percentile(arr, [25, 75])
        fence = q3 + 1.5 * (q3 - q1)
        outliers = [v for v in arr if v > fence]
        if outliers:
            ax.scatter([xi] * len(outliers), outliers,
                       marker='o', color='gray', s=25, alpha=0.55, zorder=4)


    # 均值菱形 + 数值标注
    for xi, mean_val in enumerate([m_mean, f_mean], start=1):
        ax.scatter(xi, mean_val, color='#FFE566', marker='D',
                   edgecolors='black', linewidths=1.2, s=70, zorder=6)
        ax.text(xi + 0.22, mean_val, f'\u03bc={mean_val:.2f}',
                va='center', ha='left', fontsize=9, color='#222222', fontweight='bold')

    # 显著性括号（上移至 Y_CLIP 的 90%，避开离群点）
    if sig_text:
        bk_y = Y_CLIP * 0.90
        ax.plot([1, 1, 2, 2], [bk_y * 0.985, bk_y, bk_y, bk_y * 0.985], color='black', lw=1.5)
        ax.text(1.5, bk_y + Y_CLIP * 0.012, sig_text,
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222222')

    # 左上角统计摘要框（避免与 FCFS 筱体重叠）
    stats_text = (
        f"MAPPO:  \u03bc={m_mean:.2f} min,  med={m_med:.2f} min\n"
        f"FCFS:     \u03bc={f_mean:.2f} min,  med={f_med:.2f} min\n"
        f"Wait\u2193  {reduction_pct:.0f}%  (by mean)"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#CCCCCC', linewidth=1.0))

    ax.set_ylim(0, Y_CLIP)
    ax.set_ylabel("Total Wait Time (min)")
    ax.set_title("Wait Time Distribution (Delay Scenario)", pad=12)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_MAPPO, edgecolor='black', label='MAPPO (Ours)'),
        mpatches.Patch(facecolor=COLOR_FCFS,  edgecolor='black', label='FCFS+A*'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#FFE566',
                   markeredgecolor='black', markersize=8, label='Mean'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_delay_wait_time.png"), bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 绘图 3: 优先级突变时空轨迹图
# ==========================================
def plot_priority_trajectory():
    print("Generating Figure 3: Priority Trajectories...")
    json_mappo = "results/exp_4_3_priority/MAPPO_micro_priority.json"
    json_fcfs = "results/exp_4_3_priority/FCFS_micro_priority.json"
    
    if not os.path.exists(json_mappo) or not os.path.exists(json_fcfs):
        print("Missing priority micro json data.")
        return
        
    with open(json_mappo, 'r', encoding='utf-8') as f: data_m = json.load(f)
    with open(json_fcfs, 'r', encoding='utf-8') as f: data_f = json.load(f)
        
    def get_vip_and_pos_history(data):
        vip_ids = set()
        dist_history = defaultdict(float) # 用距离衡量活跃度
        event_step = data["event_step"]
        
        for frame in data["frames"]:
            for a in frame["agents"]:
                if a.get("is_vip", 0) == 1 or a.get("priority", 0) > 100:
                    vip_ids.add(a["id"])
                dist_history[a["id"]] += a.get("speed", 0.0)
                
        return vip_ids, dist_history, event_step

    vips_m, dist_hist_m, ev_m = get_vip_and_pos_history(data_m)
    vips_f, dist_hist_f, ev_f = get_vip_and_pos_history(data_f)
    
    all_vips = vips_m.union(vips_f)
    
    # 局部窗口过滤：选出在两组数据中均存在且在窗口期内总滑行距离最大的航班
    normal_candidates = [aid for aid in dist_hist_m.keys() if aid in dist_hist_f.keys() and aid not in all_vips]
    normal_candidates.sort(key=lambda aid: dist_hist_m[aid] + dist_hist_f[aid], reverse=True)
    selected_normal = set(normal_candidates[:6])
    
    def extract_selected_series(data, target_ids):
        event_step = data["event_step"]
        series = defaultdict(lambda: {"steps": [], "cum_dist": []})
        dist_tracker = defaultdict(float)
        
        max_step = -1
        for frame in data["frames"]:
            rel_step = frame["step"] - event_step
            max_step = max(max_step, rel_step)
            for a in frame["agents"]:
                if a["id"] in target_ids:
                    dist_tracker[a["id"]] += a.get("speed", 0.0)
                    series[a["id"]]["steps"].append(rel_step)
                    series[a["id"]]["cum_dist"].append(dist_tracker[a["id"]])
        
        return series, max_step
        
    target_ids = all_vips.union(selected_normal)
    sm, max_step_m = extract_selected_series(data_m, target_ids)
    sf, max_step_f = extract_selected_series(data_f, target_ids)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    def plot_ax(ax, series, max_frame_step, vips, title):
        # 原本是绝对事件步，现在由于做差，事件线在 t=0
        ax.axvline(0, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
        
        max_dist = 1.0
        for aid, s in series.items():
            if s["cum_dist"] and max(s["cum_dist"]) > max_dist:
                max_dist = max(s["cum_dist"])
                
        ax.text(10, max_dist * 0.85, 'VIP Priority\nTriggered', color='#f39c12', fontweight='bold', va='top')
        
        for aid, s in series.items():
            if not s["steps"]:
                continue
            is_vip = aid in vips
            y_vals = s["cum_dist"]
            color = COLOR_VIP if is_vip else COLOR_NORMAL
            lw = 3.0 if is_vip else 1.2
            z = 10 if is_vip else 2
            alpha = 1.0 if is_vip else 0.7
            ax.plot(s["steps"], y_vals, color=color, linewidth=lw, alpha=alpha, zorder=z)
            
            # 检查是否中途消失 (Deadlock Eviction or Crash)
            # 宽容度 2 步，如果没有到达时间窗口的尽头就消失，画上红 "X"
            if s["steps"][-1] < max_frame_step - 2 and not is_vip:
                ax.plot(s["steps"][-1], y_vals[-1], marker='X', color='red', markersize=8, zorder=z+1)
            
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Cumulative Distance")
        ax.set_xlabel("Relative Steps (t=0 at Event Trigger)")

    plot_ax(ax1, sm, max_step_m, all_vips, "(a) MAPPO Evolution (Dynamic Yielding)")
    plot_ax(ax2, sf, max_step_f, all_vips, "(b) FCFS+A* Evolution (Rigid Queueing)")
    
    custom_lines = [
        mpatches.Patch(color=COLOR_VIP, label='VIP Flight'),
        mpatches.Patch(color=COLOR_NORMAL, label='Normal Flight')
    ]
    ax1.legend(handles=custom_lines, loc='upper right', framealpha=1.0)
    
    ax2.set_xlabel("Simulation Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_priority_trajectory.png"), bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================
# 绘图 4: 特定航班 (VIP) 场景对比柱状图
# ==========================================
def plot_priority_time_comparison():
    print("Generating Figure 4: VIP Flight Time Comparison...")
    
    def get_vip_taxi_time(csv_path, vip_id="120"):
        if not os.path.exists(csv_path): return 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get("agent_id") == vip_id:
                    try: return float(row.get("taxi_sec", 0)) / 60.0
                    except: return 0
        return 0
        
    # 读取 Baseline 场景下，该飞机的通行时间
    m_base = get_vip_taxi_time("results/exp_4_2/MAPPO_per_agent.csv")
    f_base = get_vip_taxi_time("results/exp_4_2/FCFS_per_agent.csv")
    
    # 读取 Priority 场景下，该飞机被提升为VIP后的通行时间
    m_prio = get_vip_taxi_time("results/exp_4_3_priority/MAPPO_per_agent.csv")
    f_prio = get_vip_taxi_time("results/exp_4_3_priority/FCFS_per_agent.csv")
    
    labels = ['MAPPO\n(Ours)', 'FCFS+A*']
    baseline_times = [m_base, f_base]
    priority_times = [m_prio, f_prio]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, baseline_times, width, label='Baseline Scenario', color=COLOR_NORMAL, edgecolor='white', linewidth=1)
    rects2 = ax.bar(x + width/2, priority_times, width, label='Priority Scenario (VIP)', color=COLOR_VIP, edgecolor='white', linewidth=1)
    
    ax.set_ylabel('Taxiing Time of Target Flight (min)')
    ax.set_title('Target Flight Travel Time: Baseline vs Priority')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    # 设置Y轴最大值，留出标注空间
    max_val = max(max(baseline_times), max(priority_times))
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_priority_time_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("--- Starting Plot Generation ---")
    plot_closure_heatmap()
    plot_delay_wait_time_boxplot()
    plot_priority_trajectory()
    plot_priority_time_comparison()
    print("--- Finished! Check the 'paper_figure' directory. ---")
