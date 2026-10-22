"""
plot_closure_detour.py  (v2 - 清晰版)
======================================
每个子图只展示 2 条航班（各自用独立颜色对），
Baseline 用暖色实线，Closure 绕行用冷色虚线，
形成明确的视觉对比。
"""
import argparse, ast, sys, csv, os, random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import font_manager

# ── 字体 ──────────────────────────────────────────────────────────────
for _fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Layout F 拓扑坐标（与仿真一致） ──────────────────────────────────
def build_layout_f():
    y0, y5, y10 = 0, 5, 10
    xl, xal, xar, xr = -30, -12, 12, 30
    pos = {
        100: (xl-5, y0), 101: (xl, y0), 102: (xal, y0),
        103: (xar, y0),  104: (xr,  y0), 105: (xr+5, y0),
        200: (xl-5, y5), 201: (xl, y5),  202: (xal, y5),
        203: (xar, y5),  204: (xr,  y5), 205: (xr+5, y5),
        300: (xl,  y10), 301: (-27, y10), 302: (-20, y10),
        303: (-16, y10), 304: (xal, y10), 305: (0,   y10),
        306: (xar, y10), 307: (20,  y10), 308: (27,  y10), 309: (xr, y10),
        # 上层（近似）
        401: (xl-4, 22), 402: (xl-3, 26), 403: (xl-2, 30),
        501: (xr+2, 21), 502: (xr+2, 24), 503: (xr+2, 27), 504: (xr+2, 30),
        601: (xal-2, 22), 602: (xal-2, 26), 603: (xal-2, 30),
        611: (0,   25),  612: (0,   40),
        621: (xar, 16),  622: (xar, 22),  623: (xar, 28),
        624: (xar, 34),  625: (xar, 40),
        631: (-10, 40),  632: (-5,  40),  633: (5,   40),  634: (10,  40),
    }
    backbone = [
        (100,101),(101,102),(102,103),(103,104),(104,105),
        (200,201),(201,202),(202,203),(203,204),(204,205),
        (300,301),(301,302),(302,303),(303,304),(304,305),(305,306),(306,307),(307,308),(308,309),
        (101,201),(201,300),(102,202),(202,304),(103,203),(203,306),(104,204),(204,309),
        (100,200),(105,205),
        (304,601),(601,602),(602,603),
        (300,401),(401,402),(402,403),
        (309,501),(501,502),(502,503),(503,504),
        (305,611),(611,612),(612,631),(631,632),(632,612),(612,633),(633,634),(634,625),
        (306,621),(621,622),(622,623),(623,624),(624,625),
    ]
    return pos, backbone


def load_paths(csv_path):
    try: csv.field_size_limit(sys.maxsize)
    except OverflowError: csv.field_size_limit(10*1024*1024)
    paths = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = row.get("path_nodes", "")
            if not raw or raw in ("[]", ""): continue
            try: nodes = ast.literal_eval(raw)
            except: continue
            if isinstance(nodes, list) and len(nodes) > 1:
                aid = int(row["agent_id"])
                deduped = [nodes[0]]
                for n in nodes[1:]:
                    if n != deduped[-1]: deduped.append(n)
                paths[aid] = deduped
    return paths


def has_edge(path, u, v):
    for i in range(len(path)-1):
        if (path[i]==u and path[i+1]==v) or (path[i]==v and path[i+1]==u):
            return True
    return False


def path_to_xy(path, pos, ox=0, oy=0):
    xs, ys = [], []
    for n in path:
        if n in pos:
            xs.append(pos[n][0]+ox)
            ys.append(pos[n][1]+oy)
    return xs, ys


def draw_topo(ax, pos, backbone):
    for u, v in backbone:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#4a5568", lw=1.2, alpha=0.45, zorder=1)
    for nid, (x, y) in pos.items():
        ax.plot(x, y, 'o', color="#718096", ms=3, alpha=0.5, zorder=2)
        ax.text(x, y+0.8, str(nid), fontsize=5.5, color="#a0aec0",
                ha='center', va='bottom', alpha=0.75, zorder=3)


def draw_closed_edge(ax, pos, ce, label=""):
    u, v = ce
    if u in pos and v in pos:
        xs = [pos[u][0], pos[v][0]]
        ys = [pos[u][1], pos[v][1]]
        # 用红色锯齿感线条
        ax.plot(xs, ys, color="#fc4444", lw=6, alpha=0.9, zorder=4,
                solid_capstyle='round')
        ax.plot(xs, ys, color="#ff0000", lw=2, linestyle="--", alpha=0.6, zorder=5)
        mx, my = (xs[0]+xs[1])/2, (ys[0]+ys[1])/2
        ax.annotate(f"CLOSED\n{ce}", (mx, my), xytext=(mx, my+2.5),
                    color="#ff6666", fontsize=9, fontweight='bold', ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1a0000", ec="#ff4444", alpha=0.9),
                    arrowprops=dict(arrowstyle="-|>", color="#ff4444", lw=1.5),
                    zorder=12)


def draw_one_flight(ax, pos, bp, cp, color_base, color_detour, aid, offset_idx=0):
    """画单条航班的 Baseline(暖) vs Closure(冷)"""
    oy = offset_idx * 0.4   # 轻微Y偏移防止完全重叠

    # Baseline 虚线（半透明，退至背景）
    bxs, bys = path_to_xy(bp, pos, oy=oy)
    if len(bxs) >= 2:
        ax.plot(bxs, bys, color=color_base, lw=2, alpha=0.55, zorder=5,
                linestyle=(0, (5, 3)),
                path_effects=[pe.Stroke(linewidth=3.5, foreground='black', alpha=0.25),
                               pe.Normal()])
        ax.plot(bxs[0], bys[0], 'o', color=color_base, ms=7, zorder=8,
                alpha=0.6, markeredgecolor='white', markeredgewidth=0.8)
        ax.text(bxs[0]-0.5, bys[0]+1.2, f"#{aid}", color=color_base,
                fontsize=8, fontweight='bold', alpha=0.7, zorder=10)

    # Closure 绕行 实线（高亮、突出）
    if cp:
        cxs, cys = path_to_xy(cp, pos, oy=oy)
        if len(cxs) >= 2:
            ax.plot(cxs, cys, color=color_detour, lw=3.5, alpha=0.95, zorder=7,
                    solid_capstyle='round',
                    path_effects=[pe.Stroke(linewidth=6, foreground='black', alpha=0.4),
                                   pe.Normal()])
            ax.plot(cxs[0], cys[0], 's', color=color_detour, ms=9, zorder=9,
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.plot(cxs[-1], cys[-1], 'D', color=color_detour, ms=9, zorder=9,
                    markeredgecolor='white', markeredgewidth=1.2)


# 颜色对 (baseline_color, detour_color)
FLIGHT_COLOR_PAIRS = [
    ("#f6ad55", "#63b3ed"),   # 橙 vs 天蓝
    ("#68d391", "#fc8181"),   # 绿 vs 粉红
]


def plot_one_subplot(ax, pos, backbone, baseline_paths, closure_paths,
                     ce, title, n_flights=2):
    ax.set_facecolor("#0f1117")
    draw_topo(ax, pos, backbone)
    draw_closed_edge(ax, pos, ce)

    u, v = ce
    candidates = [aid for aid, p in baseline_paths.items()
                  if has_edge(p, u, v) and aid in closure_paths]
    affected = [aid for aid in candidates
                if not has_edge(closure_paths[aid], u, v)]

    print(f"  [{ce}] 受影响航班: {len(affected)} 架，展示 {min(n_flights, len(affected))} 架")

    rng = random.Random(99)
    samples = rng.sample(affected, min(n_flights, len(affected))) if affected else []

    legend_elements = [
        mpatches.Patch(color="#fc4444", label=f"关闭边 {ce}"),
    ]
    for idx, aid in enumerate(samples):
        cb, cd = FLIGHT_COLOR_PAIRS[idx % len(FLIGHT_COLOR_PAIRS)]
        bp = baseline_paths[aid]
        cp = closure_paths.get(aid, [])
        draw_one_flight(ax, pos, bp, cp, cb, cd, aid, offset_idx=idx)
        legend_elements.append(mpatches.Patch(color=cb, label=f"航班 #{aid}  原始路径（虚线）"))
        legend_elements.append(mpatches.Patch(color=cd, label=f"航班 #{aid}  封路绕行（实线）"))

    if not affected:
        ax.text(0, 20, "未发现受该边关闭影响\n且路径发生变化的航班\n（所有航班原路径已绕行此边）",
                color="#aaaacc", fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#446688"))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              facecolor="#0f1117", edgecolor="#446688", labelcolor="white",
              framealpha=0.9, borderpad=0.8)
    ax.set_title(title, color="white", fontsize=12, pad=10, fontweight='bold')
    ax.set_xlabel("X 坐标 (拓扑单位)", color="#8899aa", fontsize=9)
    ax.set_ylabel("Y 坐标 (拓扑单位)", color="#8899aa", fontsize=9)
    ax.tick_params(colors="#8899aa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d3748")
    ax.set_xlim(-38, 40)
    ax.set_ylim(-3, 45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/exp_4_2/MAPPO_per_agent.csv")
    parser.add_argument("--closure",  default="results/exp_4_3_closure/MAPPO_per_agent.csv")
    parser.add_argument("--algo",     default="MAPPO")
    parser.add_argument("--n",        type=int, default=2)
    parser.add_argument("--out",      default="results/exp_4_3_closure/closure_detour_{algo}.png")
    args = parser.parse_args()

    out = args.out.replace("{algo}", args.algo)
    print(f"[读取] Baseline: {args.baseline}")
    bp = load_paths(args.baseline)
    print(f"  → {len(bp)} 架")
    print(f"[读取] Closure:  {args.closure}")
    cp = load_paths(args.closure)
    print(f"  → {len(cp)} 架")

    pos, backbone = build_layout_f()

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor="#0f1117")
    fig.suptitle(
        f"滑行道关闭前后路径对比 ({args.algo})  ——  虚线 = 无封路原始路径，实线 = 封路绕行路径",
        color="white", fontsize=13, y=0.99, fontweight='bold'
    )

    plot_one_subplot(axes[0], pos, backbone, bp, cp,
                     ce=(102, 103),
                     title="第一次关闭: (102, 103)  底层横向主干道",
                     n_flights=args.n)
    plot_one_subplot(axes[1], pos, backbone, bp, cp,
                     ce=(201, 300),
                     title="第二次关闭: (201, 300)  左侧纵向主动脉",
                     n_flights=args.n)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    print(f"[保存] {out}")
    plt.show()


if __name__ == "__main__":
    main()
