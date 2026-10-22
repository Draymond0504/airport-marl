"""训练日志分析脚本。

用法: python analyze_log.py [log_path]
默认读取 checkpoints/training_log_improved.csv
"""
import csv
import sys

import numpy as np

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/training_log_improved.csv"

rows = list(csv.DictReader(open(LOG_PATH, encoding="utf-8")))
total = len(rows)
print(f"日志文件: {LOG_PATH}")
print(f"总记录数: {total} episodes\n")


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def window_avg(rows_slice, key):
    vals = [safe_float(r.get(key)) for r in rows_slice]
    return np.mean(vals) if vals else 0.0


# ---- 分阶段统计 ----
print(f"{'阶段':<18} {'reward':>9} {'comp':>8} {'team':>8} {'deadlock':>9} {'conflict':>9} {'prox':>8}")
print("-" * 78)

stages = [
    ("初始(0-1k)",       0,     1000),
    ("1k-5k",          1000,   5000),
    ("5k-10k",         5000,  10000),
    ("10k-20k",       10000,  20000),
    ("20k-30k",       20000,  30000),
    ("30k-40k",       30000,  40000),
    ("40k-50k",       40000,  50000),
    ("最后1k",          max(0,total-1000), total),
]

for label, s, e in stages:
    if s >= total:
        continue
    sl = rows[s:min(e,total)]
    r   = window_avg(sl, "total_reward")
    c   = window_avg(sl, "completion")
    t   = window_avg(sl, "team_bonus")
    d   = window_avg(sl, "deadlock")
    cf  = window_avg(sl, "conflict")
    px  = window_avg(sl, "proximity")
    print(f"{label:<18} {r:>9.2f} {c:>8.2f} {t:>8.2f} {d:>9.2f} {cf:>9.2f} {px:>8.2f}")

# ---- 真实完成率估算 ----
# team_bonus 机制说明：全队全员无碰撞完成时，"每个" agent 各发一次 team_bonus（非全队共享一次）
# 因此：
#   COMP_R  = 50.0  （improved_v2.yaml 中 completion: 50.0）
#   TEAM_R  = 100.0 × NUM_AGENTS = 400  （4人每人各得100，CSV 中该列=总和）
# 注意：如果 p_agent > 100%，说明 team_bonus 已混入 completion 列的汇总，属于统计误差，
#       以 test.py 输出的实际完成率为准
COMP_R = 50.0   # completion reward per agent（改自旧值30，与yaml config匹配）
NUM_AGENTS = 4  # 根据配置修改
TEAM_R = 100.0 * NUM_AGENTS  # 全队总 team_bonus（每人各领100，4人共领400）

last = rows[max(0, total-1000):]
avg_comp = window_avg(last, "completion")
avg_team = window_avg(last, "team_bonus")
avg_dl   = window_avg(last, "deadlock")
avg_conf = window_avg(last, "conflict")

# 单 agent 完成率：avg_comp / (每人最大可能完成奖励)
# 若 > 100% 说明 team_bonus 溢出进了统计，以 test.py 输出的完成率为权威值
p_agent = min(avg_comp / (COMP_R * NUM_AGENTS), 1.0) if COMP_R * NUM_AGENTS > 0 else 0
p_team  = avg_team / TEAM_R if TEAM_R > 0 else 0

print(f"\n--- 最后 {len(last)} 轮 真实完成率估算 ---")
print(f"  单agent完成率(近似):  {p_agent*100:.1f}")
print(f"  全队成功率(实测):     {p_team*100:.1f}]")
print(f"  理论全队率(p^n):      {(p_agent**NUM_AGENTS)*100:.1f}]")
print(f"  avg deadlock:         {avg_dl:.2f}")
print(f"  avg conflict:         {avg_conf:.2f}")

# ---- 趋势分析 (每5000轮一个点) ----
print(f"\n--- 趋势曲线 (每{min(5000,total//5)}轮) ---")
window = min(5000, max(total//5, 100))
print(f"{'epoch':>8} {'reward':>9} {'comp':>8} {'team':>8} {'deadlock':>9}")
for i in range(0, total, window):
    sl = rows[i:min(i+window, total)]
    print(f"{i:>8} {window_avg(sl,'total_reward'):>9.2f} {window_avg(sl,'completion'):>8.2f} {window_avg(sl,'team_bonus'):>8.2f} {window_avg(sl,'deadlock'):>9.2f}")

# ---- 绘图部分 ----
try:
    import matplotlib.pyplot as plt
    
    # 提取序列
    epochs = [safe_float(r.get("episode", i)) for i, r in enumerate(rows)]
    rewards = [safe_float(r.get("total_reward")) for r in rows]
    comps = [safe_float(r.get("completion")) for r in rows]
    conflicts = [safe_float(r.get("conflict")) for r in rows]
    proxs = [safe_float(r.get("proximity")) for r in rows]
    
    # 平滑处理函数
    def smooth(scalars, weight=0.9):
        if not scalars: return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    sm_rewards = smooth(rewards, 0.95)
    sm_comps = smooth(comps, 0.95)
    sm_conflicts = smooth(conflicts, 0.95)
    sm_proxs = smooth(proxs, 0.95)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, rewards, alpha=0.3, color='blue', label='raw')
    plt.plot(epochs, sm_rewards, color='blue', linewidth=2, label='smoothed')
    plt.title('Total Reward')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, comps, alpha=0.3, color='green')
    plt.plot(epochs, sm_comps, color='green', linewidth=2)
    plt.title('Completion Reward')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, conflicts, alpha=0.3, color='red')
    plt.plot(epochs, sm_conflicts, color='red', linewidth=2)
    plt.title('Conflict Penalty')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, proxs, alpha=0.3, color='purple')
    plt.plot(epochs, sm_proxs, color='purple', linewidth=2)
    plt.title('Proximity Reward')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = 'training_curve_improved.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n[绘图完成] 训练曲线已生成并保存至: {save_path}")
    
except ImportError:
    print("\n[提示] 未安装 matplotlib，无法生成折线图。如果不需绘图可忽略，或执行 pip install matplotlib。")
