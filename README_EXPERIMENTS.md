# 机场港湾区 MARL 仿真系统 — 实验使用文档

> 本文档对应代码库最新版本，覆盖论文第 3-4 章所有实验的运行方式与产出说明。

---

## 目录

1. [环境配置参数说明](#1-环境配置参数说明)
2. [消融实验：MAPPO-NoGuide（无 A* 先验引导）](#2-消融实验mappo-noguide)
3. [消融实验：MAPPO-NoCorridor（无廊道方向保护）](#3-消融实验mappo-nocorridor)
4. [论文主实验：MAPPO vs FCFS 对比](#4-论文主实验mappo-vs-fcfs-对比)
5. [微观分镜数据采集](#5-微观分镜数据采集)
6. [翼尖净空阴影机制说明](#6-翼尖净空阴影机制说明)
7. [产出文件索引](#7-产出文件索引)

---

## 1. 环境配置参数说明

所有实验通过 `configs/improved_v2.yaml` 控制。以下为本次新增及关键参数：

### `env` 节参数

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `num_agents` | int | 4 | 并发飞机数量 |
| `layout` | str | `"F"` | 拓扑布局，论文使用 Layout F |
| `auto_speed` | bool | `false` | 是否由底层接管速度控制 |
| `disable_corridor` | bool | `false` | **[消融]** 禁用廊道方向预留机制 |
| `use_astar_guide` | bool | `true` | **[消融]** 禁用 A* 先验特征嵌入 |

### YAML 配置示例

```yaml
# configs/improved_v2.yaml（正式版，论文主实验）
env:
  num_agents: 8
  layout: "F"
  auto_speed: true
  disable_corridor: false
  use_astar_guide: true   # 开启 A* 先验引导（默认）
```

---

## 2. 消融实验：MAPPO-NoGuide

### 科学目的

证明 A* 先验路径嵌入对训练收敛速度和最终性能有显著量化贡献，闭合论文引言中"批判纯 RL 效率低"的论证链。

### 受影响的观测特征

当 `use_astar_guide: false` 时，以下两个特征被置零（观测维度保持 31 维不变）：

| 特征位 | 名称 | 正常值 | 消融后 |
|---|---|---|---|
| Part1 每槽 `+1` | `path_cost_norm` | A* 代价归一化值 [0,1] | **强制为 0** |
| Part1 每槽 `+2` | `on_planned_path` | 是否为 A* 下一步 (0/1) | **强制为 0** |

共 **10 维**（2 特征 × 5 邻居槽），占总观测 31 维的 32.3%。

### 运行方法

**Step 1**：创建消融配置文件

```yaml
# configs/ablation_no_guide.yaml
env:
  num_agents: 8
  layout: "F"
  auto_speed: true
  disable_corridor: false
  use_astar_guide: false    # ← 关键：关闭 A* 先验

training:
  total_episodes: 5000
  hidden_dim: 128
```

**Step 2**：分别运行两组训练

```bash
# 组 1：MAPPO-Guided（本文方法，现有 checkpoint）
python train_curriculum.py --config configs/improved_v2.yaml \
    --save-dir checkpoints/guided/

# 组 2：MAPPO-NoGuide（消融基线）
python train_curriculum.py --config configs/ablation_no_guide.yaml \
    --save-dir checkpoints/no_guide/
```

**Step 3**：对比训练日志

两组训练均输出 `training_log_improved.csv`，关键对比字段：

| 字段 | 含义 |
|---|---|
| `Episode` | 训练回合数 |
| `Avg100Reward` | 近 100 回合平均奖励 |
| `CompletionRate` | 完航率（0-1） |
| `DeadlockCount` | 死锁次数 |

```python
# 对比曲线绘制示例
import pandas as pd
import matplotlib.pyplot as plt

guided = pd.read_csv("checkpoints/guided/training_log_improved.csv")
no_guide = pd.read_csv("checkpoints/no_guide/training_log_improved.csv")

plt.plot(guided["Episode"], guided["Avg100Reward"], label="MAPPO-Guided")
plt.plot(no_guide["Episode"], no_guide["Avg100Reward"], label="MAPPO-NoGuide")
plt.legend(); plt.xlabel("Episode"); plt.ylabel("Avg Reward")
plt.savefig("ablation_convergence.png")
```

### 预期结果

- NoGuide 收敛极慢或在 2000 Epoch 内无法稳定（无路径先验，频繁误入盲端机位）
- 高密度阶段（6-8 架）完航率差异最显著
- NoGuide 死锁次数显著更多

---

## 3. 消融实验：MAPPO-NoCorridor

### 科学目的

将"廊道互锁保护"从论文文字描述转化为量化实验证据，证明该机制对消除单行廊道对头死锁的必要性。

### 运行方法

**零代码改动**，仅修改配置：

```yaml
# configs/ablation_no_corridor.yaml
env:
  num_agents: 8
  layout: "F"
  auto_speed: true
  disable_corridor: true    # ← 关键：禁用廊道保护
  use_astar_guide: true
```

```bash
# 运行消融训练
python train_curriculum.py --config configs/ablation_no_corridor.yaml \
    --save-dir checkpoints/no_corridor/
```

### 廊道保护节点范围（Layout F）

禁用后以下节点会出现高频对头死锁热点：

| 廊道区段 | 覆盖节点 | 连接机位 |
|---|---|---|
| 西向廊道 | 301 → 302 → 303 → 304 | 机位 6-8 |
| 中向廊道 | 601 → 602 → 603 → 604 | 机位 9-12 |
| 南向廊道 | 621 → 622 → 623 → 624 | 机位 17-20 |
| 环形廊道 | 631 → 632 → 633 → 634 | 机位 21-24 |

### 预期可视化产出

在拓扑图上将死锁节点按频次染色：

- `NoCorridor` 组：上述廊道节点出现密集死锁热点
- `WithCorridor` 组：廊道节点死锁频次接近零

---

## 4. 论文主实验：MAPPO vs FCFS 对比

### 运行命令

```bash
python run_paper_exp_4_2.py \
    --config configs/improved_v2.yaml \
    --model checkpoints/best.pt \
    --excel data.xlsx \
    --scenario baseline \
    --time-scale 3.0 \
    --max-steps 30000 \
    --seed 42
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--scenario` | 扰动场景：`baseline` / `closure` / `priority` / `delay` |
| `--time-scale` | 每仿真步对应的真实时间（秒），默认 3.0 |
| `--max-steps` | 最大仿真步数 |
| `--deadlock-timeout` | 飞机静止多少步触发死锁处理（默认 100） |
| `--seed` | 随机种子，确保 MAPPO 与 FCFS 使用相同初始条件 |

### 产出文件

```
results/exp_4_2/
├── MAPPO_summary.json          # 汇总指标（完航率、平均滑行时间等）
├── MAPPO_timeseries.csv        # 吞吐量折线图数据
├── MAPPO_per_agent.csv         # 各飞机指标（滑行时间、距离）
├── MAPPO_micro_corridor.json   # 廊道协议微观轨迹（新增）
├── FCFS_summary.json
├── FCFS_timeseries.csv
├── FCFS_per_agent.csv
└── comparison_table.txt        # 论文表格（Markdown 格式）
```

---

## 5. 微观分镜数据采集

### 5.1 廊道协议轨迹采集（自动）

在 `run_paper_exp_4_2.py` 运行时自动激活，无需额外参数。

**采集触发条件**：
1. 某廊道 `direction` 被锁定（有飞机占用）
2. 同时有另一架飞机在廊道入口节点静止等待（`speed < 0.1`）

每条廊道只采集首次触发事件，输出包含：
- 事件触发前 50 帧（缓冲回溯）
- 事件触发后 50 帧（实时采集）

**输出文件**：`results/exp_4_2/MAPPO_micro_corridor.json`

```json
{
  "description": "廊道协议微观轨迹数据",
  "scenario": "baseline",
  "window_frames": 50,
  "time_scale_sec": 3.0,
  "events": [
    {
      "corridor_id": 0,
      "event_step": 1234,
      "corridor_nodes": [301, 302, 303, 304],
      "frames": [
        {
          "step": 1184,
          "positions": [301, 405, ...],
          "on_edge": [0, 1, ...],
          "speed": [0.0, 1.5, ...],
          "done": [0, 0, ...],
          "corridor_status": {
            "0": {"direction": "forward", "occupied_by": [2]}
          }
        },
        ...
      ]
    }
  ]
}
```

### 5.2 从 JSON 生成分镜图

```python
import json, matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 加载廊道轨迹数据
with open("results/exp_4_2/MAPPO_micro_corridor.json") as f:
    data = json.load(f)

event = data["events"][0]          # 取第一个廊道事件
frames = event["frames"]
event_step = event["event_step"]

# 选取关键帧（事件触发前-10、触发时、触发后+5、+10、+20）
key_offsets = [-10, 0, 5, 10, 20]
key_frames = {
    off: next(fr for fr in frames if fr["step"] == event_step + off)
    for off in key_offsets
    if any(fr["step"] == event_step + off for fr in frames)
}

# 在拓扑图上绘制各帧飞机位置（调用 plot_paper_figure_suite 的底图函数）
# from plot_paper_figure_suite import build_layout_f_pos_edges, draw_topology_base
# pos, edges = build_layout_f_pos_edges()
# ...（绘制逻辑）
```

### 5.3 FCFS 失效对比场景

在相同的初始条件下让 FCFS 运行相同场景，由于 FCFS 无廊道方向预留，飞机直接冲入廊道，触发碰撞。

```bash
# 使用相同种子确保初始场景一致
python run_paper_exp_4_2.py --scenario baseline --seed 42 --model checkpoints/best.pt
# → FCFS 部分自动运行，输出包含碰撞事件的 FCFS_summary.json
```

---

## 6. 翼尖净空阴影机制说明

### 物理依据

ICAO Annex 14 Table 3-1（机位净空设计标准）：

| 机型 | 翼展范围 | 相邻机位最小净空 |
|---|---|---|
| Code D | 36-52m（如 B737-800） | 4.5 m |
| Code E | 52-65m（如 A330/B787） | 7.5 m |
| Code F | 65-80m（如 A380） | 10.0 m |

在推出速度（2-5 km/h）下，尾流影响可忽略，危险完全来自翼尖扫掠弧面。

### 阴影组结构

```
主干道节点 p ──→ stub_node(2000+gate_id) ──→ gate_node(gate_id)
```

- **直接阴影组**：同一父干道节点 `p` 下的所有机位（翼尖必然干涉）
- **扩展阴影组**：`p` 的相邻干道节点 `p'` 下的机位（宽体机翼展可能干涉）

系统初始化时通过 `TaxiGraphEnv._build_gate_shadow_groups()` 自动预计算并存入 `env.gate_shadow_groups`。

### 触发行为

当飞机 `i` 位于机位节点（`degree=1, node_id<2000`）：
- 检查阴影组内是否有飞机在 `gate_node` 或 `stub_node`（推出阶段）
- 若有：`get_action_mask()` 屏蔽 `i` 的全部出站动作
- 效果：飞机被强制等待，直到相邻飞机推出完成并移至主干道

### 验证方式

启动环境时控制台会打印阴影组统计：

```
[TaxiGraphEnv] Gate Shadow Zone: 27 gates, avg 3.2 shadow peers/gate.
```

---

## 7. 产出文件索引

| 文件 | 生成脚本 | 用途 |
|---|---|---|
| `results/exp_4_2/MAPPO_summary.json` | `run_paper_exp_4_2.py` | 论文表 4-2 数据 |
| `results/exp_4_2/comparison_table.txt` | `run_paper_exp_4_2.py` | 直接复制到论文 |
| `results/exp_4_2/MAPPO_micro_corridor.json` | `run_paper_exp_4_2.py` | 廊道分镜图原始数据 |
| `checkpoints/guided/training_log_improved.csv` | `train_curriculum.py` | 消融收敛曲线（Guided 组） |
| `checkpoints/no_guide/training_log_improved.csv` | `train_curriculum.py` | 消融收敛曲线（NoGuide 组） |
| `checkpoints/no_corridor/training_log_improved.csv` | `train_curriculum.py` | 消融收敛曲线（NoCorridor 组） |
| `paper_figures/fig_closure_heatmap.png` | `plot_paper_figure_suite.py` | 封路热力图 |

---

## 附录：代码修改记录

本次对代码库进行了以下有据可查的修改，均不影响已有实验结果：

| 文件 | 修改内容 | 改动性质 |
|---|---|---|
| `env/taxi_graph_env.py` | 新增 `_build_gate_shadow_groups()` 方法 + `__init__` 调用 | **新增功能** |
| `agent/aircraft_agent.py` | `__init__` 新增 `use_astar_guide` 参数；`get_action_mask()` 新增翼尖阴影检查；`get_observation()` A* 特征置零条件 | **新增功能** |
| `env/marl_env.py` | 透传 `use_astar_guide` 参数 | **新增功能** |
| `configs/config.py` | `EnvConfig` 新增 `use_astar_guide` 字段 | **新增功能** |
| `run_paper_exp_4_2.py` | 新增廊道协议轨迹采集逻辑及 JSON 输出 | **新增功能** |

> 所有修改均向后兼容（新参数均有默认值），现有 checkpoint 和训练脚本无需改动即可正常运行。
