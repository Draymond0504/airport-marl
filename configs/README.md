# Airport MARL — 机场滑行多智能体强化学习系统

## 一、项目概述

本项目实现了一个**机场局部滑行网络（含港湾区）多架飞机协同滑行决策系统**。采用**"传统路径算法导航 + 多智能体强化学习冲突决策"**的分层架构：

- **路径规划（A\*）** 负责为每架飞机生成从起点到目标的先验导航路径；
- **强化学习（MAPPO）** 在运行时决定每架飞机是前进、等待还是调速，从而协调冲突。

> ⚠️ 核心原则：**强化学习不重新规划路径，只做冲突协调（前进 / 等待 / 调速）**。

---

## 二、项目结构

```
airport_marl/
├── configs/
│   ├── __init__.py
│   ├── config.py          # 配置管理（dataclass + YAML 加载）
│   └── default.yaml       # 默认超参数
├── env/
│   ├── __init__.py
│   ├── taxi_graph_env.py  # 图网络环境（节点/边/冲突/动态事件）
│   └── marl_env.py        # 多智能体 RL 接口封装
├── planner/
│   ├── __init__.py
│   └── astar.py           # A* 路径规划器（手动实现）
├── agent/
│   ├── __init__.py
│   └── aircraft_agent.py  # 飞机智能体（观测/掩码/路径跟踪）
├── trainer/
│   ├── __init__.py
│   └── mappo.py           # MAPPO 训练器（Actor/Critic/GAE/PPO）
├── train.py               # 训练主入口
├── test.py                # 测试主入口（三类场景）
└── README.md              # 本文档
```

---

## 三、各模块说明

### 3.1 配置模块 (`configs/`)

| 文件 | 说明 |
|------|------|
| `default.yaml` | 环境、奖励权重、训练、测试的全部超参数 |
| `config.py` | 用 `dataclass` 将 YAML 映射为 `Config` 对象 |

修改 `default.yaml` 即可调节奖励权重、学习率、训练轮数等。

### 3.2 图网络环境 (`env/taxi_graph_env.py`)

- **数据结构**：`Node`（id / position / type）、`Edge`（start / end / length / occupied / closed）
- **默认图**：8 节点（2 Gate、4 Intersection、2 Runway），16 条有向边
- **动态事件**：滑行道关闭 / 恢复、优先级打乱、航班延误
- **核心接口**：

| 方法 | 说明 |
|------|------|
| `reset()` | 重置环境，随机分配起点和目标 |
| `step(actions)` | 执行一步，返回环境信息 |
| `get_neighbors(node_id)` | 获取可达邻居 |
| `check_conflict(agent_id, next_node)` | 检查节点冲突 |
| `get_graph_for_planner()` | 导出供 A\* 使用的邻接表 |

### 3.3 A\* 路径规划 (`planner/astar.py`)

- 手动实现标准 A\*，**不依赖外部库**
- 启发函数：欧几里得距离
- 仅考虑拓扑和静态距离，**忽略动态冲突**
- 接口：`plan_path(start, goal, graph) -> [node1, node2, ...]`

### 3.4 飞机智能体 (`agent/aircraft_agent.py`)

**动作空间（8 维离散）**：

| 动作 | 含义 | 适用状态 |
|------|------|----------|
| 0–4 | 选择第 0–4 个邻接节点 | 在节点上 |
| 5 | 减速 | 节点 / 边上 |
| 6 | 加速 | 节点 / 边上 |
| 7 | 保持当前速度 | 节点 / 边上 |

**动作掩码规则**：
- 在节点上 → 可选邻接节点 + 速度控制
- 在边上 → 仅速度控制
- `v = 0` 时禁止减速，`v = max` 时禁止加速

**观测向量**（维度 = `max_neighbors + 2`）：

```
[邻接节点距离 × 5, 当前速度, 参考速度]
```

距离 = 到邻居的边长 + 邻居到目标的最短路径距离；不可达填 −1。

### 3.5 多智能体环境接口 (`env/marl_env.py`)

将 `TaxiGraphEnv`、`AStarPlanner`、`AircraftAgent` 整合为标准 RL 接口：

```python
obs, masks = env.reset()
obs, rewards, dones, masks, info = env.step(actions)
global_state = env.get_global_state()  # 供 centralized critic
```

**奖励函数**：

| 项 | 权重（默认） | 说明 |
|----|-------------|------|
| 完成奖励 | +100 | 到达目标节点 |
| 冲突惩罚 | −50 | 与其他飞机占据同一节点 |
| 等待惩罚 | −1 | 在节点上原地不动 |
| 速度偏差惩罚 | −0.5 × \|v − v_ref\| | 偏离参考速度 |

权重可在 `default.yaml` 的 `reward` 节调节。

### 3.6 MAPPO 训练器 (`trainer/mappo.py`)

**完全手动实现**，包含：

| 组件 | 说明 |
|------|------|
| `Actor` | 两层 MLP → 动作概率分布（支持 action mask） |
| `Critic` | 两层 MLP → 全局状态价值（centralized） |
| `RolloutBuffer` | 轨迹缓存 |
| `compute_gae()` | GAE（Generalized Advantage Estimation） |
| `update()` | PPO clip loss + 熵正则 + 价值损失 |

架构特点：
- **Centralized Critic**：输入全局状态，输出统一价值
- **Decentralized Actor**：所有智能体共享网络参数，各自观测独立决策
- 支持 mini-batch 随机梯度更新

---

## 四、快速开始

### 4.1 环境依赖

```
Python >= 3.10
PyTorch
NumPy
PyYAML
```

安装依赖：

```bash
pip install torch numpy pyyaml
```

### 4.2 训练

```bash
cd airport_marl
python train.py                         # 使用默认配置
python train.py --config configs/default.yaml  # 指定配置文件
```

模型保存在 `checkpoints/` 目录下。

### 4.3 测试

```bash
python test.py                                  # 使用默认模型路径
python test.py --model checkpoints/latest.pt    # 指定模型
```

测试将依次运行三种场景，并输出以下指标：

| 指标 | 说明 |
|------|------|
| 平均滑行时间 | 每回合平均步数 |
| 冲突次数 | 所有回合累积 |
| 完成率 | 到达目标的智能体比例 |
| 吞吐量 | 完成智能体数 / 总步数 |

---

## 五、配置说明 (`configs/default.yaml`)

```yaml
env:
  num_agents: 4          # 飞机数量
  max_steps: 200         # 每回合最大步数
  max_speed: 3.0         # 最大滑行速度
  taxiway_close_prob: 0.01  # 滑行道关闭概率
  delay_prob: 0.02       # 延误概率

reward:
  completion: 100.0      # 完成奖励
  conflict: -50.0        # 冲突惩罚
  wait: -1.0             # 等待惩罚
  speed_deviation: -0.5  # 速度偏差惩罚系数

training:
  total_episodes: 50000
  lr_actor: 3.0e-4
  gamma: 0.99
  clip_epsilon: 0.2
  # ... 完整参数见文件
```

---

## 六、默认滑行网络拓扑

```
G0(0) ─── I2(2) ─── I3(3) ─── R6(6)
            │         │
G1(1) ─── I4(4) ─── I5(5) ─── R7(7)
```

- **G** = Gate（登机口），**I** = Intersection（交叉口），**R** = Runway（跑道）
- 所有边双向，长度均为 2.0
- 可通过传入自定义 `nodes` 和 `edges` 列表替换为真实机场拓扑

---

## 七、扩展指南

| 需求 | 方法 |
|------|------|
| 使用真实机场地图 | 构造自定义 `Node` / `Edge` 列表传入 `TaxiGraphEnv` |
| 调节奖励权重 | 修改 `configs/default.yaml` 的 `reward` 部分 |
| 增加智能体数量 | 修改 `env.num_agents`，确保 Gate 节点数量充足 |
| 增加动态扰动强度 | 调大 `taxiway_close_prob` / `delay_prob` |
| 更换 RL 算法 | 在 `trainer/` 下新建模块，保持相同接口 |
| 可视化 | 在 `TaxiGraphEnv` 的 `_get_info()` 中获取位置数据后绘图 |
