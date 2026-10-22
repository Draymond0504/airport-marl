# Airport MARL 项目迁移说明

生成日期：2026-07-31

## 1. 项目定位

本项目不是提出一个仅适用于港湾区的专用优化方法，而是在真实港湾区复杂运行环境中验证 planning-guided MARL 框架。核心路线为：

```text
航班计划与港湾区拓扑
        -> A* 全局参考路径
        -> MAPPO 局部协调决策
        -> 执行器安全保护与停滞重规划
        -> rolling test 与动态扰动评价
```

其中，A* 为规划先验，MAPPO 根据局部观测进行在线决策，执行器负责对名义动作进行物理和运行规则检查。执行器不是策略网络的一部分，而是测试/运行环境中的安全执行层。

## 2. 已实现内容

### 2.1 拓扑与运行环境

- Layout F 港湾区滑行网络。
- 30 个机位节点，代码索引为 `0-29`。
- 对应机位前/stand-front stub 节点为 `2000-2029`。
- 外围滑行道、连接滑行道、内部 corridor、stand-access/pushback-link 均由图结构表示。
- 节点占用、边占用、封闭边、corridor 访问和单向/对向运行保护。
- 港湾区机位附近的低速运行规则。
- 连续边内位置进度、速度和物理距离计算。
- 加速、减速、速度上限、机位低速限制、转弯提前减速和安全净空检查。
- 节点冲突、同向追尾、对向边冲突和空间安全间隔检查。
- Gate shadow zone，用于处理机位及相邻机位之间的推出/释放影响。

### 2.2 A* 与观测接口

- A* 在航空器激活时生成参考路径。
- 新航班进入 rolling slot 时重新生成路径。
- 普通 MAPPO 决策步骤不会每一步重新运行完整 A*。
- Guided 观测包含候选邻居的路径代价和是否为 A* 推荐下一节点等特征。
- NoGuide 通过 `use_astar_guide: false` 关闭这些 A* 观测特征，但保持观测维度一致。
- 当 closure 使原路径失效时，执行器可以事件触发 A* 路径刷新，然后将更新后的路径特征提供给策略。

### 2.3 MAPPO 训练

- CTDE 结构：分散式 Actor 和集中式 Critic。
- Actor 使用航空器自身状态、目标进度、邻居状态、冲突风险、corridor 状态和 A* 局部引导。
- Critic 使用全局状态估计联合价值。
- PPO clipped objective、GAE、熵正则、Actor/Critic 更新和 checkpoint 保存。
- 支持固定智能体训练、8-agent 训练和从旧权重继续微调。
- 支持 Guided/NoGuide 消融配置。
- 支持训练日志、确定性评价日志、holdout 诊断和微观动作诊断。

### 2.4 Rolling 执行框架

- 完整航班计划通过外部队列管理，未释放航班不占用 active slot。
- `active_limit=8` 表示局部 rolling 协调的最大激活槽位，不是计划中航班总数。
- 航班完成或 timeout 后释放 slot。
- MAPPO 和 FCFS+A* 使用相同的航班计划、拓扑、安全约束和执行器。
- FCFS 根据固定服务顺序和 A* 路径运行，速度由默认执行规则推进，并接受同一执行器约束。
- 最终终止状态主要区分为 `completed` 和 `timeout`；执行器不会把未在时间窗内完成的航班表述为“无法分配路径”。

### 2.5 执行器与统计

当前 layered rolling executor 已实现以下分层逻辑：

```text
名义动作
  -> 方向/拓扑检查
  -> 资源占用和 corridor 准入检查
  -> 冲突保护
  -> 连续运动学执行
  -> 持续停滞判断
  -> 停滞路径恢复或 A* 重规划
  -> completed / timeout
```

已支持或已输出的核心量包括：

- Overall task completion rate；
- Target-aircraft completion rate；
- Target-aircraft throughput；
- Executed taxi time；
- Conflict-protection intervention count/frequency；
- `R_stall`，即持续无位置或边进度变化触发的停滞重规划次数；
- closure-triggered route replanning；
- operational timeout；
- 每架航空器最终状态、节点轨迹、运行步数和时间分解；
- execution events、stall recovery diagnostics、edge occupancy 和 edge traversal 日志。

## 3. 已实现但需要在新主机上特别注意的内容

### 3.1 Priority 场景当前实现

基础 `paper_rolling_executor.py` 中原始设置为：

```python
PRIORITY_TARGETS = {120, 180, 240, 300}
```

即 4 架固定航班分别在 admission 后获得优先级，窗口为 300 个仿真步。推荐的最终入口 `run_layered_paper_exp_final.py` 会通过 `run_layered_paper_exp_operational_v2.py` 将目标改为：

```python
{120, 180, 240}
```

因此，只有使用最终 layered 入口时，实验才对应“3 架代表性航空器”。这 3 架飞机仍然是从各自 admission 时刻开始分别计时，不是同一全局时刻同时提升。

当 `time_scale=3.0` 时：

```text
300 steps × 3.0 sec/step = 900 sec = 15 min
```

### 3.2 Closure 场景当前实现

推荐的 operational v2/final 入口将两次 closure 设置在服务期内，并把持续时间调整为 2400 个仿真步。closure 发生后，活动航班的当前路径可由执行层触发 A* 刷新。该过程不是每一步全局规划，也不需要在线重新训练策略。

### 3.3 时间参数

代码中的 `time_scale` 是 rolling 实验的仿真步到报告时间的换算参数。它不等同于底层连续动力学中的固定模型校准步长。迁移时必须保持实验命令中的 `time-scale` 一致，否则任务时间、吞吐量和扰动持续时间都会改变。

## 4. 尚未实现或不应过度声称的内容

### 4.1 完整 Pushback 物理建模：未实现

当前代码只通过 stand-access/pushback-link、机位占用和低速规则对推出阶段进行简化表示，尚未完整表示：

- 独立的 PUSHBACK 状态转换；
- 推出方向与后推轨迹；
- 牵引车或机型差异；
- 推出完成事件和机位释放时刻；
- 推出与主滑行道的时空冲突；
- 推出需求强度消融实验。

论文中不应把 Pushback 宣称为完整核心创新。推荐表述为“stand-access/pushback-link abstraction with low-speed operating rules”。

### 4.2 理论推导的 active limit=8：尚未自动完成

代码把 8 作为实验场景的 rolling active 上限，但当前实现没有自动根据航班计划计算“6、8、9 中为何选 8”的正式容量推导。若要形成严格论文证据，需要补充：

- 不受 slot 上限影响的计划并发统计；
- 预计滑行时间下的并发数量分布；
- 关键 corridor/共享资源的 headway 和容量分析；
- `N=6/8/9` 敏感性实验。

因此，当前可以把 8 描述为“场景特定的局部滚动协调上限”，不能直接描述为整个机场的物理容量。

### 4.3 Closure-specific policy training：未完整实现

主训练配置通常将 closure 概率设为 0。测试时 closure 由执行器触发路径刷新，因此当前结果体现的是“固定策略 + 事件触发规划刷新 + 执行器保护”的鲁棒性，而不是策略已经在大量 closure 拓扑上训练过。

若要声称策略本身学习了 closure 适应，应重新加入 closure 训练分布并完成对应训练和消融。

### 4.4 多随机种子统计：不完整

当前大量结果基于单个 seed。不能据此报告跨种子显著性结论。若用于正式论文，建议至少使用 3-5 个 seed，并报告均值和标准差/置信区间。

### 4.5 历史实验脚本版本较多

项目中同时存在 `paper_rolling_executor.py`、多个 `layered_rolling_executor_v*.py` 和多个 `run_paper_exp*.py`。它们并非完全等价。正式实验应优先使用：

```text
run_layered_paper_exp_final.py
run_layered_paper_exp_operational_v2.py
layered_rolling_executor.py
```

不要把旧版脚本和最终版结果混合到同一张论文表中。

## 5. 新主机安装与验证

### 5.1 环境

建议使用 Python 3.10 或更高版本，并根据新主机 CUDA 版本单独安装匹配的 PyTorch：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 当前只列出：`numpy`、`torch`、`pyyaml`、`matplotlib`、`networkx`。如果使用 Excel 航班计划，通常还需要：

```powershell
pip install pandas openpyxl
```

### 5.2 源码自检

```powershell
python -m py_compile train_improved.py train_curriculum.py paper_rolling_executor.py layered_rolling_executor.py
python -m py_compile env\taxi_graph_env.py env\marl_env.py agent\aircraft_agent.py planner\astar.py trainer\mappo_improved.py
```

### 5.3 训练入口

单阶段训练：

```powershell
python train_improved.py --config configs\fine_tune_8agents.yaml --save-dir checkpoints\new_run --seed 42
```

课程训练：

```powershell
python train_curriculum.py --config configs\improved_stable.yaml --save-dir checkpoints\curriculum_new --seed 42
```

继续训练：

```powershell
python train_improved.py --config configs\fine_tune_8agents.yaml --resume checkpoints\stage3_clearance_pilot_v1\best_eval.pt --save-dir checkpoints\new_finetune --seed 42
```

### 5.4 推荐的完整 rolling 实验入口

单个场景：

```powershell
python run_layered_paper_exp_final.py --config configs\fine_tune_8agents.yaml --model checkpoints\models\guided_best_eval.pt --excel data.xlsx --scenario baseline --result-dir results\new_run\baseline --time-scale 3.0 --schedule-compression 0.10 --max-steps 50000 --active-limit 8 --seed 42
```

四类场景建议分别运行 `baseline`、`delay`、`closure` 和 `priority`，并保持模型、航班计划、seed、time-scale、active-limit 和 max-steps 一致。

### 5.5 出图

多数绘图脚本直接读取 rolling 输出目录。优先使用：

```powershell
python plot_trb_results.py --baseline-dir results\new_run\baseline --delay-dir results\new_run\delay --closure-dir results\new_run\closure --priority-dir results\new_run\priority --out-dir results\new_run\figures
python plot_final_paper_figures.py --result-root results\new_run
```

## 6. 迁移包目录结构

```text
airport_marl_core_transfer/
├── MIGRATION_README_CN.md
├── requirements.txt
├── *.py                         # 训练、rolling、baseline、导出和绘图入口
├── agent/
├── env/
├── planner/
├── trainer/
├── configs/
├── data/
├── schedule_inputs/             # data.xlsx 等航班计划
├── models/                      # 代表性权重，不包含全部历史 checkpoint
└── docs/
```

项目中的 `results/` 约 17 GB、`checkpoints/` 约 1 GB，未全部打入核心代码包。它们属于历史实验产物，应根据需要单独迁移。

## 7. 迁移后的第一步

先运行编译检查和一个小规模 rolling smoke test，确认新主机的 PyTorch、Excel 读取、模型观测维度和 GPU/CPU 环境一致，再运行完整 363 架航班实验。不要直接用旧结果目录覆盖新主机上的结果目录。
