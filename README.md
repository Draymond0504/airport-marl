# Airport MARL

机场航空器滑行协调的算法与实验研究工程：A* 路径先验、MAPPO、
物理滑行仿真、滚动执行器及实验图表。当前 main 是 2026 年 7 月迁移版
（Layout F），不等同于桌面的 2026-10-10 Layout H 论文交付包。

## 安装与运行

使用 Python 3.11 或 3.12。GPU 服务器根据 CUDA 版本安装匹配的 PyTorch。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_improved.py --config configs/fine_tune_8agents.yaml --save-dir checkpoints/new_run --seed 42
```

滚动评估示例（旧版 Layout F 协议）：

```bash
python run_layered_paper_exp_final.py --config configs/fine_tune_8agents.yaml --model models/guided_best_eval.pt --excel data.xlsx --scenario baseline --result-dir results/new_run/baseline --time-scale 3.0 --schedule-compression 0.10 --max-steps 50000 --active-limit 8 --seed 42
```

详细方法、限制与实验入口见 `MIGRATION_README_CN.md`。
训练输出放在 `checkpoints/`，测试输出放在 `results/`，均默认忽略。
`models/` 保留精选权重用于复现。

## 平台拆分

ApronOps 机场机坪自主作业协同与异常处置平台的工作树已拆至独立项目
[arponviz](https://github.com/Draymond0504/arponviz)，用于 Streamlit 看板、FastAPI 后端、车辆仿真、
测试与 Docker 配置的维护。算法仓库不再包含这些业务模块。
拆分前内容可通过初始提交 `f70a553` 追溯。平台与 MAPPO 尚未集成。
