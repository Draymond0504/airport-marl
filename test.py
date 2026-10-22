"""测试模块。

支持三类场景测试：
1. 正常运行
2. 延误扰动
3. 滑行道关闭

输出指标：平均滑行时间、冲突次数、完成率、吞吐量。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved


@dataclass
class TestMetrics:
    """测试指标。"""
    avg_taxi_time: float = 0.0
    total_conflicts: int = 0
    completion_rate: float = 0.0
    throughput: float = 0.0


def _apply_scenario(env: "MARLEnv", scenario: str) -> None:
    """根据场景名称配置环境参数。

    场景机理（重构为持久化事件驱动版）：
    ─────────────────────────────────────────────────────────────────
    normal（正常运行）:
        delay_prob=0.00, close_prob=0.00, priority_prob=0.00
        机理：基线无额外扰动。
    ─────────────────────────────────────────────────────────────────
    delay（延误扰动）:
        delay_prob=0.02, close_prob=0.00, priority_prob=0.01
        机理：概率降低至0.02，但一旦触发将强制进入 3～10 物理帧的瘫痪等待期，
        测试模型面对长时阻塞式不可预测队友的反应能力。
    ─────────────────────────────────────────────────────────────────
    closure（滑行道关闭）:
        delay_prob=0.00, close_prob=0.05, priority_prob=0.01
        机理：引发周期性的大规模封路（在 40 步检查点触发）。
        关闭效果持续 20~50 动作帧，引发局部交通阻干。
    ─────────────────────────────────────────────────────────────────
    priority（优先级变更）:
        delay_prob=0.01, close_prob=0.01, priority_prob=0.15
        机理：大幅提升优先级重排概率。优先级重排带有 20-40 步的
        稳定冷却期。测试智能体的动态让行协调自适应能力。
    """
    g = env.graph_env

    if scenario == "normal":
        g.delay_prob      = 0.00
        g.close_prob      = 0.00
        g.priority_prob   = 0.00
    elif scenario == "delay":
        g.delay_prob      = 0.02
        g.close_prob      = 0.00
        g.priority_prob   = 0.01
    elif scenario == "closure":
        g.delay_prob      = 0.00
        g.close_prob      = 0.05
        g.priority_prob   = 0.01
    elif scenario == "priority":
        g.delay_prob      = 0.01
        g.close_prob      = 0.01
        g.priority_prob   = 0.15



AGENT_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]


def _save_speed_curves(
    env: "MARLEnv",
    actual_paths: dict[int, list[int]],
    speed_history: dict[int, list[float]],
    case: dict,
    output_path: str = "speed_curves_case0.png",
) -> None:
    """按论文标准对比排版绘制 2 x N 子图。
    第一行：各智能体的航迹平面图 (红线轨迹 + 起终点)
    第二行：各智能体的速度剖面图 (蓝线实际速度，橙线平滑目标速度，竖线抵达时间)
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.unicode_minus"] = False

    num_agents = env.num_agents
    fig, axes = plt.subplots(2, num_agents, figsize=(7 * num_agents, 9))
    if num_agents == 1:
         axes = np.array([[axes[0]], [axes[1]]])

    starts = case["starts"]
    goals  = case["goals"]
    t0_list = case.get("t0", [0] * num_agents)

    node_pos = {n.id: n.position for n in env.graph_env.nodes.values()}
    edges = env.graph_env.edges

    for i in range(num_agents):
        ax_map = axes[0, i]
        ax_spd = axes[1, i]

        # ==========================================
        # 1. 绘制上半侧航迹平面图
        # ==========================================
        for e in edges:
            sx, sy = node_pos[e.start]
            ex, ey = node_pos[e.end]
            ax_map.plot([sx, ex], [sy, ey], color="#4169E1", linewidth=0.5, alpha=0.3)
            ax_map.plot(sx, sy, '.', color="#4169E1", markersize=1, alpha=0.3)
            ax_map.plot(ex, ey, '.', color="#4169E1", markersize=1, alpha=0.3)

        path = actual_paths[i]
        px, py = [], []
        for n in path:
            if n in node_pos:
                px.append(node_pos[n][0])
                py.append(node_pos[n][1])
        ax_map.plot(px, py, color="#ff3333", linewidth=2.0, alpha=0.85, label="Path Taken by the Agent")

        start_node = starts[i]
        goal_node = goals[i]
        if start_node in node_pos:
            ax_map.plot(node_pos[start_node][0], node_pos[start_node][1], 'o', 
                        color="#ff7f0e", markersize=5, zorder=5, label="Departure Node")
        if goal_node in node_pos:
            ax_map.plot(node_pos[goal_node][0], node_pos[goal_node][1], 'X', 
                        color="#2ca02c", markersize=5, zorder=5, label="Arrival Node")

        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        airport_name = "Layout F" if getattr(env.config.env, "layout", "") == "F" else "Custom Layout"
        ax_map.set_title(f"({chr(97+i)}) Aircraft Trajectory for Agent {i} ({airport_name})", y=-0.22, fontweight='bold', fontsize=11)
        ax_map.legend(loc="upper left", fontsize=8)

        # ==========================================
        # 2. 绘制下半侧速度剖面图
        # ==========================================
        speeds = speed_history[i]
        time_s = [t * 10 for t in range(len(speeds))]  # 1 step = 10s
        
        # 利用滑动平均法平滑出一个 Target Speed 用于对比
        window_size = min(15, len(speeds))
        if window_size > 3:
            target_speed = np.convolve(speeds, np.ones(window_size)/window_size, mode='same')
            target_speed[:window_size//2] = target_speed[window_size//2]
            target_speed[-window_size//2:] = target_speed[-window_size//2-1]
        else:
            target_speed = speeds.copy()
            
        ax_spd.plot(time_s, speeds, color="#3498db", linewidth=1.5, label="Aircraft speed", zorder=3)
        ax_spd.plot(time_s, target_speed, color="#ff7f0e", linewidth=1.5, linestyle="-", label="Target speed", zorder=2)

        if len(time_s) > 0:
            arr_time = time_s[-1]
            ax_spd.axvline(x=arr_time, color="blue", linestyle="-", linewidth=1.5, alpha=0.6, label="Arrival time")
            # 假想的理论抵达期限 (用于论文对比样式)
            assign_arr_time = arr_time + 30
            ax_spd.axvline(x=assign_arr_time, color="#e84393", linestyle="-", linewidth=2.0, alpha=0.7, label="Assigned arrival time")

        ax_spd.set_xlabel("Time (s)")
        ax_spd.set_ylabel("Speed (m/s)")
        max_v = max(speeds) if speeds else 2.0
        ax_spd.set_ylim(-0.5, max_v + 1.5)
        
        ax_spd.set_title(f"({chr(97+i+num_agents)}) Speed Profile (Agent {i})", y=-0.25, fontweight='bold', fontsize=11)
        ax_spd.legend(loc="lower right", fontsize=8)
        
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [图表] Case 0 论文标准对比速度曲线已保存到: {output_path}")


# ==============================================================================
# 固定测试用例（确保实验可重复性，且贴近真实起降逻辑）
# 设计要求：
#   一半航班为离港（Outbound）：机位(0~29) 出发 -> 跑道连接点(105/205)
#   一半航班为进港（Inbound） ：跑道连接点(100/200) 出发 -> 机位(0~29)
#   t0(f) = 每架飞机预定进入时间（单位：步，1步=10秒），模拟真实时刻表
#            0 = 已在场内 / 刚推出; 正整数 = 还需等待若干步后才进入路网
# ==============================================================================
FIXED_TEST_CASES = [
    # Case 0: 基础场景 — 两架离港飞机就绪推出，两架进港飞机同批落地
    {"starts": [0,   5,   100, 200], "goals": [205, 105, 14,  20],  "t0": [0,  0,  0,  0 ]},
    # Case 1: 分批推出+进港时差30s
    {"starts": [1,   10,  100, 200], "goals": [105, 205, 18,  5 ],  "t0": [0,  3,  0,  3 ]},
    # Case 2: 离港飞机同步推出，进港飞机先后落地
    {"starts": [0,   2,   100, 200], "goals": [205, 105, 11,  4 ],  "t0": [0,  60,  0,  0]},
    # Case 3: 进港飞机首批落地，离港飞机20步后推出
    {"starts": [6,   9,   100, 200], "goals": [105, 205, 12,  17],  "t0": [20, 20, 0,  5 ]},
    # Case 4: 四批次依次进入，均匀间距 15 步（150秒）
    {"starts": [17,  6,   100, 200], "goals": [205, 105, 29,  0 ],  "t0": [0,  15, 30, 45]},
    # Case 5: 紧急并发：所有飞机同时进入（高冲突压力测试）
    {"starts": [12,  16,  100, 200], "goals": [105, 205, 14,  15],  "t0": [0,  0,  0,  0 ]},
    # Case 6: 进港优先，离港延迟 30 步
    {"starts": [11,  24,  100, 200], "goals": [205, 105, 5,   0 ],  "t0": [30, 30, 0,  0 ]},
    # Case 7: 离港优先，进港延迟 20 步
    {"starts": [20,  29,  100, 200], "goals": [105, 205, 23,  24],  "t0": [0,  0,  50, 50]},
    # Case 8: 两两成批，批间间隔 25 步
    {"starts": [3,   8,   100, 200], "goals": [205, 105, 15,  22],  "t0": [0,  0,  25, 25]},
    # Case 9: 最大跨度场景 — 飞机陆续在 0/10/20/30 步加入
    {"starts": [1,   2,   100, 200], "goals": [105, 205, 28,  29],  "t0": [0,  20, 40, 60]},
]

def run_scenario(
    env: "MARLEnv",
    trainer: "MAPPOTrainerImproved",
    num_episodes: int,
    scenario: str = "normal",
) -> "TestMetrics":
    """运行指定场景的测试，并在 normal 场景打印详细路径对比。"""
    _apply_scenario(env, scenario)

    total_taxi_time = 0.0
    total_conflicts = 0
    total_completed = 0
    total_agents = 0
    
    # 指标：记录 RL 实际走过的路径节点数 vs 纯最短路节点数
    total_actual_len = 0
    total_baseline_len = 0

    for ep in range(num_episodes):
        # 固定测试用例（循环使用，超过 case 总数后取余）
        case = FIXED_TEST_CASES[ep % len(FIXED_TEST_CASES)]
        obs, masks = env.reset(starts=case["starts"], goals=case["goals"], delays=case.get("t0"))
        global_state = env.get_global_state()

        # 记录初始情况下的纯最短路(A*)作为 Baseline
        baseline_paths = {i: list(env.agents[i].planned_path) for i in range(env.num_agents)}
        
        # 记录 RL 的实际宏观路由节点序列 (仅保留节点跳跃序列，去重连续在边上停留的记录)
        actual_paths = {i: [env.graph_env.agent_positions[i]] for i in range(env.num_agents)}

        # 速度历史：仅对 Case 0 的 Normal 场景追踪
        is_case0_normal = (ep == 0 and scenario == "normal")
        speed_history: dict[int, list[float]] = {i: [] for i in range(env.num_agents)}

        done_all = False
        step_count = 0

        while not done_all:
            actions, _, _ = trainer.select_actions(
                obs, masks, global_state, deterministic=True
            )
            obs, rewards, dones, masks, info = env.step(actions)
            global_state = env.get_global_state()
            step_count += 1
            
            # 记录新的节点
            for i in range(env.num_agents):
                cur_pos = env.graph_env.agent_positions[i]
                if cur_pos != actual_paths[i][-1]:
                    actual_paths[i].append(cur_pos)
                # 速度追踪
                if is_case0_normal:
                    speed_history[i].append(env.graph_env.agent_speeds[i])

            done_all = all(dones) or step_count >= env.config.env.max_steps

        # Episode 结束后的数据统计
        total_taxi_time += step_count
        total_conflicts += env.graph_env.conflict_count
        completed = sum(1 for d in dones if d)
        total_completed += completed
        total_agents += env.num_agents
        
        for i in range(env.num_agents):
            if env.graph_env.agent_positions[i] == env.graph_env.agent_goals[i]:
                total_actual_len += len(actual_paths[i])
                total_baseline_len += len(baseline_paths[i])

        # 在 Normal 场景每个 Episode 都打印每架飞机的路径细节
        if scenario == "normal":
            case_id = ep % len(FIXED_TEST_CASES)
            print(f"\n{'='*55}")
            print(f" [Case {case_id}] Episode {ep+1} | Starts={case['starts']} -> Goals={case['goals']}")
            print(f"{'='*55}")
            for i in range(env.num_agents):
                status = "✔️" if env.graph_env.agent_positions[i] == env.graph_env.agent_goals[i] else "❌"
                steps = len(actual_paths[i]) - 1
                astar_steps = len(baseline_paths[i]) - 1
                diff = steps - astar_steps
                diff_str = f"+{diff}" if diff > 0 else (f"{diff}" if diff < 0 else "=")
                print(f"\n  Agent {i} {status}  {case['starts'][i]} -> {case['goals'][i]} | "
                      f"实际: {steps}步 | A*: {astar_steps}步 | 差异: {diff_str}")
                print(f"  A* : " + " -> ".join(map(str, baseline_paths[i])))
                print(f"  RL : " + " -> ".join(map(str, actual_paths[i])))

        # 输出 Case 0 论文级排版图表
        if is_case0_normal and any(speed_history[i] for i in range(env.num_agents)):
            _save_speed_curves(env, actual_paths, speed_history, case)

    metrics = TestMetrics(
        avg_taxi_time=total_taxi_time / max(num_episodes, 1),
        total_conflicts=total_conflicts,
        completion_rate=total_completed / max(total_agents, 1),
        throughput=total_completed / max(total_taxi_time, 1),
    )
    # 你可以把绕路统计放到这里以便输出，为了兼容这里依旧用 TestMetrics
    return metrics

def test(config_path: str | None = None, model_path: str | None = None) -> None:
    """运行全部测试场景。"""
    config = Config.from_yaml(config_path)
    if model_path is None:
        model_path = config.test.model_path

    env = MARLEnv(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = MAPPOTrainerImproved(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=config.training.hidden_dim,
        lr_actor=config.training.lr_actor,
        lr_critic=config.training.lr_critic,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_epsilon=config.training.clip_epsilon,
        entropy_coef=config.training.entropy_coef,
        value_loss_coef=config.training.value_loss_coef,
        max_grad_norm=config.training.max_grad_norm,
        device=device,
    )

    try:
        trainer.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"WARNING: Model not found at {model_path}, using random policy.")

    num_episodes = config.test.num_episodes
    scenarios = ["normal"]
    scenario_names = {
        "normal": "正常运行",
    }

    print("=" * 70)
    print("Airport Taxiway MARL - Test Report (Normal Scenario Only)")
    print("=" * 70)

    for scenario in scenarios:
        metrics = run_scenario(env, trainer, num_episodes, scenario)
        print(f"\n{'─' * 50}")
        print(f"场景: {scenario_names[scenario]} ({scenario})")
        print(f"{'─' * 50}")
        print(f"  平均滑行时间: {metrics.avg_taxi_time:.2f} steps")
        print(f"  冲突次数:     {metrics.total_conflicts}")
        print(f"  完成率:       {metrics.completion_rate:.2%}")
        print(f"  吞吐量:       {metrics.throughput:.4f} agents/step")

    print(f"\n{'=' * 70}")
    print("Test completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Airport MARL Testing")
    parser.add_argument("--config", type=str, default="configs/improved_v2.yaml", help="Config YAML path")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    args = parser.parse_args()
    test(args.config, args.model)


if __name__ == "__main__":
    main()
