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

def _save_delay_visualization(env: "MARLEnv", speed_history: dict[int, list[float]], delay_history: dict[int, list[bool]], output_path: str = "scenario_delay.png"):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.unicode_minus"] = False
    num_agents = env.num_agents
    fig, axes = plt.subplots(num_agents, 1, figsize=(10, 2.5 * num_agents))
    if num_agents == 1: axes = [axes]
    for i in range(num_agents):
        ax = axes[i]
        speeds = speed_history[i]
        time_s = [t * 10 for t in range(len(speeds))]
        ax.plot(time_s, speeds, color="#3498db", linewidth=1.5, label="Actual Speed")
        
        in_delay, start_delay = False, 0
        for t, is_d in enumerate(delay_history[i]):
            if is_d and not in_delay:
                in_delay, start_delay = True, t * 10
            elif not is_d and in_delay:
                in_delay = False
                label = "Forced Delay Block" if start_delay == 0 or "Forced Delay Block" not in [l.get_label() for l in ax.patches] else ""
                ax.axvspan(start_delay, t * 10, color='red', alpha=0.3, label=label)
        if in_delay:
            ax.axvspan(start_delay, len(delay_history[i]) * 10, color='red', alpha=0.3)
        
        ax.set_title(f"Agent {i} Delay Analysis", fontsize=11, fontweight="bold")
        ax.set_ylabel("Speed (m/s)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\n  [图表] Delay 场景速度与延误线已保存: {output_path}")

def _save_closure_visualization(env: "MARLEnv", actual_paths: dict, closure_history: set, starts: list, goals: list, output_path: str = "scenario_closure.png"):
    plt.rcParams["font.family"] = "serif"
    fig, ax_map = plt.subplots(figsize=(10, 8))
    node_pos = {n.id: n.position for n in env.graph_env.nodes.values()}
    edges = env.graph_env.edges

    for i, e in enumerate(edges):
        sx, sy = node_pos[e.start]
        ex, ey = node_pos[e.end]
        if i in closure_history:
            ax_map.plot([sx, ex], [sy, ey], color="red", linewidth=4.0, linestyle="--", alpha=0.9, zorder=4)
            ax_map.text((sx+ex)/2, (sy+ey)/2, "X", color="red", fontsize=14, fontweight="bold", ha="center")
        else:
            ax_map.plot([sx, ex], [sy, ey], color="#4169E1", linewidth=0.8, alpha=0.3, zorder=1)

    for nid, pos in node_pos.items():
        ax_map.plot(pos[0], pos[1], '.', color="#4169E1", markersize=2, alpha=0.3, zorder=2)

    for i in range(env.num_agents):
        path = actual_paths.get(i, [])
        px, py = [node_pos[n][0] for n in path if n in node_pos], [node_pos[n][1] for n in path if n in node_pos]
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax_map.plot(px, py, color=c, linewidth=2.0, alpha=0.8, label=f"Agent {i} Path", zorder=3)
        if starts[i] in node_pos: ax_map.plot(node_pos[starts[i]][0], node_pos[starts[i]][1], 'o', color=c, markersize=8, zorder=5)
        if goals[i] in node_pos: ax_map.plot(node_pos[goals[i]][0], node_pos[goals[i]][1], 'X', color=c, markersize=8, zorder=5)

    ax_map.plot([], [], color="red", linewidth=4.0, linestyle="--", label="Closed Taxiway")
    ax_map.set_title("Closure Scenario: Blocked Taxiways & Emergency Routing", fontweight="bold")
    ax_map.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\n  [图表] Closure 场景封路与绕行图已保存: {output_path}")

def _save_priority_visualization(env: "MARLEnv", priority_history: list, output_path: str = "scenario_priority.png"):
    if not priority_history: return
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(10, 4))
    num_agents = env.num_agents
    time_s = [t * 10 for t in range(len(priority_history))]
    ph = list(map(list, zip(*priority_history)))
    
    for i in range(num_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(time_s, ph[i], color=c, linewidth=2.5, drawstyle='steps-post', label=f"Agent {i} Priority")
        
    ax.set_ylabel("Priority Rank")
    ax.set_xlabel("Time (s)")
    ax.set_title("Dynamic Priority Shuffling Over Time", fontweight="bold", fontsize=12)
    ax.set_yticks(range(num_agents))
    ax.legend(loc="upper right")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\n  [图表] Priority 场景优先权折线图已保存: {output_path}")

# ==============================================================================


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

        is_ep0 = (ep == 0)
        speed_history = {i: [] for i in range(env.num_agents)}
        delay_history = {i: [] for i in range(env.num_agents)}
        closure_history = set()
        priority_history = []

        done_all = False
        step_count = 0

        while not done_all:
            actions, _, _ = trainer.select_actions(
                obs, masks, global_state, deterministic=True
            )
            obs, rewards, dones, masks, info = env.step(actions)
            global_state = env.get_global_state()
            step_count += 1
            
            # --- 采集数据 (仅第一回合作图用) ---
            if is_ep0:
                for i in range(env.num_agents):
                    speed_history[i].append(env.graph_env.agent_speeds[i])
                    delay_history[i].append(info.get("delayed", [False]*env.num_agents)[i])
                
                closed_edges = info.get("closure", {}).keys()
                for ce in closed_edges:
                    closure_history.add(ce)
                
                priority_history.append(info.get("priority", list(range(env.num_agents))))
            # -------------------------------
            
            # 记录新的节点
            for i in range(env.num_agents):
                cur_pos = env.graph_env.agent_positions[i]
                if cur_pos != actual_paths[i][-1]:
                    actual_paths[i].append(cur_pos)

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

        # 在对比脚本中，由于场景跨度大且主要是跑分，不再逐一打印每个 epoch 下全部飞机的航迹。
        if is_ep0:
            if scenario == "delay":
                _save_delay_visualization(env, speed_history, delay_history)
            elif scenario == "closure":
                _save_closure_visualization(env, actual_paths, closure_history, case["starts"], case["goals"])
            elif scenario == "priority":
                _save_priority_visualization(env, priority_history)

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
    scenarios = ["normal", "delay", "closure", "priority"]
    scenario_names = {
        "normal": "正常运行",
        "delay": "延误扰动",
        "closure": "滑行道关闭",
        "priority": "优先级变更",
    }

    print("=" * 70)
    print("Airport Taxiway MARL - Test Report (4 Scenarios)")
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
