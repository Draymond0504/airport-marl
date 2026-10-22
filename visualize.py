"""智能体行为可视化模块（视频导出版）。

支持两种输出模式：
  1. 交互窗口（默认）：实时播放 episode
  2. 视频导出（--save-video）：将 episode 渲染为 .mp4 或 .gif 文件

使用方法:
    python visualize.py                                     # 交互窗口播放
    python visualize.py --save-video episode.mp4           # 导出 mp4
    python visualize.py --save-video episode.gif           # 导出 GIF
    python visualize.py --model checkpoints/model_ep45000.pt --save-video result.mp4
    python visualize.py --random --save-video random.gif   # 随机策略导出
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # 无头后端，既可保存文件，也可在窗口模式下切换
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved


# 动作名称映射
ACTION_NAMES = {
    0: "N0", 1: "N1", 2: "N2", 3: "N3", 4: "N4",
    5: "减速", 6: "加速", 7: "等待",
}

# 智能体颜色
AGENT_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12",
                "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


def _build_graph_data(env: MARLEnv) -> tuple[dict, list]:
    """从环境中提取可视化用的节点坐标和边列表。"""
    node_pos = {}
    for nid, node in env.graph_env.nodes.items():
        node_pos[nid] = node.position

    edges_set: set[tuple[int, int]] = set()
    for e in env.graph_env.edges:
        pair = (min(e.start, e.end), max(e.start, e.end))
        edges_set.add(pair)

    return node_pos, list(edges_set)


def _draw_frame(
    ax: plt.Axes,
    env: MARLEnv,
    node_pos: dict,
    edges: list,
    step: int,
    actions: list[int] | None,
    episode: int,
    extra_info: dict[str, Any] | None = None,
) -> None:
    """绘制单帧。"""
    ax.clear()

    ge = env.graph_env

    # ---------- 绘制边 ----------
    for u, v in edges:
        x = [node_pos[u][0], node_pos[v][0]]
        y = [node_pos[u][1], node_pos[v][1]]
        eidx = ge.get_edge_between(u, v)
        color = "#cccccc" if (eidx is not None and ge.edges[eidx].closed) else "#d5d8dc"
        lw = 0.5 if (eidx is not None and ge.edges[eidx].closed) else 1.2
        ax.plot(x, y, color=color, linewidth=lw, zorder=1)

    # ---------- 绘制节点 ----------
    from env.taxi_graph_env import NodeType
    for nid, (nx_, ny_) in node_pos.items():
        ntype = ge.nodes[nid].type
        if ntype == NodeType.GATE:
            ax.plot(nx_, ny_, 'o', color='#fadbd8', markersize=10,
                    markeredgecolor='#e74c3c', markeredgewidth=0.8, zorder=2)
        else:
            ax.plot(nx_, ny_, 's', color='#d6eaf8', markersize=8,
                    markeredgecolor='#2980b9', markeredgewidth=0.6, zorder=2)
        ax.text(nx_, ny_, str(nid), fontsize=5, ha='center', va='center',
                fontweight='bold', zorder=3)

    # ---------- 绘制走廊占用状态 ----------
    for corridor in ge.corridor_mgr.corridors:
        if corridor.direction is not None:
            seq = corridor.nodes_seq
            for k in range(len(seq) - 1):
                u, v = seq[k], seq[k + 1]
                if u in node_pos and v in node_pos:
                    mx = (node_pos[u][0] + node_pos[v][0]) / 2
                    my = (node_pos[u][1] + node_pos[v][1]) / 2
                    dx = node_pos[v][0] - node_pos[u][0]
                    dy = node_pos[v][1] - node_pos[u][1]
                    arrow_scale = 0.3
                    if corridor.direction == "reverse":
                        dx, dy = -dx, -dy
                    ax.annotate("", xy=(mx + dx * arrow_scale, my + dy * arrow_scale),
                                xytext=(mx - dx * arrow_scale, my - dy * arrow_scale),
                                arrowprops=dict(arrowstyle="->", color="#f1c40f",
                                                lw=1.5, alpha=0.7), zorder=4)

    # ---------- 绘制智能体 ----------
    for i in range(env.num_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        pos_id = ge.agent_positions[i]
        goal_id = ge.agent_goals[i]

        if ge.done[i]:
            gx, gy = node_pos[goal_id]
            ax.plot(gx, gy, '*', color=color, markersize=16, zorder=10)
            ax.text(gx, gy + 0.8, f"A{i}√", fontsize=7, color=color,
                    ha='center', fontweight='bold', zorder=11)
            continue

        # 规划路径（半透明线）
        path = env.agents[i].planned_path
        for k in range(len(path) - 1):
            if path[k] in node_pos and path[k + 1] in node_pos:
                px = [node_pos[path[k]][0], node_pos[path[k + 1]][0]]
                py = [node_pos[path[k]][1], node_pos[path[k + 1]][1]]
                ax.plot(px, py, color=color, linewidth=2, alpha=0.25, zorder=5)

        # 目标位置（空心大圆）
        gx, gy = node_pos[goal_id]
        ax.plot(gx, gy, 'o', color=color, markersize=18, fillstyle='none',
                markeredgewidth=2, alpha=0.5, zorder=6)

        # 考虑异步发车机制 (Start Delays / t0)
        start_delay = ge.start_delays[i] if hasattr(ge, 'start_delays') else 0
        is_ready = (start_delay <= 0)

        # 当前位置
        if ge.agent_on_edge[i] and ge.edge_idx[i] is not None:
            edge = ge.edges[ge.edge_idx[i]]
            sx, sy = node_pos[edge.start]
            ex, ey = node_pos[edge.end]
            prog = ge.progress[i]
            cx = sx + (ex - sx) * prog
            cy = sy + (ey - sy) * prog
        else:
            cx, cy = node_pos[pos_id]

        # 如果未起动，显示为半透明并增加倒计时
        alpha = 1.0 if is_ready else 0.3
        ax.plot(cx, cy, 'o', color=color, markersize=14, zorder=8,
                markeredgecolor='white', markeredgewidth=1.5, alpha=alpha)
        ax.text(cx, cy, f"A{i}", fontsize=6, color='white',
                ha='center', va='center', fontweight='bold', zorder=9, alpha=alpha)

        if not is_ready:
            label = f"T-{start_delay}"
            label_color = "#7f8c8d"
        else:
            speed = ge.agent_speeds[i]
            stuck = env._stuck_counters[i] if hasattr(env, '_stuck_counters') else 0
            action_str = ACTION_NAMES.get(actions[i], "?") if actions else "?"
            label = f"v={speed:.1f} act={action_str}"
            if stuck > 0: label += f" stuck={stuck}"
            
            if stuck > 10: label_color = "#c0392b"
            elif stuck > 6: label_color = "#e67e22"
            elif stuck > 3: label_color = "#f1c40f"
            else: label_color = "#2c3e50"

        ax.text(cx + 0.5, cy + 0.8, label, fontsize=5.5, color=label_color,
                ha='left', zorder=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor=label_color, linewidth=0.5))

    # ---------- 标题和图例 ----------
    info_parts = [f"Episode {episode}  |  Step {step}/{ge.max_steps}"]
    if extra_info:
        if "reward" in extra_info:
            info_parts.append(f"R={extra_info['reward']:.1f}")
        if "conflicts" in extra_info:
            info_parts.append(f"Crash={extra_info['conflicts']}")

    ax.set_title("  ".join(info_parts), fontsize=10, fontweight='bold', pad=10)

    handles = []
    for i in range(env.num_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        goal = ge.agent_goals[i]
        start_node = env.agents[i].planned_path[0] if env.agents[i].planned_path else "?"
        handles.append(mpatches.Patch(color=color, label=f"A{i}: {start_node}→{goal}"))
    ax.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.8)

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.margins(0.15)


def _reset_episode(env: MARLEnv) -> tuple:
    """归一化场景初始化。
    直接调用 env.reset()，环境会自动根据 layout (如 F) 及其内部逻辑进行随机发车及异步时延采样，
    这与其在训练/测试时的行为完全一致。
    """
    return env.reset()


def collect_episode_frames(
    env: MARLEnv,
    trainer: MAPPOTrainerImproved | None,
    episode: int,
    use_random: bool = False,
) -> list[dict]:
    """运行一个 episode，收集所有帧信息（不绘图）。"""
    obs, masks = _reset_episode(env)
    global_state = env.get_global_state()

    node_pos, edges = _build_graph_data(env)
    frames = []

    done_all = False
    step_count = 0
    cumulative_reward = 0.0

    # 初始帧
    frames.append({
        "step": step_count,
        "actions": None,
        "reward": cumulative_reward,
        "conflicts": env.graph_env.conflict_count,
        "env_snapshot": None,  # 占位，实际用 callback 绘制
    })

    while not done_all:
        if use_random or trainer is None:
            actions = []
            for i in range(env.num_agents):
                valid = np.where(masks[i] > 0)[0]
                actions.append(int(np.random.choice(valid)) if len(valid) > 0 else 7)
        else:
            actions, _, _ = trainer.select_actions(obs, masks, global_state, deterministic=True)

        next_obs, rewards, dones, next_masks, info = env.step(actions)
        next_global_state = env.get_global_state()

        step_count += 1
        cumulative_reward += sum(rewards)

        frames.append({
            "step": step_count,
            "actions": list(actions),
            "reward": cumulative_reward,
            "conflicts": env.graph_env.conflict_count,
        })

        obs = next_obs
        masks = next_masks
        global_state = next_global_state
        done_all = all(dones) or step_count >= env.config.env.max_steps

    goal_count = sum(
        1 for i in range(env.num_agents)
        if env.graph_env.agent_positions[i] == env.graph_env.agent_goals[i]
    )
    print(f"  Episode {episode} 结束: Steps={step_count}, Reward={cumulative_reward:.1f}, "
          f"完成={goal_count}/{env.num_agents}, 冲突={env.graph_env.conflict_count}")

    return frames, node_pos, edges, goal_count


def save_episode_video(
    env: MARLEnv,
    trainer: MAPPOTrainerImproved | None,
    episode: int,
    output_path: str,
    fps: int = 10,
    use_random: bool = False,
) -> None:
    """录制一个 episode 并保存为视频文件（实时截取状态保真避免跳跃）。"""
    print(f"\n  [录制] Episode {episode} → {output_path} ...")

    import io
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL (Pillow) is required for high-fidelity GIF/Video saving.")
        return

    obs, masks = _reset_episode(env)
    global_state = env.get_global_state()
    node_pos, edges = _build_graph_data(env)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    step_count = 0
    cumulative_reward = 0.0
    done_all = False

    images = []

    # 捕获函数
    def _capture_frame(acts_):
        _draw_frame(ax, env, node_pos, edges, step_count, acts_, episode,
                    extra_info={"reward": cumulative_reward, "conflicts": env.graph_env.conflict_count})
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).copy()
        images.append(img)
        buf.close()

    # 初始状态第0帧
    _capture_frame(None)

    while not done_all:
        if use_random or trainer is None:
            acts = []
            for i in range(env.num_agents):
                valid = np.where(masks[i] > 0)[0]
                acts.append(int(np.random.choice(valid)) if len(valid) > 0 else 7)
        else:
            acts, _, _ = trainer.select_actions(obs, masks, global_state, deterministic=True)

        next_obs, rewards, dones, next_masks, info = env.step(acts)
        next_global_state = env.get_global_state()

        step_count += 1
        cumulative_reward += sum(rewards)

        # 在执行动作且环境状态严格更新结算后，立刻捕获现场
        _capture_frame(acts)

        obs = next_obs
        masks = next_masks
        global_state = next_global_state
        done_all = all(dones) or step_count >= env.config.env.max_steps

    goal_count = sum(
        1 for i in range(env.num_agents)
        if env.graph_env.agent_positions[i] == env.graph_env.agent_goals[i]
    )
    print(f"  Episode {episode} 结束: Steps={step_count}, Reward={cumulative_reward:.1f}, "
          f"完成={goal_count}/{env.num_agents}, 冲突={env.graph_env.conflict_count}")

    output_path = output_path.strip()
    # 强制统一走 GIF 导出线（对于此基于 Pillow 的高保真内存直录方式）
    if not output_path.lower().endswith(".gif"):
        output_path = str(Path(output_path).with_suffix(".gif"))
        print(f"  [提示] 使用实时高保真截帧导出，格式已自动转换为: {output_path}")

    try:
        images[0].save(
            output_path, 
            save_all=True, 
            append_images=images[1:], 
            optimize=False, 
            duration=1000 // fps, 
            loop=0
        )
        print(f"  [完成] 高保真 GIF 动画已保存到: {output_path}")
    except Exception as e:
        print(f"  [错误] 动画保存失败: {e}")

    plt.close(fig)


def run_visual_episode_interactive(
    env: MARLEnv,
    trainer: MAPPOTrainerImproved | None,
    episode: int,
    delay: float = 0.05,
    use_random: bool = False,
) -> None:
    """交互窗口模式运行并实时播放一个 episode。"""
    matplotlib.use("TkAgg")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    plt.ion()

    node_pos, edges = _build_graph_data(env)
    obs, masks = _reset_episode(env)
    global_state = env.get_global_state()

    done_all = False
    step_count = 0
    cumulative_reward = 0.0

    _draw_frame(ax, env, node_pos, edges, step_count, None, episode)
    plt.tight_layout()
    plt.pause(delay)

    while not done_all:
        if use_random or trainer is None:
            actions = []
            for i in range(env.num_agents):
                valid = np.where(masks[i] > 0)[0]
                actions.append(int(np.random.choice(valid)) if len(valid) > 0 else 7)
        else:
            actions, _, _ = trainer.select_actions(obs, masks, global_state, deterministic=True)

        next_obs, rewards, dones, next_masks, info = env.step(actions)
        next_global_state = env.get_global_state()

        step_count += 1
        cumulative_reward += sum(rewards)

        _draw_frame(ax, env, node_pos, edges, step_count, actions, episode,
                    extra_info={"reward": cumulative_reward,
                                "conflicts": env.graph_env.conflict_count})
        plt.tight_layout()
        plt.pause(delay)

        obs = next_obs
        masks = next_masks
        global_state = next_global_state
        done_all = all(dones) or step_count >= env.config.env.max_steps

    goal_count = sum(
        1 for i in range(env.num_agents)
        if env.graph_env.agent_positions[i] == env.graph_env.agent_goals[i]
    )
    print(f"  Episode {episode} 结束: Steps={step_count}, Reward={cumulative_reward:.1f}, "
          f"完成={goal_count}/{env.num_agents}, 冲突={env.graph_env.conflict_count}")

    plt.ioff()
    ax.set_title(
        f"Episode {episode} 完成 | Steps={step_count} | "
        f"Reward={cumulative_reward:.1f} | Goal={goal_count}/{env.num_agents} | "
        f"(关闭窗口继续下一轮)",
        fontsize=10, fontweight='bold',
        color='#27ae60' if goal_count == env.num_agents else '#c0392b'
    )
    plt.draw()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Airport MARL 智能体行为可视化")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--model", type=str, default="checkpoints/4.23.pt", help="模型权重路径")
    parser.add_argument("--episodes", type=int, default=3, help="可视化回合数")
    parser.add_argument("--delay", type=float, default=0.05, help="帧间延迟(秒)，仅交互模式有效")
    parser.add_argument("--random", action="store_true", help="使用随机策略(不加载模型)")
    parser.add_argument(
        "--save-video", type=str, default=None, metavar="OUTPUT",
        help="保存为视频文件，支持 .mp4 和 .gif，例：--save-video episode.mp4"
    )
    parser.add_argument("--fps", type=int, default=10, help="导出视频帧率（默认10）")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)  
    env = MARLEnv(config)

    trainer = None
    if not args.random:
        model_path = args.model or config.test.model_path
        trainer = MAPPOTrainerImproved(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            global_state_dim=env.global_state_dim,
            hidden_dim=config.training.hidden_dim,
        )
        try:
            trainer.load(model_path)
            print(f"✓ 已加载模型: {model_path}")
        except FileNotFoundError:
            print(f"✗ 模型文件不存在: {model_path}，使用随机策略")
            trainer = None

    print(f"\n{'='*50}")
    print(f"可视化配置:")
    print(f"  布局: {config.env.layout}")
    print(f"  智能体数: {env.num_agents}")
    print(f"  最大步数: {config.env.max_steps}")
    print(f"  回合数: {args.episodes}")
    print(f"  输出模式: {'视频导出 → ' + args.save_video if args.save_video else '交互窗口'}")
    print(f"  策略: {'随机' if trainer is None else '模型'}")
    print(f"{'='*50}\n")

    for ep in range(1, args.episodes + 1):
        print(f"\n--- Episode {ep}/{args.episodes} ---")
        if args.save_video:
            # 视频导出：处理多 episode 时自动加编号后缀
            if args.episodes > 1:
                p = Path(args.save_video)
                out = str(p.with_stem(f"{p.stem}_ep{ep}"))
            else:
                out = args.save_video
            save_episode_video(env, trainer, ep, out, fps=args.fps,
                               use_random=(trainer is None))
        else:
            run_visual_episode_interactive(
                env, trainer, ep, delay=args.delay, use_random=(trainer is None)
            )

    print("\n可视化完成。")


if __name__ == "__main__":
    main()
