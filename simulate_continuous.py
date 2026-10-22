import argparse
import numpy as np
import pandas as pd
import os
import re
import random
import torch

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved
from data_loader import STAND_TO_NODE_MAP
from cbs_baseline import build_edge_length_map
from perturbation_scenarios import PerturbationScenario, ScenarioInjector


def load_full_flight_schedule(filepath: str, time_scale: float = 5.0, transit_ratio: float = 0.35, start_hour: float = 8.0, end_hour: float = 24.0):
    """加载全量航班计划，在控制台打印筛选后的计划表，返回可运行任务列表

    Args:
        transit_ratio: 对机位不在局部网络内的航班，按此比例随机采样为"过境干扰代理"
                       （100→105穿越任务）。0=不加载过境流量，1=全部加载。
                       调整此参数可控制场面平均并发数。约0.35时并发数约为3架次。
        start_hour: 筛选的时间窗口起点（默认8.0即08:00）
        end_hour: 筛选的时间窗口终点（默认24.0即24:00）
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件 {filepath} 不存在")
        return None
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"加载失败: {e}")
        return None

    def parse_time_sec(t_val):
        if pd.isna(t_val): return None, None
        t_str = str(t_val).strip()
        if not t_str or t_str == 'nan': return None, None
        h, m = 0, 0
        if hasattr(t_val, 'hour'):
            h, m = t_val.hour, t_val.minute
        else:
            match = re.search(r'(\d+):(\d+)', t_str)
            if match:
                h, m = int(match.group(1)), int(match.group(2))
            else:
                return None, None
        return h * 3600 + m * 60, f"{h:02d}:{m:02d}"

    def format_time_sec(seconds):
        seconds = int(seconds) % (24 * 3600)
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"

    col_type  = "进出"
    col_aldt  = "实际到达"
    col_atot  = "实际起飞"
    col_gate  = "机位"
    col_airline = "航空公司"

    flights = []
    transit_candidates = []  # 暂存过境代理候选

    for idx, row in df.iterrows():
        gate_str = str(row.get(col_gate, "")).strip()
        airline  = str(row.get(col_airline, "未知"))
        f_type   = str(row.get(col_type, ""))

        m_gate = re.search(r'\d+', gate_str)
        gate_num = m_gate.group() if m_gate else None
        if not gate_num:
            continue

        is_local = gate_num in STAND_TO_NODE_MAP  # 机位在局部网络内？

        if "进" in f_type:
            t_sec, t_str = parse_time_sec(row.get(col_aldt)) or parse_time_sec(row.get("计划到达"))
            if t_sec is None:
                continue
            
            # 过滤不在指定时间窗口内的航班
            if not (start_hour * 3600 <= t_sec <= end_hour * 3600):
                continue
                
            t_sec += 5 * 60  # 进港延后5分钟到达局部网络

            if is_local:
                # 目标代理：精确机位任务
                flights.append({
                    "time_sec": t_sec, "time_str": t_str,
                    "start": random.choice([100, 200]), "goal": STAND_TO_NODE_MAP[gate_num],
                    "type": "ARR", "airline": airline, "gate": gate_num,
                    "agent_role": "TARGET"
                })
            else:
                # 过境代理候选：跑道→另一侧跑道穿越
                transit_candidates.append({
                    "time_sec": t_sec, "time_str": t_str,
                    "start": random.choice([100, 200]), "goal": random.choice([105, 205]),
                    "type": "TRN", "airline": airline, "gate": gate_num,
                    "agent_role": "TRANSIT"
                })

        if "出" in f_type:
            t_sec, t_str = parse_time_sec(row.get(col_atot)) or parse_time_sec(row.get("计划起飞"))
            if t_sec is None:
                continue
                
            # 过滤不在指定时间窗口内的航班
            if not (start_hour * 3600 <= t_sec <= end_hour * 3600):
                continue
                
            t_sec -= 10 * 60  # 离港提前10分钟推出

            if is_local:
                # 目标代理：精确机位任务
                flights.append({
                    "time_sec": t_sec, "time_str": t_str,
                    "start": STAND_TO_NODE_MAP[gate_num], "goal": random.choice([105, 205]),
                    "type": "DEP", "airline": airline, "gate": gate_num,
                    "agent_role": "TARGET"
                })
            else:
                # 过境代理候选：机位侧→跑道侧穿越（起点用跑道入口代替，因不在局部网络）
                transit_candidates.append({
                    "time_sec": t_sec, "time_str": t_str,
                    "start": random.choice([100, 200]), "goal": random.choice([105, 205]),
                    "type": "TRN", "airline": airline, "gate": gate_num,
                    "agent_role": "TRANSIT"
                })

    # 按transit_ratio随机采样过境代理并合并
    if transit_ratio > 0 and transit_candidates:
        random.seed(42)
        n_transit = int(len(transit_candidates) * transit_ratio)
        selected_transit = random.sample(transit_candidates, min(n_transit, len(transit_candidates)))
        flights.extend(selected_transit)
        print(f"[DataLoader] 目标代理: {len(flights) - len(selected_transit)} | 过境代理: {len(selected_transit)} (ratio={transit_ratio})")
    else:
        print(f"[DataLoader] 目标代理: {len(flights)} | 过境代理: 0 (disabled)")

    if not flights: return None
    flights.sort(key=lambda x: x["time_sec"])

    # --- 自动剔除头部/尾部孤立离群航班 ---
    # 若相邻航班时间间隔 > 1 小时，认为前者是离群点，截断至后者
    GAP_THRESHOLD_SEC = 3600  # 1小时
    removed_count = 0
    while len(flights) > 1 and (flights[1]["time_sec"] - flights[0]["time_sec"]) > GAP_THRESHOLD_SEC:
        removed_item = flights.pop(0)
        removed_count += 1
        print(f"  [跳过] 孤立航班: {removed_item['airline']}({removed_item['type']}-{removed_item['gate']}) @{removed_item.get('time_str', '?')}，与后续航班间隔过大")
    if removed_count:
        print(f"  共跳过 {removed_count} 个头部孤立航班，仿真从 {flights[0].get('time_str', '?')} 开始。")

    base_t = flights[0]["time_sec"]
    for f in flights:
        f["t0"] = int(max(0, f["time_sec"] - base_t) / time_scale)

    # ① 打印筛选后的计划表 (增加时间列)
    print("\n" + "="*80)
    print("航班计划表（已筛选，拓扑可用机位）".center(75))
    print("="*80)
    print(f"{'#':<5} {'计划时刻':<8} {'释放时刻':<8} {'航空公司':<10} {'类型':<8} {'机位':<6} {'t0(step)':<10} {'起点/终点':<10}")
    print("-"*80)
    for i, f in enumerate(flights):
        node_pair  = f"{f['start']}->{f['goal']}"
        role_tag   = f"[{f.get('agent_role','?')}]"
        release_str = format_time_sec(f["time_sec"])
        print(f"  {i:<4} {f['time_str']:<8} {release_str:<8} {f['airline']:<10} {f['type']:<6}{role_tag:<4} {f['gate']:<6} {f['t0']:<10} {node_pair:<10}")
    print("="*80)
    print(f"共加载 {len(flights)} 个可用任务。")

    return flights



def simulate_continuous_marl_analysis():
    parser = argparse.ArgumentParser(description="动态队列仿真 + 轨迹/指标记录")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="checkpoints/stage4_8agents_latest.pt")
    parser.add_argument("--excel", type=str, default="data.xlsx")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--gui", action="store_true", help="开启图形界面（默认关闭）")
    parser.add_argument("--result-dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--time-scale", type=float, default=1.127,
                        help="物理秒/仿真步。默认5(1step=5s)。设为更大值可压缩时间轴提升并发密度，"
                             "例如 --time-scale 20 使航班间隔缩小4倍，峰值并发约4-6架" \
                             "1.127为实际标定值")
    parser.add_argument("--deadlock-timeout", type=int, default=300,
                        help="连续N步无位移判定死锁（默认200步）")
    parser.add_argument("--scenario", type=str, default="baseline",
                        choices=["baseline", "delay", "priority", "closure"],
                        help="测试扰动场景：baseline(无扰动)/delay(航班延误)/priority(优先级变化)/closure(滑行道关闭)")
    parser.add_argument("--transit-ratio", type=float, default=0.35,
                        help="过境干扰代理采样比例(0~1)。0=纯目标代理，调大可提升并发密度，"
                             "约0.35时平均并发≈3架次")
    parser.add_argument("--peak-hours", type=float, nargs=2, default=[8.0, 22.0],
                        help="筛选的时间窗口，格式为 '8.0 24.0'")
    args = parser.parse_args()

    # 1. 准备目录
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 2. 加载数据
    flights = load_full_flight_schedule(args.excel, time_scale=args.time_scale,
                                        transit_ratio=args.transit_ratio,
                                        start_hour=args.peak_hours[0],
                                        end_hour=args.peak_hours[1])
    if not flights:
        print("未找到有效航班数据。")
        return

    num_total = len(flights)
    max_t0 = max(f["t0"] for f in flights)+200

    # 3. 初始化环境
    config = Config.from_yaml(args.config)
    config.env.num_agents = num_total
    config.env.max_steps = args.max_steps

    env = MARLEnv(config)
    # 构建边长映射（用于统计实际滑行距离）
    edge_len_map = build_edge_length_map(env.graph_env)

    # --- 扰动场景初始化 ---
    scenario_enum = PerturbationScenario(args.scenario)
    injector = ScenarioInjector(
        scenario=scenario_enum,
        graph_env=env.graph_env,
        num_agents=num_total,
        total_steps=args.max_steps,
        seed=42,
    )
    print(f"\n[SCENARIO] 当前测试场景: {scenario_enum.value.upper()}")

    # 4. 加载模型（只用 Actor，忽略 Critic 不匹配）
    trainer = MAPPOTrainerImproved(
        num_agents=num_total,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=config.training.hidden_dim,
    )
    try:
        trainer.load(args.model)
        print(f"\n✓ 模型加载成功: {args.model}")
    except Exception as e:
        print(f"[WARN] {e}")

    # 5. 环境重置
    starts = [f["start"] for f in flights]
    goals = [f["goal"] for f in flights]
    delays = [f["t0"] for f in flights]
    obs, masks = env.reset(starts=starts, goals=goals, delays=delays)

    # 6. 可视化准备
    if args.gui:
        import matplotlib
        try: matplotlib.use("TkAgg")
        except: pass
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        from visualize_cases import _build_graph_data, _draw_frame
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        plt.ion()
        node_pos, edges = _build_graph_data(env)

    # 7. 记录器
    agent_paths = {i: [] for i in range(num_total)}
    agent_spawn_step = [None] * num_total
    agent_finish_step = [None] * num_total
    # 死锁检测：记录上次位置和滞留计数
    agent_last_pos = [-1] * num_total
    agent_stall_cnt = [0] * num_total
    deadlock_count = 0
    concurrency_log = []  # 每步活跃智能体数量

    print(f"\n[RUNNING] 开始推演，共 {num_total} 个任务...")
    import time as _time
    wall_start = _time.time()

    total_steps_executed = args.max_steps
    for step in range(args.max_steps):
        # --- 每步注入预设扰动（仅在目标场景下生效）---
        injector.step(step)

        active_agents = []
        actions = [7] * num_total

        for i in range(num_total):
            if agent_spawn_step[i] is None and env.graph_env.start_delays[i] == 0 and not env.graph_env.done[i]:
                agent_spawn_step[i] = step

            if agent_finish_step[i] is None and env.graph_env.done[i]:
                agent_finish_step[i] = step

            if not env.graph_env.done[i] and env.graph_env.start_delays[i] == 0:
                active_agents.append(i)
                agent_paths[i].append(env.graph_env.agent_positions[i])

        if all(env.graph_env.done):
            total_steps_executed = step
            break

        # 记录本步并发数（仅在有飞机活跃时计入，排除纯空档）
        if active_agents:
            concurrency_log.append(len(active_agents))

        # --- 快进（Fast-Forward）：无活跃智能体时直接跳到下一个 t0 ---
        if not active_agents:
            # 找到最早的下一个未完成任务的剩余延迟
            pending_delays = [
                env.graph_env.start_delays[i]
                for i in range(num_total)
                if not env.graph_env.done[i] and env.graph_env.start_delays[i] > 0
            ]
            if pending_delays:
                jump = min(pending_delays) - 1  # -1 确保下一步能正确激活
                if jump > 0:
                    for i in range(num_total):
                        if not env.graph_env.done[i]:
                            env.graph_env.start_delays[i] = max(0, env.graph_env.start_delays[i] - jump)
                    step += jump
                    total_steps_executed = step
                    continue  # 跳过本步的 env.step()

        # --- 死锁检测：对连续滞留过长的智能体强制跳过 ---
        for i in list(active_agents):
            cur_pos = env.graph_env.agent_positions[i]
            if cur_pos == agent_last_pos[i]:
                agent_stall_cnt[i] += 1
            else:
                agent_stall_cnt[i] = 0
                agent_last_pos[i] = cur_pos

            if agent_stall_cnt[i] >= args.deadlock_timeout:
                # 强制标记该智能体为死锁失败
                env.graph_env.crashed[i] = True
                env.graph_env.done[i] = True
                if agent_finish_step[i] is None:
                    agent_finish_step[i] = step
                active_agents.remove(i)
                deadlock_count += 1
                agent_stall_cnt[i] = 0  # 重置防止重复计数

        # --- 批量推理：一次性处理所有活跃航空器的决策 ---
        if active_agents:
            with torch.no_grad():
                # 打包观测值与掩码
                obs_batch = torch.FloatTensor(np.array([obs[i] for i in active_agents])).to(trainer.device)
                mask_batch = torch.FloatTensor(np.array([masks[i] for i in active_agents])).to(trainer.device)
                
                # 批量推断概率分布
                dist = trainer.actor(obs_batch, mask_batch)
                # 获取各个智能体的最优动作
                batch_actions = torch.argmax(dist.probs, dim=-1).cpu().numpy()
                
                for idx, agent_id in enumerate(active_agents):
                    actions[agent_id] = int(batch_actions[idx])

        # 进度反馈（每 200 步打印一次）
        if step % 200 == 0:
            active_cnt = len(active_agents)
            # 统计真正成功的：已 done 且没有 crashed
            success_cnt = sum(1 for i in range(num_total) if env.graph_env.done[i] and not env.graph_env.crashed[i])
            done_cnt = sum(env.graph_env.done)
            print(f"  推演进度: Step {step:<5} | 场内活跃: {active_cnt:<2} | 成功引导: {success_cnt:<3} | 已处理: {done_cnt:<3}/{num_total}")

        obs, rewards, dones, masks, info = env.step(actions)

        if args.gui and step % 5 == 0:
            _draw_frame(ax, env, node_pos, edges, step, actions, 1)
            ax.set_title(f"Step {step} | Active: {len(active_agents)} | Done: {sum(env.graph_env.done)}/{num_total}")
            plt.pause(0.01)

    # 8. 统计结果
    print("\n" + "="*60)
    print("仿真性能报告".center(56))
    print("="*60)

    rows = []
    success_list = []
    crash_list = []
    unfinished_list = []
    taxi_times = []
    path_lengths = []

    for i in range(num_total):
        f = flights[i]
        path = agent_paths[i]
        deduped = [path[0]] if path else []
        for node in path[1:]:
            if node != deduped[-1]:
                deduped.append(node)
        path_str = "-".join(str(n) for n in deduped)

        spawn = agent_spawn_step[i]
        finish = agent_finish_step[i]
        duration = (finish - spawn) if (spawn is not None and finish is not None) else None
        crashed = env.graph_env.crashed[i]

        plan_id = f"R{i+1}"
        flight_id = f"{f['airline'].strip()}_{f['gate']}"

        row = {
            "plan_id": plan_id,
            "flight_id": flight_id,
            "flight_type": f["type"],
            "path_nodes": path_str,
            "start_time": f["time_sec"],
            "end_time": f["time_sec"] + (duration * 5 if duration else 0),
            "duration": duration if duration else -1
        }
        rows.append(row)

        if crashed:
            crash_list.append(i)
        elif finish is not None and duration is not None:
            success_list.append(i)
            taxi_times.append(duration)
            # 实际物理距离：相邻节点边权累加
            dist = sum(
                edge_len_map.get((deduped[k], deduped[k+1]), 0)
                for k in range(len(deduped) - 1)
            )
            path_lengths.append(dist)
        else:
            unfinished_list.append(i)

    wall_elapsed = _time.time() - wall_start
    success_count = len(success_list)
    # 碰撞：两架同时 crash（来自物理碰撞），死锁：被超时强制结束的
    collision_count = (len(crash_list) - deadlock_count * 2) // 2
    collision_count = max(0, collision_count)
    avg_taxi = np.mean(taxi_times) if taxi_times else 0
    avg_path_len = np.mean(path_lengths) if path_lengths else 0
    throughput = success_count / total_steps_executed if total_steps_executed > 0 else 0

    print(f"1. 成功引导总数: {success_count} / {num_total}")
    if unfinished_list:
        print(f"   未完成航班 ({len(unfinished_list)} 架):")
        for i in unfinished_list:
            f = flights[i]
            print(f"     - #{i} {f['time_str']} {f['airline']}({f['type']}-{f['gate']})")

    print(f"2. 碰撞冲突总数: {collision_count}  |  死锁强制跳过: {deadlock_count}")
    print(f"3. 平均滑行时间: {avg_taxi:.2f} steps (约 {avg_taxi*5:.1f} 秒)")
    print(f"4. 平均滑行距离: {avg_path_len:.1f} 单位（场内拓扑边长之和）")
    print(f"5. 系统吞吐率:   {throughput:.4f} agents/step")
    print(f"6. 总仿真时长:   {total_steps_executed} steps (= {total_steps_executed*5/3600:.2f} 小时)")
    max_conc = max(concurrency_log) if concurrency_log else 0
    avg_conc = np.mean(concurrency_log) if concurrency_log else 0
    print(f"6. 最大同时在场: {max_conc} 架次")
    print(f"7. 平均同时在场: {avg_conc:.2f} 架次")
    print(f"8. 实际运行时间: {wall_elapsed:.1f} 秒")
    print("="*60)
    print("[注] 1 step = 5 秒物理时间，与 FCFS A* / CBS 的时间步单位一致")

    # 打印扰动事件统计
    injector.summary()

    # 9. 保存结果至 result 文件夹
    save_path = os.path.join(args.result_dir, "simulation_trajectories.csv")
    try:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n任务轨迹已保存至: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")

    if args.gui:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    simulate_continuous_marl_analysis()
