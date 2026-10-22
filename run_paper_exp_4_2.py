"""
论文实验脚本 4.2 节：基准场景性能对比
========================================
在相同的航班需求、相同随机种子下，
分别运行 FCFS+A*(动态重规划) 与 MAPPO，采集并对比关键指标。

生成结果目录 results/exp_4_2/ 下：
  - MAPPO_summary.json          汇总指标
  - MAPPO_timeseries.csv        吞吐量折线图数据
  - MAPPO_per_agent.csv         滑行时间箱线图数据
  - FCFS_summary.json
  - FCFS_timeseries.csv
  - FCFS_per_agent.csv
  - comparison_table.txt        论文表格（Markdown + 纯文本双格式）

运行示例：
  python run_paper_exp_4_2.py --config configs/improved_v2.yaml \\
         --model checkpoints/best.pt --excel data.xlsx \\
         --scenario baseline --time-scale 3.0 --max-steps 30000
"""

import argparse
import math
import heapq
import itertools
import time
import os
import json
import random
import numpy as np
import torch

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved
from perturbation_scenarios import PerturbationScenario, ScenarioInjector
from simulate_continuous import load_full_flight_schedule
from cbs_baseline import build_adj, build_edge_length_map
from baseline_fcfs_dynamic import bfs_heuristic_lazy, spatial_astar
from simulation_tracker import SimulationTracker
from paper_rolling_executor import run_fcfs_rolling, run_mappo_rolling


# ================================================================
# MAPPO 评估
# ================================================================
def run_mappo(args, flights, config, edge_len_map, result_dir):
    # The policy was trained with a fixed eight-aircraft population.  Keep that
    # population fixed and feed the full schedule through reusable slots.
    return run_mappo_rolling(args, flights, config, edge_len_map, result_dir)

    # Legacy all-flights-at-once executor retained below for historical
    # comparison only.  It must not be used for paper results.
    print("\n" + "=" * 62)
    print("  [MAPPO] 开始评估...")
    print("=" * 62)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_total = len(flights)
    config.env.num_agents = num_total
    config.env.max_steps = args.max_steps
    env = MARLEnv(config)

    # 扰动注入器
    scenario_enum = PerturbationScenario(args.scenario)
    injector = ScenarioInjector(scenario_enum, env.graph_env, num_total,
                                total_steps=args.max_steps, seed=args.seed,
                                flights=flights, time_scale=args.time_scale)

    # 加载模型
    trainer = MAPPOTrainerImproved(
        num_agents=num_total,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=config.training.hidden_dim,
    )
    trainer.load(args.model, load_critic=False, load_optimizer=False)
    print(f"  [OK] 模型加载: {args.model}")

    starts  = [f["start"] for f in flights]
    goals   = [f["goal"]  for f in flights]
    delays  = [f["t0"]    for f in flights]
    obs, masks = env.reset(starts=starts, goals=goals, delays=delays)

    # 跟踪器
    tracker = SimulationTracker(num_total, snapshot_interval=args.snapshot_interval,
                                time_step_sec=args.time_scale, algorithm_name="MAPPO")

    agent_last_pos  = list(env.graph_env.pos)
    agent_last_prog = [0.0] * num_total
    agent_stall_cnt = [0] * num_total
    total_steps_executed = args.max_steps
    # 用于碰撞差分（仅统计本步新增崩溃，而非累计）
    _prev_crashed_set: set[int] = set()

    # 重规划延误追踪（统计额外等待时间，无上限——路径规划仿真必须完成每一架飞机）
    DELAY_STEPS = max(1, int(60.0 / args.time_scale))
    delay_log   = [0] * num_total
    replan_cnt  = [0] * num_total  # 每架飞机被重规划的次数（纯统计用）

    # 微观轨迹记录（Priority 场景专用）
    _micro_log: list = []            # 存储每步的微观状态
    _micro_active = (args.scenario == "priority")
    _priority_event_step = -1        # 第一次优先级事件发生的 step
    _micro_window = 300              # 事件前后各 300 步的轨迹窗口

    # ── 廊道协议轨迹采集（所有场景均激活）──────────────────────
    # 目的：自动捕获“廊道等待”事件，提取前后各 N 帧快照用于微观分镜图
    # 采集条件：廊道被占用（direction != None）且有飞机在廊道入口节点静止等待
    # 每条廊道只采集首次触发事件
    _corridor_log: list = []                       # 已采集的廊道事件帧集合
    _corridor_captured: set = set()                # 已记录过的 corridor id
    _corridor_window = 50                          # 事件触发后再采集帧数
    _corridor_buffer: list = []                    # 滑动缓冲（存最近 N 帧，用于事件前回溯）
    _corridor_capturing: dict = {}                 # {corridor_id: 剩余待采集帧数}
    # ───────────────────────────────────────────────────────────────────────

    step = 0
    while step < args.max_steps:
        injector.step(step)

        # 更新跟踪器（放在推理之前）
        tracker.record_step(step, env.graph_env)

        active_agents = [
            i for i in range(num_total)
            if not env.graph_env.done[i]
            and env.graph_env.start_delays[i] == 0
        ]

        if all(env.graph_env.done):
            total_steps_executed = step
            break

        # ── Closure 事件驱动重规划 ──────────────────────────────────────────
        # 每当 closure 触发新封路，立即为所有未完成航班重规划（包括还没出发的）
        # 这样保证路径中含有封闭边的飞机，不论当前是否活跃，都能提前获得绕行路径
        _cur_closed = frozenset((e.start, e.end) for e in env.graph_env.edges if e.closed)
        if _cur_closed and _cur_closed != getattr(run_mappo, "_last_closed_set", frozenset()):
            run_mappo._last_closed_set = _cur_closed
            print(f"  [Closure 感知] 封路集合变化 -> {_cur_closed}，开始全量重规划...")
            tracker.start_replan()
            _replanned = 0
            for i in range(num_total):
                if env.graph_env.done[i]:
                    continue
                agent = env.agents[i]
                path = agent.planned_path
                # 判断剩余路径是否包含封闭边
                if any((path[k], path[k+1]) in _cur_closed for k in range(len(path)-1)):
                    # 以当前实际位置或出发起点为重规划起点
                    cur = env.graph_env.pos[i] if env.graph_env.start_delays[i] == 0 else starts[i]
                    g = env.graph_env.goal[i]
                    h_map = bfs_heuristic_lazy(g, env.graph_env.adj)
                    new_route = spatial_astar(cur, g, env.graph_env.adj, edge_len_map, h_map,
                                              hard_blocked_edges=_cur_closed)
                    if new_route and len(new_route) > 1:
                        agent.planned_path = new_route
                        agent.planned_path_set = set(new_route)
                        agent.path_index = 0
                        _replanned += 1
            tracker.end_replan()
            print(f"  [Closure 感知] 完成重规划 {_replanned} 架航班")
        # ───────────────────────────────────────────────────────────────────

        # 快进
        if not active_agents:
            pending = [env.graph_env.start_delays[i]
                       for i in range(num_total)
                       if not env.graph_env.done[i] and env.graph_env.start_delays[i] > 0]
            if pending:
                jump = min(pending) - 1
                if jump > 0:
                    for i in range(num_total):
                        if not env.graph_env.done[i]:
                            env.graph_env.start_delays[i] = max(0, env.graph_env.start_delays[i] - jump)
                    step += jump
                    total_steps_executed = step
                    continue

        # 死锁检测（严格的物理静止检测）
        for i in list(active_agents):
            cur_pos = env.graph_env.pos[i]
            cur_prog = env.graph_env.progress[i] if env.graph_env.on_edge[i] else 0.0
            
            if cur_pos == agent_last_pos[i] and abs(cur_prog - agent_last_prog[i]) < 1e-4:
                agent_stall_cnt[i] += 1
            else:
                agent_stall_cnt[i] = 0
                agent_last_pos[i]  = cur_pos
                agent_last_prog[i] = cur_prog
                
            if agent_stall_cnt[i] >= args.deadlock_timeout:
                replan_cnt[i]     += 1
                agent_stall_cnt[i] = 0

                # ── 超过最大遣返次数：永久放弃该飞机 ─────────────────────
                if replan_cnt[i] > args.max_deadlock_per_agent:
                    env.graph_env.done[i]    = True
                    env.graph_env.crashed[i] = True
                    print(f"  [MAPPO] 飞机 #{i} 第 {replan_cnt[i]} 次死锁，"
                          f"超过上限 {args.max_deadlock_per_agent} 次，永久放弃")
                    tracker.record_deadlock(agent_id=i)
                    active_agents.remove(i)
                    continue
                # ──────────────────────────────────────────────────────────────

                # 死锁 = 当前方案无效 → 重置回出发点重新规划
                env.graph_env.pos[i]        = starts[i]
                env.graph_env.on_edge[i]    = False
                env.graph_env.edge_idx[i]   = -1
                env.graph_env.progress[i]   = 0.0
                env.graph_env.speed[i]      = 0.0
                env.graph_env.done[i]       = False
                env.graph_env.crashed[i]    = False
                backoff_delay = DELAY_STEPS + random.randint(20, 200)
                env.graph_env.start_delays[i] = backoff_delay
                delay_log[i]      += backoff_delay

                print(f"  [MAPPO] 飞机 #{i} 第 {replan_cnt[i]} 次死锁，遣返并等待 {backoff_delay} 步以错峰出行")
                tracker.agents[i].spawn_step = None
                active_agents.remove(i)
                tracker.record_deadlock(agent_id=i)

        actions = [7] * num_total

        if active_agents:
            tracker.start_inference()
            with torch.no_grad():
                obs_batch  = torch.FloatTensor(np.array([obs[i]   for i in active_agents])).to(trainer.device)
                mask_batch = torch.FloatTensor(np.array([masks[i] for i in active_agents])).to(trainer.device)
                dist = trainer.actor(obs_batch, mask_batch)
                batch_acts = torch.argmax(dist.probs, dim=-1).cpu().numpy()
                
                # 【打破绝对确定性】：对曾经死锁过的飞机引入一定的探索概率 (Epsilon-greedy escape)
                for idx, aid in enumerate(active_agents):
                    if replan_cnt[aid] > 0:  # 只要曾被遣返过，说明原本贪心策略在该局部有缺陷
                        if random.random() < 0.2:  # 20% 概率尝试非贪心合法动作跳出死锁
                            valid_acts = torch.where(mask_batch[idx] > 0)[0].cpu().numpy()
                            if len(valid_acts) > 0:
                                batch_acts[idx] = random.choice(valid_acts)

            tracker.end_inference()

            for idx, aid in enumerate(active_agents):
                actions[aid] = int(batch_acts[idx])

            # ====== 【方案1：VIP 动作强制覆写】======
            # 强制 VIP 航班不听从 MAPPO 的协同减速/绕行指令，而是按最短路径“直线冲锋”
            if hasattr(injector, "get_boosted_agents"):
                boosted_set = set(injector.get_boosted_agents())
                for aid in boosted_set:
                    if aid not in active_agents:
                        continue
                    if env.graph_env.on_edge[aid]:
                        actions[aid] = 6 # 强制加速前进
                    else:
                        cur_pos = env.graph_env.pos[aid]
                        g = env.graph_env.goals[aid]
                        h_map = bfs_heuristic_lazy(g, env.graph_env.adj)
                        route = spatial_astar(cur_pos, g, env.graph_env.adj, edge_len_map, h_map)
                        if route and len(route) > 1:
                            next_node = route[1]
                            for a_idx, edge_info in enumerate(env.graph_env.adj[cur_pos]):
                                nb = edge_info[0] if isinstance(edge_info, tuple) else edge_info
                                if nb == next_node:
                                    actions[aid] = a_idx
                                    break
            # ==================================

        # Safety Shield：防止两架飞机被规划到同一节点（纯规划冲突，不是物理事故）
        occupied_nodes = set()
        occupied_edges = set()
        for j in range(num_total):
            if env.graph_env.done[j] or env.graph_env.start_delays[j] > 0:
                continue
            if env.graph_env.on_edge[j]:
                e = env.graph_env.edges[env.graph_env.edge_idx[j]]
                occupied_edges.add((e.start, e.end))
                # 移除 occupied_nodes.add(e.end)，允许同向尾随进入同一条边
            # Removed: occupied_nodes.add(env.graph_env.pos[j]) to allow tight following

        sorted_active = sorted(active_agents, key=lambda x: env.graph_env.priority[x], reverse=True)
        for i in sorted_active:
            a = actions[i]
            if a >= 5 or env.graph_env.on_edge[i]:
                if not env.graph_env.on_edge[i]:
                    occupied_nodes.add(env.graph_env.pos[i])
                continue
            nbrs = env.graph_env.adj[env.graph_env.pos[i]]
            if a >= len(nbrs):
                if not env.graph_env.on_edge[i]:
                    occupied_nodes.add(env.graph_env.pos[i])
                continue
            nxt, _ = nbrs[a]
            cur = env.graph_env.pos[i]
            if (nxt in occupied_nodes and nxt != cur) or (nxt, cur) in occupied_edges:
                actions[i] = 7
                occupied_nodes.add(cur)
            else:
                occupied_nodes.add(nxt)
                occupied_edges.add((cur, nxt))

        # ATC Gate Holding: 限制场面并发飞机数 (匹配训练环境的 num_agents=8)
        current_active = len(active_agents)
        for i in range(num_total):
            if not env.graph_env.done[i] and env.graph_env.start_delays[i] == 1:
                if current_active >= args.active_limit:
                    env.graph_env.start_delays[i] += 1  # 延长等待，暂不释放
                else:
                    current_active += 1

        obs, rewards, dones, masks, info = env.step(actions)

        # 微观轨迹采集（Priority 场景专用）
        if _micro_active:
            ev_log = injector.event_log
            if _priority_event_step < 0:
                for ev in ev_log:
                    if ev.get("type") == "priority_boost":
                        _priority_event_step = ev["step"]
                        print(f"  [微观记录] 检测到优先级事件@Step {_priority_event_step}，窗口 +/-{_micro_window} 步")
                        break
            if _priority_event_step >= 0 and abs(step - _priority_event_step) <= _micro_window:
                boosted_set = set(injector.get_boosted_agents())
                frame = {"step": step, "boosted": list(boosted_set), "agents": []}
                for j in range(num_total):
                    if env.graph_env.done[j] or env.graph_env.start_delays[j] > 0:
                        continue
                    frame["agents"].append({
                        "id": j,
                        "pos": env.graph_env.pos[j],
                        "on_edge": int(env.graph_env.on_edge[j]),
                        "progress": round(env.graph_env.progress[j] if env.graph_env.on_edge[j] else 0.0, 4),
                        "speed": round(env.graph_env.speed[j], 4),
                        "action": actions[j],
                        "is_vip": int(j in boosted_set),
                        "priority": env.graph_env.priority[j],
                    })
                _micro_log.append(frame)

        # ── 廊道协议微观采集────────────────────────────────────────
        # Step 1：构建当前帧快照（滚动入缓冲区）
        _cur_frame = {
            "step": step,
            "positions": list(env.graph_env.pos),
            "on_edge": [int(v) for v in env.graph_env.on_edge],
            "speed": [round(v, 4) for v in env.graph_env.speed],
            "done": [int(v) for v in env.graph_env.done],
            "corridor_status": {
                str(c.id): {
                    "direction": c.direction,
                    "occupied_by": list(c.occupied_by),
                }
                for c in env.graph_env.corridor_mgr.corridors
            },
        }
        _corridor_buffer.append(_cur_frame)
        if len(_corridor_buffer) > _corridor_window:
            _corridor_buffer.pop(0)

        # Step 2：检测廊道等待事件（首次触发）
        for c in env.graph_env.corridor_mgr.corridors:
            if c.id in _corridor_captured or c.id in _corridor_capturing:
                continue
            if c.direction is not None and len(c.occupied_by) > 0:
                # 检查是否有飞机局院在廊道入口节点静止（建立廊道端点集合）
                entry_nodes = {c.nodes_seq[0], c.nodes_seq[-1]}
                waiting_agents = [
                    j for j in range(num_total)
                    if not env.graph_env.done[j]
                    and not env.graph_env.on_edge[j]
                    and env.graph_env.pos[j] in entry_nodes
                    and env.graph_env.speed[j] < 0.1
                    and j not in c.occupied_by
                ]
                if waiting_agents:
                    print(f"  [廊道微观] 廊道 {c.id} @Step {step} "
                          f"|方向锁={c.direction} |占用={list(c.occupied_by)} "
                          f"|等待飞机={waiting_agents}")
                    # 开启后续采集模式：将前缓冲区内帧 + 后续 N 帧一起存储
                    _corridor_capturing[c.id] = _corridor_window
                    event_entry = {
                        "corridor_id": c.id,
                        "event_step": step,
                        "corridor_nodes": list(c.nodes_seq),
                        "frames": list(_corridor_buffer),   # 建存廊道前帧
                    }
                    _corridor_log.append(event_entry)

        # Step 3：继续采集已触发廊道的后续帧
        for cid in list(_corridor_capturing.keys()):
            _corridor_capturing[cid] -= 1
            # 对应的 event_entry 在 _corridor_log 中，直接 append
            for ev in _corridor_log:
                if ev["corridor_id"] == cid:
                    ev["frames"].append(_cur_frame)
                    break
            if _corridor_capturing[cid] <= 0:
                _corridor_captured.add(cid)
                del _corridor_capturing[cid]
                print(f"  [廊道微观] 廊道 {cid} 采集完成，共 {_corridor_window*2} 帧")
        # ───────────────────────────────────────────

        # Safety Shield 后若底层仍检测到碰撞（极端情况），直接重置涉碰飞机而非抛弃
        for i in range(num_total):
            if env.graph_env.crashed[i]:
                tracker.record_collision(agent_id=i)
                env.graph_env.crashed[i]    = False
                env.graph_env.done[i]       = False
                env.graph_env.pos[i]        = starts[i]
                env.graph_env.on_edge[i]    = False
                env.graph_env.edge_idx[i]   = -1
                env.graph_env.progress[i]   = 0.0
                env.graph_env.speed[i]      = 0.0
                env.graph_env.start_delays[i] = DELAY_STEPS
                delay_log[i]    += DELAY_STEPS
                replan_cnt[i]   += 1
                agent_stall_cnt[i] = 0
                tracker.agents[i].spawn_step = None  # 统计口径：下次出发重新计时

        if step % 500 == 0:
            sc = sum(1 for i in range(num_total) if env.graph_env.done[i] and not env.graph_env.crashed[i])
            ac = len(active_agents)
            print(f"  Step {step:<6} | 活跃: {ac:<3} | 成功: {sc:<3}/{num_total}")
            # 检测长时间卡在相同成功数（MAPPO 2架卡死保护）
            if step > 0 and step % 2000 == 0:
                pending_unfinished = [
                    i for i in range(num_total)
                    if not env.graph_env.done[i]
                ]
                if pending_unfinished:
                    all_waiting = all(
                        env.graph_env.start_delays[i] > 0
                        for i in pending_unfinished
                    )
                    if all_waiting:
                        # 所有未完成飞机都在等待，但无路径可到达——强制超时遣返
                        print(f"  [WARN] 所有未完成飞机处于等待状态，强制推出最长等待者")
                        worst = max(pending_unfinished,
                                    key=lambda i: env.graph_env.start_delays[i])
                        env.graph_env.start_delays[worst] = 1

        step += 1

    tracker.finalize(env.graph_env, flights, edge_len_map, delay_log=delay_log)
    tracker.print_summary()
    tracker.export_csv(result_dir)

    # Keep the perturbation timeline beside the metric files so plots can mark
    # the actual event window instead of relying on hard-coded step numbers.
    events_path = os.path.join(result_dir, "MAPPO_events.json")
    with open(events_path, "w", encoding="utf-8") as fout:
        json.dump({
            "algorithm": "MAPPO",
            "scenario": args.scenario,
            "seed": args.seed,
            "time_scale": args.time_scale,
            "max_steps": args.max_steps,
            "events": injector.event_log,
        }, fout, ensure_ascii=False, indent=2)

    # 保存微观轨迹日志（Priority 场景专用）
    if _micro_active and _micro_log:
        micro_path = os.path.join(result_dir, "MAPPO_micro_priority.json")
        with open(micro_path, "w", encoding="utf-8") as fout:
            json.dump({
                "scenario": "priority",
                "event_step": _priority_event_step,
                "window": _micro_window,
                "time_scale": args.time_scale,
                "frames": _micro_log,
            }, fout, ensure_ascii=False, indent=2)
        print(f"  [微观记录] 已保存 {len(_micro_log)} 帧轨迹数据 -> {micro_path}")

    # 保存廊道协议轨迹日志
    if _corridor_log:
        corridor_path = os.path.join(result_dir, "MAPPO_micro_corridor.json")
        with open(corridor_path, "w", encoding="utf-8") as fout:
            json.dump({
                "description": "廊道协议微观轨迹数据",
                "scenario": args.scenario,
                "window_frames": _corridor_window,
                "time_scale_sec": args.time_scale,
                "events": _corridor_log,
            }, fout, ensure_ascii=False, indent=2)
        print(f"  [廊道微观] 已保存 {len(_corridor_log)} 个廊道事件 -> {corridor_path}")
    else:
        print("  [廊道微观] 未检测到廊道等待事件")

    return tracker.get_summary()



# ================================================================
# FCFS+A* 评估
# ================================================================
def run_fcfs(args, flights, config, edge_len_map, result_dir):
    # FCFS uses the identical rolling-slot admission and safety protocol.
    return run_fcfs_rolling(args, flights, config, edge_len_map, result_dir)

    # Legacy all-flights-at-once executor retained below for historical
    # comparison only.  It must not be used for paper results.
    print("\n" + "=" * 62)
    print("  [FCFS+A*] 开始评估（纯空间路径版）...")
    print("=" * 62)

    random.seed(args.seed)
    np.random.seed(args.seed)

    num_total = len(flights)
    config.env.num_agents = num_total
    config.env.max_steps = args.max_steps
    env = MARLEnv(config)

    adj    = build_adj(env.graph_env)
    goals  = [f["goal"]  for f in flights]
    starts = [f["start"] for f in flights]
    t0s    = [f["t0"]    for f in flights]

    h_cache: dict[int, dict] = {}
    def get_h(g):
        if g not in h_cache:
            h_cache[g] = bfs_heuristic_lazy(g, adj)
        return h_cache[g]

    env.reset(starts=starts, goals=goals, delays=t0s)

    scenario_enum = PerturbationScenario(args.scenario)
    injector = ScenarioInjector(scenario_enum, env.graph_env, num_total,
                                total_steps=args.max_steps, seed=args.seed,
                                flights=flights, time_scale=args.time_scale)

    tracker = SimulationTracker(num_total, snapshot_interval=args.snapshot_interval,
                                time_step_sec=args.time_scale, algorithm_name="FCFS")

    # 纯节点序列路径 & 游标（无时间戳，无预留表）
    planned_routes: dict[int, list[int]] = {i: [] for i in range(num_total)}
    route_cursor:   dict[int, int]       = {i: 0  for i in range(num_total)}

    replan_needed = True
    agent_last_pos  = list(env.graph_env.pos)
    agent_last_prog = [0.0] * num_total
    agent_stall_cnt = [0] * num_total
    total_steps_executed = args.max_steps
    # 延误步数（遣返后推迟出发时间）
    DELAY_STEPS = max(1, int(120.0 / args.time_scale))  # 2分钟回避窗口
    delay_log   = [0] * num_total
    replan_cnt  = [0] * num_total  # 每架飞机重规划次数

    # 微观轨迹记录（Priority 场景专用）
    _micro_log: list = []            # 存储每步的微观状态
    _micro_active = (args.scenario == "priority")
    _priority_event_step = -1        # 第一次优先级事件发生的 step
    _micro_window = 300              # 事件前后各 300 步的轨迹窗口

    step = 0
    while step < args.max_steps:
        prev_log_len = len(injector.event_log)
        injector.step(step)
        # 检测到新事件时触发重规划
        new_events = injector.event_log[prev_log_len:]
        if new_events:
            replan_needed = True
            # Closure 事件：强制清空所有已规划路径，硬性要求全量重规划
            for ev in new_events:
                if ev.get("type") == "closure":
                    for i in range(num_total):
                        if not env.graph_env.done[i]:
                            planned_routes[i] = []
                            route_cursor[i] = 0
                    print(f"  [FCFS] Step {step}: 滑行道关闭，已清空所有路径，强制全量重规划")

        tracker.record_step(step, env.graph_env)

        active_agents = [
            i for i in range(num_total)
            if not env.graph_env.done[i] and env.graph_env.start_delays[i] == 0
        ]

        if all(env.graph_env.done):
            total_steps_executed = step
            break

        # 快进（jump>0 跳步跳过env.step；jump==0 放行让环境递减 start_delays）
        if not active_agents:
            pending = [env.graph_env.start_delays[i]
                       for i in range(num_total)
                       if not env.graph_env.done[i] and env.graph_env.start_delays[i] > 0]
            if pending:
                jump = min(pending) - 1
                if jump > 0:
                    for i in range(num_total):
                        if not env.graph_env.done[i]:
                            env.graph_env.start_delays[i] = max(0, env.graph_env.start_delays[i] - jump)
                    step += jump
                    total_steps_executed = step
                    step += 1
                    continue
            # jump==0：放行到 env.step() 让飞机正常降临

        # 死锁检测（严格物理静止检测）
        for i in list(active_agents):
            cur_pos = env.graph_env.pos[i]
            cur_prog = env.graph_env.progress[i] if env.graph_env.on_edge[i] else 0.0

            if cur_pos == agent_last_pos[i] and abs(cur_prog - agent_last_prog[i]) < 1e-4:
                agent_stall_cnt[i] += 1
            else:
                agent_stall_cnt[i] = 0
                agent_last_pos[i]  = cur_pos
                agent_last_prog[i] = cur_prog
                
                # 只有当到达新节点（离开边）时才需要推进 A* 游标
                if not env.graph_env.on_edge[i]:
                    c = route_cursor[i]
                    route = planned_routes[i]
                    while c < len(route) - 1 and route[c] != cur_pos:
                        c += 1
                    route_cursor[i] = c

            if agent_stall_cnt[i] >= args.deadlock_timeout:
                replan_cnt[i]     += 1
                agent_stall_cnt[i] = 0

                # ── 超过最大遣返次数：永久放弃该飞机 ─────────────────────
                if replan_cnt[i] > args.max_deadlock_per_agent:
                    env.graph_env.done[i]    = True
                    env.graph_env.crashed[i] = True
                    print(f"  [FCFS] 飞机 #{i} 第 {replan_cnt[i]} 次死锁，"
                          f"超过上限 {args.max_deadlock_per_agent} 次，永久放弃")
                    tracker.record_deadlock(agent_id=i)
                    active_agents.remove(i)
                    continue
                # ──────────────────────────────────────────────────────────────

                # 死锁遣返：重新入队
                env.graph_env.pos[i]        = starts[i]
                env.graph_env.on_edge[i]    = False
                env.graph_env.edge_idx[i]   = -1
                env.graph_env.progress[i]   = 0.0
                env.graph_env.speed[i]      = 0.0
                env.graph_env.done[i]       = False
                env.graph_env.crashed[i]    = False
                backoff_delay = DELAY_STEPS + random.randint(20, 200)
                env.graph_env.start_delays[i] = backoff_delay
                planned_routes[i] = []
                route_cursor[i]   = 0
                delay_log[i]      += backoff_delay

                print(f"  [FCFS] 飞机 #{i} 第 {replan_cnt[i]} 次死锁，遣返并等待 {backoff_delay} 步以错峰出行")
                tracker.agents[i].spawn_step = None
                replan_needed = True
                active_agents.remove(i)
                tracker.record_deadlock(agent_id=i)

        if args.fcfs_mode == "gate_hold":
            # 模式 A: 停机坪等待制 (Sequential Routing with Gate Holding)
            # 每步都尝试为没有路径的飞机分配纯最短路，若与已有路径冲突则停机坪延误
            closed_edges: set[tuple] = set()
            for e in env.graph_env.edges:
                if e.closed:
                    closed_edges.add((e.start, e.end))
            
            unfinished = sorted(
                [i for i in range(num_total) 
                 if not env.graph_env.done[i] and env.graph_env.start_delays[i] == 0],
                key=lambda i: env.graph_env.priority[i],
                reverse=True # 高优先级先选
            )
            
            # 1. 收集已放行飞机的剩余路径锁定
            locked_nodes = set()
            for i in unfinished:
                if planned_routes[i] and env.graph_env.start_delays[i] == 0:
                    locked_nodes.update(planned_routes[i][route_cursor[i]:])
            
            # 2. 为未分配路径的飞机尝试分配
            did_replan = False
            for i in unfinished:
                if planned_routes[i]:
                    continue
                if not did_replan:
                    tracker.start_replan()
                    did_replan = True
                
                s = starts[i] if env.graph_env.start_delays[i] > 0 else env.graph_env.pos[i]
                g = goals[i]
                if s == g:
                    planned_routes[i] = [s]
                    route_cursor[i] = 0
                    continue
                
                # 纯物理最短路，不考虑 soft_blocked
                route = spatial_astar(s, g, adj, edge_len_map, get_h(g),
                                      soft_blocked=set(),
                                      hard_blocked_edges=closed_edges)
                if route:
                    path_set = set(route)
                    if path_set.intersection(locked_nodes):
                        # 空间冲突！强制在停机坪/当前位置等待
                        env.graph_env.start_delays[i] = max(1, env.graph_env.start_delays[i])
                    else:
                        # 无冲突，允许放行，并将其路径加入锁定池保护
                        planned_routes[i] = route
                        route_cursor[i] = 0
                        locked_nodes.update(route)
            
            if did_replan:
                tracker.end_replan()
            replan_needed = False

        elif replan_needed:
            # 模式 B: 动态绕行制 (Dynamic Detour Routing)
            tracker.start_replan()

            # 车道实时占用节点（当前时刻已在场的飞机所在节点）
            current_occupied: set[int] = set()
            for j in range(num_total):
                if env.graph_env.done[j] or env.graph_env.start_delays[j] > 0:
                    continue
                if not env.graph_env.on_edge[j]:
                    current_occupied.add(env.graph_env.pos[j])

            # 硬约束：物理关闭的边（不可通行）
            closed_edges: set[tuple] = set()
            for e in env.graph_env.edges:
                if e.closed:
                    closed_edges.add((e.start, e.end))

            # 软惩罚：当前占用节点（优先绕行，但不强制）
            OCCUPANCY_PENALTY = 50.0

            unfinished = sorted(
                [i for i in range(num_total) if not env.graph_env.done[i]],
                key=lambda i: env.graph_env.priority[i],
            )

            for i in unfinished:
                # 若路径已存在（不论是否入场），不重新规划
                # 注：start_delays>0 的飞机路径从起点规划，入场后直接使用
                if planned_routes[i]:
                    continue
                s = starts[i] if env.graph_env.start_delays[i] > 0 else env.graph_env.pos[i]
                g = goals[i]
                if s == g:
                    planned_routes[i] = [s]
                    route_cursor[i]   = 0
                    continue
                # 软阻塞 = 现有占用（起点/终点除外）
                soft_blocked = current_occupied - {s, g}
                route = spatial_astar(s, g, adj, edge_len_map, get_h(g),
                                      soft_blocked=soft_blocked,
                                      hard_blocked_edges=closed_edges)  # 关闭边为硬约束
                if route:
                    planned_routes[i] = route
                    route_cursor[i]   = 0

            tracker.end_replan()
            replan_needed = False

        # ── 动作提取（纯节点序列）──
        actions = [7] * num_total
        for i in active_agents:
            if env.graph_env.on_edge[i]:
                # 飞机已在滑行道上行进中：自动补发加速动作（a=6），让飞机以最大巡航速度行进
                # FCFS 不发加速指令，导致飞机以 gate 限速 0.15 unit/step 极低速爬行
                actions[i] = 6
                continue
            if env.graph_env.delayed[i] and env.graph_env.delay_timers[i] > 0:
                continue
            route = planned_routes[i]
            if not route:
                continue
            cur_pos = env.graph_env.pos[i]
            c = route_cursor[i]
            while c < len(route) - 1 and route[c] != cur_pos:
                c += 1
            route_cursor[i] = c
            if c >= len(route) - 1:
                continue
            next_node = route[c + 1]
            for a_idx, edge_info in enumerate(env.graph_env.adj[cur_pos]):
                nb = edge_info[0] if isinstance(edge_info, tuple) else edge_info
                if nb == next_node:
                    actions[i] = a_idx
                    break

        # Safety Shield：防止两架飞机被规划到同一节点（纯规划冲突，不是物理事故）
        occupied_nodes = set()
        occupied_edges = set()
        for j in range(num_total):
            if env.graph_env.done[j] or env.graph_env.start_delays[j] > 0:
                continue
            if env.graph_env.on_edge[j]:
                e = env.graph_env.edges[env.graph_env.edge_idx[j]]
                occupied_edges.add((e.start, e.end))
                # 移除 occupied_nodes.add(e.end)，允许同向尾随进入同一条边
            # 移除静态的 occupied_nodes.add(pos) 以允许首尾相连的同步移动

        sorted_active = sorted(active_agents, key=lambda x: env.graph_env.priority[x], reverse=True)
        for i in sorted_active:
            a = actions[i]
            if a >= 5 or env.graph_env.on_edge[i]:
                if not env.graph_env.on_edge[i]:
                    occupied_nodes.add(env.graph_env.pos[i])
                continue
            nbrs = env.graph_env.adj[env.graph_env.pos[i]]
            if a >= len(nbrs):
                if not env.graph_env.on_edge[i]:
                    occupied_nodes.add(env.graph_env.pos[i])
                continue
            nxt, _ = nbrs[a]
            cur = env.graph_env.pos[i]
            if (nxt in occupied_nodes and nxt != cur) or (nxt, cur) in occupied_edges:
                actions[i] = 7
                occupied_nodes.add(cur)
            else:
                occupied_nodes.add(nxt)
                occupied_edges.add((cur, nxt))

        # ATC Gate Holding: 限制场面并发飞机数 (匹配训练环境的 num_agents=8)
        current_active = len(active_agents)
        for i in range(num_total):
            if not env.graph_env.done[i] and env.graph_env.start_delays[i] == 1:
                if current_active >= args.active_limit:
                    env.graph_env.start_delays[i] += 1  # 延长等待，暂不释放
                else:
                    current_active += 1

        obs, rewards, dones, masks, info = env.step(actions)

        # ── 微观轨迹采集（Priority 场景专用） ──
        if _micro_active:
            ev_log = injector.event_log
            if _priority_event_step < 0:
                for ev in ev_log:
                    if ev.get("type") == "priority_boost":
                        _priority_event_step = ev["step"]
                        print(f"  [微观记录] 检测到优先级事件@Step {_priority_event_step}，窗口 +/-{_micro_window} 步")
                        break
            if _priority_event_step >= 0 and abs(step - _priority_event_step) <= _micro_window:
                boosted_set = set(injector.get_boosted_agents())
                frame = {"step": step, "boosted": list(boosted_set), "agents": []}
                for j in range(num_total):
                    if env.graph_env.done[j] or env.graph_env.start_delays[j] > 0:
                        continue
                    frame["agents"].append({
                        "id": j,
                        "pos": env.graph_env.pos[j],
                        "on_edge": int(env.graph_env.on_edge[j]),
                        "progress": round(env.graph_env.progress[j] if env.graph_env.on_edge[j] else 0.0, 4),
                        "speed": round(env.graph_env.speed[j], 4),
                        "action": actions[j],
                        "is_vip": int(j in boosted_set),
                        "priority": env.graph_env.priority[j],
                    })
                _micro_log.append(frame)

        # 碰撞兜底重置（视为推出失败重新入队，spawn_step重置）
        for i in range(num_total):
            if env.graph_env.crashed[i]:
                tracker.record_collision(agent_id=i)
                env.graph_env.crashed[i]    = False
                env.graph_env.done[i]       = False
                env.graph_env.pos[i]        = starts[i]
                env.graph_env.on_edge[i]    = False
                env.graph_env.edge_idx[i]   = -1
                env.graph_env.progress[i]   = 0.0
                env.graph_env.speed[i]      = 0.0
                env.graph_env.start_delays[i] = DELAY_STEPS
                planned_routes[i] = []
                route_cursor[i]   = 0
                delay_log[i]    += DELAY_STEPS
                replan_cnt[i]   += 1
                agent_stall_cnt[i] = 0
                # 重置spawn_step，taxi_time从下次出发重新计时
                tracker.agents[i].spawn_step = None
                replan_needed = True

        if step % 500 == 0:
            sc = sum(1 for i in range(num_total) if env.graph_env.done[i] and not env.graph_env.crashed[i])
            ac = len(active_agents)
            print(f"  Step {step:<6} | 活跃:{ac:<3} | 成功:{sc:<3}/{num_total}")
            # 检测所有未完成飞机都卡在等待中（FCFS 卡死保护）
            if step > 0 and step % 2000 == 0:
                pending_unfinished = [
                    i for i in range(num_total)
                    if not env.graph_env.done[i]
                ]
                if pending_unfinished:
                    all_waiting = all(
                        env.graph_env.start_delays[i] > 0
                        for i in pending_unfinished
                    )
                    if all_waiting:
                        print(f"  [WARN] FCFS: 所有未完成飞机处于等待状态，强制推出最长等待者")
                        worst = max(pending_unfinished,
                                    key=lambda i: env.graph_env.start_delays[i])
                        env.graph_env.start_delays[worst] = 1

        step += 1
    tracker.finalize(env.graph_env, flights, edge_len_map, delay_log=delay_log)
    tracker.print_summary()
    tracker.export_csv(result_dir)

    events_path = os.path.join(result_dir, "FCFS_events.json")
    with open(events_path, "w", encoding="utf-8") as fout:
        json.dump({
            "algorithm": "FCFS+A*",
            "scenario": args.scenario,
            "seed": args.seed,
            "time_scale": args.time_scale,
            "max_steps": args.max_steps,
            "events": injector.event_log,
        }, fout, ensure_ascii=False, indent=2)

    # 保存微观轨迹日志（Priority 场景专用）
    if _micro_active and _micro_log:
        micro_path = os.path.join(result_dir, "FCFS_micro_priority.json")
        with open(micro_path, "w", encoding="utf-8") as fout:
            json.dump({
                "scenario": "priority",
                "event_step": _priority_event_step,
                "window": _micro_window,
                "time_scale": args.time_scale,
                "frames": _micro_log,
            }, fout, ensure_ascii=False, indent=2)
        print(f"  [微观记录] FCFS 已保存 {len(_micro_log)} 帧轨迹数据 -> {micro_path}")

    return tracker.get_summary()




# ================================================================
# 生成论文对比表格
# ================================================================
def print_comparison_table(mappo_s: dict, fcfs_s: dict, result_dir: str):
    def fmt(v, suffix=""):
        if v == "N/A" or v is None:
            return "N/A"
        return f"{v}{suffix}"

    lines = []
    lines.append("\n" + "=" * 76)
    lines.append("  论文表格 4.2  基准场景性能对比（Baseline Scene Performance）")
    lines.append("=" * 76)
    lines.append(f"  {'指标':<22} {'FCFS+A*':<22} {'MAPPO (本文)':<22}")
    lines.append("  " + "-" * 70)

    rows = [
        ("安全完航率 (%)",        fmt(fcfs_s['completion_rate_pct'], "%"),  fmt(mappo_s['completion_rate_pct'], "%")),
        ("最终服务率 (%)",        fmt(fcfs_s.get('eventual_completion_rate_pct', fcfs_s['completion_rate_pct']), "%"),
                                  fmt(mappo_s.get('eventual_completion_rate_pct', mappo_s['completion_rate_pct']), "%")),
        ("碰撞事件数",             fmt(fcfs_s['collision_count']),           fmt(mappo_s['collision_count'])),
        ("平均实际移动时间 (min)",  fmt(fcfs_s.get('avg_moving_min', 'N/A'), " min"), fmt(mappo_s.get('avg_moving_min', 'N/A'), " min")),
        ("平均在途等待时间 (min)",  fmt(fcfs_s.get('avg_onroute_wait_min', 'N/A'), " min"), fmt(mappo_s.get('avg_onroute_wait_min', 'N/A'), " min")),
        ("平均总在途时间 (min)",   fmt(fcfs_s['avg_taxi_min'], " min"),      fmt(mappo_s['avg_taxi_min'], " min")),
        ("平均遣返延误时间 (min)",  fmt(fcfs_s.get('avg_delay_min', 'N/A'), " min"), fmt(mappo_s.get('avg_delay_min', 'N/A'), " min")),
        ("平均滑行距离 (拓扑单位)", fmt(fcfs_s['avg_path_dist']),             fmt(mappo_s['avg_path_dist'])),
        ("系统吞吐率 (架/小时)",    fmt(fcfs_s['throughput_per_hour']),       fmt(mappo_s['throughput_per_hour'])),
        ("平均推理/重规划延迟(ms)", fmt(fcfs_s['avg_replan_ms'], " ms"),      fmt(mappo_s['avg_inference_ms'], " ms")),
        ("最大推理/重规划延迟(ms)", fmt(fcfs_s['max_replan_ms'], " ms"),      fmt(mappo_s['max_inference_ms'], " ms")),
        ("重规划次数",             fmt(fcfs_s['replan_count'], " 次"),        fmt(mappo_s.get('replan_count', 0), " 次")),
    ]
    for name, fcfs_v, mappo_v in rows:
        lines.append(f"  {name:<22} {fcfs_v:<22} {mappo_v:<22}")

    lines.append("=" * 76)

    # Markdown 格式
    md_lines = [
        "| 指标 | FCFS+A* | MAPPO（本文）|",
        "| :--- | :--- | :--- |",
    ]
    for name, fcfs_v, mappo_v in rows:
        md_lines.append(f"| {name} | {fcfs_v} | {mappo_v} |")

    output = "\n".join(lines)
    print(output)
    print("\nMarkdown 格式：")
    print("\n".join(md_lines))

    # 保存到文件
    save_path = os.path.join(result_dir, "comparison_table.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output + "\n\n")
        f.write("Markdown 格式：\n")
        f.write("\n".join(md_lines))
    print(f"\n  [OK] 对比表格已保存: {save_path}")


# ================================================================
# 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="论文实验 4.2：FCFS+A* vs MAPPO 基准对比")
    parser.add_argument("--config",     type=str,   default="configs/improved_v2.yaml")
    parser.add_argument("--model",      type=str,   default="checkpoints/4.23.pt",
                        help="MAPPO 模型权重路径")
    parser.add_argument("--excel",      type=str,   default="data.xlsx",help="data.xlsx,每仿真步对应的物理秒数")
    parser.add_argument("--scenario",   type=str,   default="baseline",
                        choices=["baseline", "delay", "closure", "priority"],
                        help="扰动场景（baseline=4.2节，其余=4.3节）")
    parser.add_argument("--max-steps",  type=int,   default=50000)
    parser.add_argument("--time-scale", type=float, default=1.127,
                        help="每仿真步对应的物理秒数（需与 simulate_continuous 保持一致）")
    parser.add_argument("--schedule-compression", type=float, default=0.10,
                        help="计划释放间隔缩放系数。1.0 保留原始时刻；小于 1.0 构造受控高负荷场景")
    parser.add_argument("--fcfs-mode",  type=str,   default="gate_hold",
                        choices=["dynamic", "gate_hold"],
                        help="FCFS的执行模式：dynamic(动态绕行), gate_hold(停机坪等待)")
    parser.add_argument("--transit-ratio", type=float, default=0.35)
    parser.add_argument("--start-hour", type=float, default=8.0,
                        help="航班计划筛选起始小时，例如 8.0")
    parser.add_argument("--end-hour", type=float, default=24.0,
                        help="航班计划筛选结束小时，例如 24.0")
    parser.add_argument("--max-flights", type=int, default=None,
                        help="只取筛选后前 N 个航班，用于小规模调试；默认使用全部航班")
    parser.add_argument("--exclude-flight-indices", type=int, nargs="*", default=[],
                        help="显式排除筛选后航班序号，例如 --exclude-flight-indices 100 235 241")
    parser.add_argument("--active-limit", type=int, default=8,
                        help="场面最大并发放行数，应与训练阶段 num_agents 保持一致")
    parser.add_argument("--entry-headway-steps", type=int, default=5,
                        help="同一入口节点连续放行的最小间隔；作为 MAPPO/FCFS 共享 ATC 安全规则")
    parser.add_argument("--deadlock-timeout", type=int, default=180,
                        help="兼容参数；rolling 论文运行不因停滞遣返航空器，航班保留至时间窗结束")
    parser.add_argument("--max-deadlock-per-agent", type=int, default=20,
                        help="兼容参数；rolling 论文运行不使用死锁遣返")
    parser.add_argument("--snapshot-interval", type=int, default=10,
                        help="每隔多少步记录一次吞吐量快照（折线图精度）")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--result-dir", type=str,   default=None,
                        help="输出目录。默认会根据 scenario 自动分配到 results/exp_4_3_xxx")
    parser.add_argument("--skip-fcfs",  action="store_true",
                        help="仅跑 MAPPO，跳过 FCFS（用于快速调试）")
    parser.add_argument("--skip-mappo", action="store_true",
                        help="仅跑 FCFS，跳过 MAPPO")
    args = parser.parse_args()

    if not 0 < args.schedule_compression <= 1.0:
        raise ValueError("--schedule-compression must be in (0, 1].")
    if args.entry_headway_steps < 0:
        raise ValueError("--entry-headway-steps must be non-negative.")

    # 自动分配输出目录，防止不同场景数据互相覆盖
    if args.result_dir is None:
        if args.scenario == "baseline":
            args.result_dir = "results/exp_4_2"
        else:
            args.result_dir = f"results/exp_4_3_{args.scenario}"

    os.makedirs(args.result_dir, exist_ok=True)

    # —— 准备公共数据（两者共享，保证公平对比）——
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = Config.from_yaml(args.config)
    flights = load_full_flight_schedule(
        args.excel,
        time_scale=args.time_scale,
        transit_ratio=args.transit_ratio,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )
    if not flights:
        print("错误：未找到有效航班数据，请检查 --excel 路径。")
        return
        
    source_target_count = sum(
        str(f.get("agent_role", "TARGET")).upper() == "TARGET" for f in flights
    )
    source_transit_count = sum(
        str(f.get("agent_role", "TARGET")).upper() == "TRANSIT" for f in flights
    )
    if not args.allow_noncanonical_schedule and (source_target_count, source_transit_count) != (271, 92):
        raise RuntimeError(
            "Rolling evaluation requires the complete 08:00-24:00 schedule "
            "(271 TARGET + 92 TRANSIT = 363 movements). "
            f"Loaded {source_target_count} TARGET + {source_transit_count} TRANSIT "
            f"from {args.excel}. Use data.xlsx or explicitly pass "
            "--allow-noncanonical-schedule for a debug/demo run."
        )

    if args.exclude_flight_indices:
        removed = []
        for idx in sorted(set(args.exclude_flight_indices), reverse=True):
            if 0 <= idx < len(flights):
                f = flights.pop(idx)
                removed.append((idx, f))
            else:
                print(f"  [数据清洗] 跳过越界航班序号 {idx}，当前航班数 {len(flights)}")
        for idx, f in sorted(removed, key=lambda item: item[0]):
            desc = f"{f.get('time_str', '?')} {f.get('airline', '?')} {f.get('type', '?')}-{f.get('gate', '?')} {f.get('start', '?')}->{f.get('goal', '?')}"
            print(f"  [数据清洗] 已删除航班序号 {idx}: {desc}")

    if args.max_flights is not None:
        if args.max_flights <= 0:
            print("错误：--max-flights 必须为正整数。")
            return
        flights = flights[:args.max_flights]
        print(f"  [数据清洗] 已截取前 {len(flights)} 个航班用于本次实验")

    # 临时 env 用于构造 edge_len_map
    _tmp_config = Config.from_yaml(args.config)
    if args.active_limit != _tmp_config.env.num_agents:
        raise ValueError(
            f"--active-limit ({args.active_limit}) must match the trained policy population "
            f"in config ({_tmp_config.env.num_agents})."
        )
    _tmp_env = MARLEnv(_tmp_config)
    edge_len_map = build_edge_length_map(_tmp_env.graph_env)
    del _tmp_env

    print(f"\n[实验配置]")
    print(f"  航班总数  : {len(flights)} 架次")
    print(f"  扰动场景  : {args.scenario}")
    print(f"  时间步长  : {args.time_scale} 秒/步")
    print(f"  计划压缩  : {args.schedule_compression}x")
    print(f"  时间窗口  : {args.start_hour} - {args.end_hour}")
    print(f"  并发上限  : {args.active_limit} 架")
    print(f"  结果目录  : {args.result_dir}")

    # —— 运行对比 ——
    # Machine-readable manifest for plotting and reproducibility.
    with open(os.path.join(args.result_dir, "experiment_meta.json"), "w", encoding="utf-8") as fout:
        json.dump({
            "scenario": args.scenario,
            "seed": args.seed,
            "config": args.config,
            "model": args.model,
            "excel": args.excel,
            "flight_count": len(flights),
            "time_scale": args.time_scale,
            "schedule_compression": args.schedule_compression,
            "start_hour": args.start_hour,
            "end_hour": args.end_hour,
            "max_steps": args.max_steps,
            "active_limit": args.active_limit,
            "policy_slots": args.active_limit,
            "execution_protocol": "fixed_slot_rolling",
            "safety_filter": "shared_deterministic_atc_filter",
            "retry_policy": "no training recovery; stalled flights remain active until the operational horizon",
            "entry_headway_steps": args.entry_headway_steps,
        }, fout, ensure_ascii=False, indent=2)

    mappo_summary = None
    fcfs_summary = None

    if not args.skip_mappo:
        mappo_summary = run_mappo(args, flights, Config.from_yaml(args.config),
                                  edge_len_map, args.result_dir)

    if not args.skip_fcfs:
        fcfs_summary = run_fcfs(args, flights, Config.from_yaml(args.config),
                                edge_len_map, args.result_dir)

    # —— 对比表格 ——
    if mappo_summary and fcfs_summary:
        print_comparison_table(mappo_summary, fcfs_summary, args.result_dir)
    elif mappo_summary:
        print("\n[跳过 FCFS] 仅 MAPPO 结果：")
        import pprint; pprint.pprint(mappo_summary)
    elif fcfs_summary:
        print("\n[跳过 MAPPO] 仅 FCFS 结果：")
        import pprint; pprint.pprint(fcfs_summary)


if __name__ == "__main__":
    main()
