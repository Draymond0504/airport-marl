import argparse
import random
import numpy as np

from configs.config import Config
from env.marl_env import MARLEnv
from trainer.mappo_improved import MAPPOTrainerImproved

def explore_capacity():
    """探究智能体运行数量瓶颈的压测脚本"""
    parser = argparse.ArgumentParser(description="运行数量/吞吐量容量探究")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="checkpoints/4.20.pt")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    print("开始智能体承载能力探测（压力测试）...\n")
    
    agent_counts = [4, 6, 8, 10, 12, 15, 20]
    episodes_per_N = 30
    results = []

    for n in agent_counts:
        # 重分配配置和环境
        config.env.num_agents = n
        env = MARLEnv(config)
        
        trainer = MAPPOTrainerImproved(
            num_agents=n,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            global_state_dim=env.global_state_dim,
            hidden_dim=config.training.hidden_dim,
        )
        try:
            trainer.load(args.model)
        except Exception:
            print(f"加载模型 {args.model} 失败，退出测试。")
            return

        success_count = 0
        total_time_steps = 0
        total_conflicts = 0

        for ep in range(episodes_per_N):
            # 制造高并发压力的发车间距，t0 都在 0~5步之内，极高密度
            delays = [random.randint(0, 5) for _ in range(n)]
            obs, masks = env.reset(delays=delays)
            global_state = env.get_global_state()
            
            done_all = False
            step_count = 0
            
            while not done_all:
                actions, _, _ = trainer.select_actions(obs, masks, global_state, deterministic=True)
                obs, rewards, dones, masks, _ = env.step(actions)
                global_state = env.get_global_state()
                step_count += 1
                done_all = all(dones) or step_count >= env.config.env.max_steps
                
            goals_reached = sum(1 for i in range(n) if env.graph_env.agent_positions[i] == env.graph_env.agent_goals[i])
            success_count += goals_reached
            total_time_steps += step_count
            total_conflicts += env.graph_env.conflict_count
            
        avg_success_rate = (success_count / (n * episodes_per_N)) * 100
        avg_steps = total_time_steps / episodes_per_N
        avg_conflicts = total_conflicts / episodes_per_N
        # 近似模型吞吐量(完成架次/总步数)
        throughput = success_count / total_time_steps if total_time_steps > 0 else 0
        
        results.append((n, avg_success_rate, avg_steps, avg_conflicts, throughput))
        print(f"并发N={n:2d} | 成功率: {avg_success_rate:6.2f}% | 平均耗时: {avg_steps:5.1f} | 冲突: {avg_conflicts:4.1f} | 吞吐: {throughput:5.4f}")

    print("\n===== 评估结论建议 =====")
    print("请观察上方的【成功率】数据，如果发生断崖式下跌（例如从 92% 掉到 70% 甚至更低），")
    print("同时【冲突】次数显著上升，该阈值 N 即是模型在这个物理拓扑结构下的极限。")


if __name__ == "__main__":
    explore_capacity()
