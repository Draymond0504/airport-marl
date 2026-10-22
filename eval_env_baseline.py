import argparse
import numpy as np
from configs.config import Config
from env.marl_env import MARLEnv

def run_random_baseline(env, episodes=1000):
    print("=" * 50)
    print("第一步：运行「随机策略基线测试」 (跑 1000 个 episode)")
    print("=" * 50)
    n_success = 0
    for ep in range(1, episodes + 1):
        obs, masks = env.reset()
        done = False
        step_count = 0
        while not done:
            acts = [np.random.choice(np.where(m > 0)[0]) for m in masks]
            obs, rewards, dones, next_masks, info = env.step(acts)
            masks = next_masks
            step_count += 1
            # 停止条件：所有飞机都done或者超过最大步数
            # 如果 MARL env dones 是布尔数组
            done = all(dones) or step_count >= env.graph_env.max_steps
            
        ep_goal_reached = info.get("goal_reached", [False] * env.num_agents)
        if all(ep_goal_reached):
            n_success += 1
            
        if ep % 100 == 0:
            print(f"进度: {ep}/{episodes}, 全队到达次数: {n_success}")
            
    success_rate = (n_success / episodes) * 100
    print(f"\n随机策略完成率: {success_rate:.1f}%")
    
    print("\n判断标准结论:")
    if success_rate > 0.1:
        print("-> > 0.1%: 环境可达，问题在奖励引导不足")
    else:
        print("-> ≈ 0%: 环境本身存在设计缺陷（路径不可达/任务过难）")
    print("=" * 50 + "\n")
    return success_rate

def run_astar_expert_baseline(env, episodes=100):
    print("=" * 50)
    print("第二步：「A* 专家策略」对照测试 (跑 100 个 episode)")
    print("=" * 50)
    n_success = 0
    for ep in range(1, episodes + 1):
        obs, masks = env.reset()
        done = False
        step_count = 0
        
        while not done:
            acts = []
            for i, agent in enumerate(env.agents):
                # 确保获取当前所在节点，兼容可能是 int 或其他情况
                # 很多时候 pos 保存在 env.graph_env.pos 或 agent.pos
                current_node = env.graph_env.pos[i] if hasattr(env.graph_env, 'pos') else agent.pos
                
                # 让每个 agent 按照 planned_path 走下一步
                try:
                    next_node = agent.get_next_planned_node(env.graph_env)
                except AttributeError:
                    # 获取下一步路径如果遇到问题，尝试兼容其他调用方式
                    if hasattr(agent, 'planned_path') and len(agent.planned_path) > 0:
                        # 找到当前节点在 planned_path 的位置，并返回下一个节点
                        try:
                            idx = agent.planned_path.index(current_node)
                            next_node = agent.planned_path[min(idx + 1, len(agent.planned_path) - 1)]
                        except ValueError:
                            next_node = current_node
                    else:
                        next_node = current_node

                # 在 adj 中找这个 next_node 对应的动作 idx
                # 如果没有 adj 属性，可能需要通过 get_valid_actions
                # 根据提示代码：
                try:
                    nbrs = env.graph_env.adj[current_node]
                except AttributeError:
                    # 如果用的是 adjacency_list 等
                    nbrs = env.graph_env.graph[current_node] if hasattr(env.graph_env, "graph") else []
                
                act = 7 # 默认可能是 7 (Wait动作)
                for idx, edge_info in enumerate(nbrs):
                    # 取出节点编号(兼容 (node, attr) 或仅 node)
                    n = edge_info[0] if isinstance(edge_info, tuple) else edge_info
                    if n == next_node:
                        act = idx
                        break
                acts.append(act)
                
            obs, rewards, dones, masks, info = env.step(acts)
            step_count += 1
            done = all(dones) or step_count >= env.graph_env.max_steps
            
        ep_goal_reached = info.get("goal_reached", [False] * env.num_agents)
        if all(ep_goal_reached):
            n_success += 1
            
        if ep % 10 == 0:
            print(f"进度: {ep}/{episodes}, 全队到达次数: {n_success}")
            
    success_rate = (n_success / episodes) * 100
    print(f"\nA* 专家完成率: {success_rate:.1f}%")
    
    print("\n判断标准结论:")
    if success_rate >= 60:
        print("-> > 60%: 环境可通，RL 策略没学好，重点查奖励函数")
    elif 10 <= success_rate < 60:
        print("-> 10-60%: 路径规划已是瓶颈，需增大 max_steps 或减少 agent")
    else:
        print("-> < 10%: 环境存在根本性阻断，如走廊死锁、路径断路")
    print("=" * 50 + "\n")
    return success_rate

def main():
    parser = argparse.ArgumentParser(description="Baseline Test for MARL Env")
    parser.add_argument("--config", type=str, default="configs/improved_v2.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    env = MARLEnv(config)
    
    run_random_baseline(env, episodes=1000)
    run_astar_expert_baseline(env, episodes=100)

if __name__ == "__main__":
    main()
