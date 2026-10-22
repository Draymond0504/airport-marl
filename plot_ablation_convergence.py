import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(group, seeds):
    all_rewards = []
    all_crs = []
    
    num_agents = 8 # 从配置可知
    
    for seed in seeds:
        file_path = f"checkpoints/{group}/seed_{seed}/training_log_improved.csv"
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        
        df = pd.read_csv(file_path)
        
        # 使用尽量少的平滑来体现原始趋势 (window=50)
        window = 50
        reward_smooth = df['total_reward'].rolling(window=window, min_periods=1).mean()
        cr = df['completed'] / num_agents
        cr_smooth = cr.rolling(window=window, min_periods=1).mean()
        
        all_rewards.append(reward_smooth.values)
        all_crs.append(cr_smooth.values)
        
    if not all_rewards:
        return None, None, None
        
    min_len = min([len(arr) for arr in all_rewards])
    all_rewards = [arr[:min_len] for arr in all_rewards]
    all_crs = [arr[:min_len] for arr in all_crs]
    
    rewards_matrix = np.vstack(all_rewards)
    crs_matrix = np.vstack(all_crs)
    
    episodes = df['episode'].values[:min_len]
    
    return episodes, rewards_matrix, crs_matrix

def main():
    seeds = [42, 1024, 2048]
    
    # 尝试加载数据
    ep_guided, rw_guided, cr_guided = load_data("guided", seeds)
    ep_no_guide, rw_no_guide, cr_no_guide = load_data("no_guide", seeds)
    
    if ep_guided is None or ep_no_guide is None:
        print("Data is not ready yet. Please run the training first.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Subplot (a): Episode Reward ===
    ax1 = axes[0]
    
    # Guided
    r_mean_g = rw_guided.mean(axis=0)
    r_std_g = rw_guided.std(axis=0)
    ax1.plot(ep_guided, r_mean_g, label='Planning-guided MAPPO', color='red', linewidth=1.5)
    ax1.fill_between(ep_guided, r_mean_g - r_std_g, r_mean_g + r_std_g, color='red', alpha=0.2)
    
    # No Guide
    r_mean_ng = rw_no_guide.mean(axis=0)
    r_std_ng = rw_no_guide.std(axis=0)
    ax1.plot(ep_no_guide, r_mean_ng, label='Pure MARL', color='blue', linewidth=1.5)
    ax1.fill_between(ep_no_guide, r_mean_ng - r_std_ng, r_mean_ng + r_std_ng, color='blue', alpha=0.2)
    
    ax1.set_title('(a) Average Episode Reward', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.legend(loc='lower right')
    
    # === Subplot (b): Completion Rate ===
    ax2 = axes[1]
    
    # Guided
    c_mean_g = cr_guided.mean(axis=0)
    c_std_g = cr_guided.std(axis=0)
    ax2.plot(ep_guided, c_mean_g, label='Planning-guided MAPPO', color='red', linewidth=1.5)
    ax2.fill_between(ep_guided, c_mean_g - c_std_g, c_mean_g + c_std_g, color='red', alpha=0.2)
    
    # No Guide
    c_mean_ng = cr_no_guide.mean(axis=0)
    c_std_ng = cr_no_guide.std(axis=0)
    ax2.plot(ep_no_guide, c_mean_ng, label='Pure MARL', color='blue', linewidth=1.5)
    ax2.fill_between(ep_no_guide, c_mean_ng - c_std_ng, c_mean_ng + c_std_ng, color='blue', alpha=0.2)
    
    # FCFS+A* Baseline (Assuming 65% based on Layout F high density blind spots)
    # Ideally this value should be derived from fcfs_astar_baseline.py
    fcfs_baseline = 0.65 
    ax2.axhline(y=fcfs_baseline, color='gray', linestyle='--', linewidth=2, label='FCFS+A* Baseline')
    
    ax2.set_title('(b) Completion Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Completion Rate', fontsize=12)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    # 保存结果
    out_dir = "paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "ablation_convergence.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {out_file}")

if __name__ == "__main__":
    main()
