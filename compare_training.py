"""
训练结果对比与可视化工具

使用方法:
    python compare_training.py --baseline checkpoints/training_log.csv --improved checkpoints/training_log_improved.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(csv_path: str) -> pd.DataFrame:
    """加载训练日志CSV文件。"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded: {csv_path} ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return None


def plot_single_training(df: pd.DataFrame, title: str = "Training Progress", save_path: str = None):
    """绘制单个训练的进度曲线。"""
    if df is None or len(df) == 0:
        print("❌ No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 奖励曲线
    ax = axes[0, 0]
    ax.plot(df['episode'], df['total_reward'], alpha=0.3, label='Raw Reward', linewidth=0.8)
    
    # 计算滑动平均
    if len(df) >= 100:
        rolling_mean = df['total_reward'].rolling(window=100, center=True).mean()
        ax.plot(df['episode'], rolling_mean, label='Avg100', linewidth=2, color='red')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 完成率
    ax = axes[0, 1]
    ax.plot(df['episode'], df['completed'], linewidth=1.5, color='green')
    ax.fill_between(df['episode'], 0, df['completed'], alpha=0.3, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Completed Agents')
    ax.set_title('Completion Rate')
    ax.set_ylim([-0.5, 4.5])
    ax.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Loss曲线
    ax = axes[1, 0]
    if 'actor_loss' in df.columns:
        ax.plot(df['episode'], df['actor_loss'], label='Actor Loss', alpha=0.7, linewidth=1)
    if 'critic_loss' in df.columns:
        ax.plot(df['episode'], df['critic_loss'], label='Critic Loss', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 子项奖励
    ax = axes[1, 1]
    reward_cols = ['completion', 'proximity', 'team_bonus', 'efficiency', 'deadlock']
    for col in reward_cols:
        if col in df.columns:
            rolling = df[col].rolling(window=100).mean() if len(df) >= 100 else df[col]
            ax.plot(df['episode'], rolling, label=col, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sub-reward (Avg100)')
    ax.set_title('Sub-reward Components')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


def plot_comparison(df_baseline: pd.DataFrame, df_improved: pd.DataFrame, save_path: str = None):
    """对比绘制两个训练配置。"""
    if df_baseline is None or df_improved is None:
        print("❌ Missing data for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Comparison: Baseline vs Improved', fontsize=16, fontweight='bold')
    
    # 1. 奖励对比
    ax = axes[0, 0]
    if len(df_baseline) >= 100:
        baseline_avg = df_baseline['total_reward'].rolling(100).mean()
        ax.plot(df_baseline['episode'], baseline_avg, label='Baseline (Avg100)', linewidth=2, color='red', alpha=0.7)
    
    if len(df_improved) >= 100:
        improved_avg = df_improved['total_reward'].rolling(100).mean()
        ax.plot(df_improved['episode'], improved_avg, label='Improved (Avg100)', linewidth=2, color='green', alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Comparison')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 完成率对比
    ax = axes[0, 1]
    ax.plot(df_baseline['episode'], df_baseline['completed'], label='Baseline', linewidth=1.5, alpha=0.7)
    ax.plot(df_improved['episode'], df_improved['completed'], label='Improved', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Completed Agents')
    ax.set_title('Completion Rate Comparison')
    ax.set_ylim([-0.5, 4.5])
    ax.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Loss对比
    ax = axes[1, 0]
    if 'critic_loss' in df_baseline.columns:
        baseline_loss = df_baseline['critic_loss'].rolling(100).mean()
        ax.plot(df_baseline['episode'], baseline_loss, label='Baseline Critic Loss', linewidth=1.5, alpha=0.7)
    
    if 'critic_loss' in df_improved.columns:
        improved_loss = df_improved['critic_loss'].rolling(100).mean()
        ax.plot(df_improved['episode'], improved_loss, label='Improved Critic Loss', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Critic Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title('Loss Stability Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 改善指标
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = "📊 Key Metrics Comparison\n" + "=" * 40 + "\n"
    
    # 计算关键指标
    if len(df_baseline) > 0 and len(df_improved) > 0:
        # 最后100episode平均
        baseline_final_avg = df_baseline['total_reward'].tail(100).mean()
        improved_final_avg = df_improved['total_reward'].tail(100).mean()
        improvement_pct = ((improved_final_avg - baseline_final_avg) / abs(baseline_final_avg) * 100) if baseline_final_avg != 0 else 0
        
        stats_text += f"Final Avg100 Reward:\n"
        stats_text += f"  Baseline: {baseline_final_avg:.2f}\n"
        stats_text += f"  Improved: {improved_final_avg:.2f}\n"
        stats_text += f"  Change:   {improvement_pct:+.1f}%\n\n"
        
        # 完成率
        baseline_final_done = df_baseline['completed'].tail(100).mean()
        improved_final_done = df_improved['completed'].tail(100).mean()
        
        stats_text += f"Final Avg Completion Rate:\n"
        stats_text += f"  Baseline: {baseline_final_done:.2f} agents\n"
        stats_text += f"  Improved: {improved_final_done:.2f} agents\n"
        stats_text += f"  Change:   {improved_final_done - baseline_final_done:+.2f}\n\n"
        
        # Loss稳定性
        if 'critic_loss' in df_baseline.columns and 'critic_loss' in df_improved.columns:
            baseline_loss_std = df_baseline['critic_loss'].tail(100).std()
            improved_loss_std = df_improved['critic_loss'].tail(100).std()
            
            stats_text += f"Final Loss Stability (std):\n"
            stats_text += f"  Baseline: {baseline_loss_std:.2f}\n"
            stats_text += f"  Improved: {improved_loss_std:.2f}\n"
            stats_text += f"  Reduced:  {baseline_loss_std - improved_loss_std:+.2f}\n"
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


def print_summary(df: pd.DataFrame, name: str = "Training"):
    """打印训练摘要统计。"""
    if df is None or len(df) == 0:
        return
    
    print(f"\n📊 {name} Summary")
    print("=" * 60)
    
    total_episodes = len(df)
    print(f"Total Episodes:        {total_episodes}")
    print(f"Total Steps:           {df['steps'].sum():.0f}")
    
    final_100_episodes = df.tail(100) if len(df) >= 100 else df
    
    print(f"\nFinal 100 Episodes Statistics:")
    print(f"  Avg Reward:          {final_100_episodes['total_reward'].mean():.2f}")
    print(f"  Std Reward:          {final_100_episodes['total_reward'].std():.2f}")
    print(f"  Max Reward:          {final_100_episodes['total_reward'].max():.2f}")
    print(f"  Min Reward:          {final_100_episodes['total_reward'].min():.2f}")
    print(f"  Avg Completion:      {final_100_episodes['completed'].mean():.2f} agents")
    
    if 'critic_loss' in df.columns:
        print(f"  Avg Critic Loss:     {final_100_episodes['critic_loss'].mean():.4f}")
        print(f"  Max Critic Loss:     {final_100_episodes['critic_loss'].max():.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare and visualize training results")
    parser.add_argument("--baseline", type=str, help="Path to baseline training log")
    parser.add_argument("--improved", type=str, help="Path to improved training log")
    parser.add_argument("--single", type=str, help="Path to single training log to visualize")
    parser.add_argument("--output", type=str, default="./", help="Output directory for plots")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.baseline and args.improved:
        # 对比模式
        print("🔍 Comparison Mode")
        df_baseline = load_training_log(args.baseline)
        df_improved = load_training_log(args.improved)
        
        print_summary(df_baseline, "Baseline")
        print_summary(df_improved, "Improved")
        
        if df_baseline is not None and df_improved is not None:
            plot_comparison(
                df_baseline, df_improved,
                save_path=str(output_dir / "comparison.png")
            )
    elif args.single:
        # 单个可视化模式
        print("📈 Single Training Visualization")
        df = load_training_log(args.single)
        print_summary(df)
        
        if df is not None:
            plot_single_training(
                df, 
                title="Training Progress",
                save_path=str(output_dir / "training_progress.png")
            )
    else:
        # 默认：尝试加载改进和原始日志进行对比
        baseline_path = "checkpoints/training_log.csv"
        improved_path = "checkpoints/training_log_improved.csv"
        
        if Path(baseline_path).exists() and Path(improved_path).exists():
            print("📈 Auto-loading baseline and improved logs")
            args.baseline = baseline_path
            args.improved = improved_path
            df_baseline = load_training_log(args.baseline)
            df_improved = load_training_log(args.improved)
            
            print_summary(df_baseline, "Baseline")
            print_summary(df_improved, "Improved")
            
            if df_baseline is not None and df_improved is not None:
                plot_comparison(
                    df_baseline, df_improved,
                    save_path=str(output_dir / "comparison.png")
                )
        else:
            print("ℹ️ Usage examples:")
            print("  # 对比两个训练结果")
            print("  python compare_training.py --baseline logs/baseline.csv --improved logs/improved.csv")
            print("\n  # 可视化单个训练")
            print("  python compare_training.py --single logs/training.csv")
            print("\n  # 自动加载默认日志（如果存在）")
            print("  python compare_training.py")
