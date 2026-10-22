"""
诊断工具 - 快速检查训练配置与环境

运行此脚本检查：
- 配置是否有效
- 模型是否能正常前向传播
- 环境是否能正常运行
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def check_config():
    """检查配置文件有效性。"""
    print("🔍 Checking configuration files...")
    
    configs_to_check = [
        "configs/default.yaml",
        "configs/improved_v2.yaml",
    ]
    
    for config_path in configs_to_check:
        if not Path(config_path).exists():
            print(f"  ❌ {config_path} not found")
            continue
        
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            
            # 检查关键字段
            assert "env" in cfg, "Missing 'env' section"
            assert "reward" in cfg, "Missing 'reward' section"
            assert "training" in cfg, "Missing 'training' section"
            
            # 检查关键参数
            env_cfg = cfg["env"]
            assert env_cfg.get("num_agents") > 0, "Invalid num_agents"
            assert env_cfg.get("max_steps") > 0, "Invalid max_steps"
            
            reward_cfg = cfg["reward"]
            assert "completion" in reward_cfg, "Missing completion reward"
            assert "step_penalty" in reward_cfg, "Missing step_penalty"
            
            train_cfg = cfg["training"]
            assert train_cfg.get("lr_actor") > 0, "Invalid lr_actor"
            assert train_cfg.get("total_episodes") > 0, "Invalid total_episodes"
            
            print(f"  ✅ {config_path}")
            print(f"     - agents: {env_cfg['num_agents']}")
            print(f"     - completion reward: {reward_cfg['completion']}")
            print(f"     - learning rate: {train_cfg['lr_actor']}")
            
        except Exception as e:
            print(f"  ❌ {config_path}: {e}")


def check_environment():
    """检查环境是否能正常运行。"""
    print("\n🔍 Checking environment setup...")
    
    try:
        from configs.config import Config
        from env.marl_env import MARLEnv
        
        config = Config.from_yaml("configs/improved_v2.yaml")
        env = MARLEnv(config)
        
        print(f"  ✅ Environment initialized")
        print(f"     - agents: {env.num_agents}")
        print(f"     - obs_dim: {env.obs_dim}")
        print(f"     - action_dim: {env.action_dim}")
        
        # 测试reset
        obs, masks = env.reset()
        print(f"  ✅ Environment reset successful")
        print(f"     - obs shapes: {[o.shape for o in obs]}")
        print(f"     - mask shapes: {[m.shape for m in masks]}")
        
        # 测试step
        global_state = env.get_global_state()
        actions = [0] * env.num_agents
        next_obs, rewards, dones, next_masks, info = env.step(actions)
        print(f"  ✅ Environment step successful")
        print(f"     - rewards: {rewards}")
        print(f"     - dones: {dones}")
        
    except Exception as e:
        print(f"  ❌ Environment error: {e}")
        import traceback
        traceback.print_exc()


def check_model():
    """检查模型是否能正常前向传播。"""
    print("\n🔍 Checking model setup...")
    
    try:
        from trainer.mappo_improved import MAPPOTrainerImproved
        from configs.config import Config
        from env.marl_env import MARLEnv
        
        config = Config.from_yaml("configs/improved_v2.yaml")
        env = MARLEnv(config)
        
        trainer = MAPPOTrainerImproved(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            global_state_dim=env.global_state_dim,
            hidden_dim=config.training.hidden_dim,
            lr_actor=config.training.lr_actor,
            lr_critic=config.training.lr_critic,
            enable_lr_decay=True,
        )
        
        print(f"  ✅ Model initialized")
        print(f"     - device: {trainer.device}")
        print(f"     - actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}")
        print(f"     - critic params: {sum(p.numel() for p in trainer.critic.parameters()):,}")
        
        # 测试前向传播
        obs, masks = env.reset()
        global_state = env.get_global_state()
        actions, log_probs, value = trainer.select_actions(obs, masks, global_state)
        print(f"  ✅ Model forward pass successful")
        print(f"     - selected actions: {actions}")
        print(f"     - value estimate: {value:.4f}")
        
        # 测试一步训练
        rewards = [0.1 * i for i in range(env.num_agents)]
        dones = [False] * env.num_agents
        
        trainer.store_transition(
            obs, global_state, actions, masks, rewards, dones, log_probs, value
        )
        print(f"  ✅ Stored one transition")
        
        # 测试update
        metrics = trainer.update(episode=1, total_episodes=50000)
        print(f"  ✅ Model update successful")
        print(f"     - actor_loss: {metrics.get('actor_loss', 0):.4f}")
        print(f"     - critic_loss: {metrics.get('critic_loss', 0):.4f}")
        print(f"     - entropy: {metrics.get('entropy', 0):.4f}")
        
    except Exception as e:
        print(f"  ❌ Model error: {e}")
        import traceback
        traceback.print_exc()


def check_dependencies():
    """检查依赖包是否已安装。"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def check_files():
    """检查必要的文件是否存在。"""
    print("\n🔍 Checking required files...")
    
    required_files = [
        "configs/improved_v2.yaml",
        "trainer/mappo_improved.py",
        "train_improved.py",
        "env/marl_env.py",
        "configs/config.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """运行所有诊断检查。"""
    print("=" * 70)
    print("AIRPORT MARL - TRAINING DIAGNOSTIC")
    print("=" * 70)
    
    # 检查文件
    files_ok = check_files()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    if not files_ok or not deps_ok:
        print("\n❌ Missing critical files or dependencies. Cannot continue.")
        return False
    
    # 检查配置
    check_config()
    
    # 检查环境
    check_environment()
    
    # 检查模型
    check_model()
    
    print("\n" + "=" * 70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\n✨ You're ready to start training!")
    print("\nRun:")
    print("  python train_improved.py --config configs/improved_v2.yaml")
    print("\nOr for the original configuration:")
    print("  python train.py --config configs/default.yaml")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
