import os
import subprocess

def run_multiple_seeds():
    seeds = [42, 1024, 2048]
    
    experiments = [
        {
            "name": "guided",
            "config": "configs/improved_v2.yaml"
        },
        {
            "name": "no_guide",
            "config": "configs/ablation_no_guide.yaml"
        }
    ]
    
    for exp in experiments:
        for seed in seeds:
            save_dir = f"checkpoints/{exp['name']}/seed_{seed}"
            print("="*80)
            print(f"Starting Training: Group [{exp['name']}], Seed [{seed}]")
            print(f"Config: {exp['config']}, Output Dir: {save_dir}")
            print("="*80)
            
            import sys
            cmd = [
                sys.executable, "train_improved.py",
                "--config", exp['config'],
                "--seed", str(seed),
                "--save-dir", save_dir
            ]
            
            # Use subprocess to run and wait for it to finish
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_multiple_seeds()
