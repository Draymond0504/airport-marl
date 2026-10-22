# Airport MARL Simulation - Core Context & Architecture

This document summarizes the core architecture, code structure, and overarching goals of the Airport MAPPO (Multi-Agent Proximal Policy Optimization) project. It is designed to help Large Language Models (LLMs) quickly understand the context and immediately assist with development.

## 1. Project Objectives & Current Needs
The ultimate goal of this project is to produce a publication-grade multi-agent reinforcement learning (MARL) system for airport apron taxiway scheduling.
- **Goal 1: Paper & Framework Implementation**: Rigorously implement the mathematical models and MARL framework suitable for an academic paper.
- **Goal 2: Ablation Studies**: Prove the superiority of the "Planning-guided MARL" approach by running robust ablations comparing **Pure MARL (No Guide)** vs **A* + MARL**.
- **Goal 3: Scenario Testing & Visualization**: Demonstrate that the MAPPO model outperforms traditional **A* + FCFS (First-Come-First-Serve)** baselines, particularly in dynamic/perturbed scenarios (e.g., taxiway closures, priority changes), and generate high-fidelity publication-ready visualizations (heatmaps, convergence curves, trajectory animations).

---

## 2. Core Code Structure

### 2.1 The Physical Environment (`env/taxi_graph_env.py`)
- **Role**: Simulates the physical physics and graph topology of the airport apron (Layout F).
- **Key Mechanics**:
  - **`auto_speed=True`**: Agents automatically accelerate/decelerate along edges based on physical constraints (inertia, turn speed limits). The RL agent only needs to make discrete routing choices at intersections.
  - **Collision Detection**: Detects node collisions, head-on edge collisions, and rear-end collisions. Evaluated precisely using physical Euclidean distance (`L_SCALE`).
  - **Corridor Manager**: Prevents agents from entering narrow bi-directional corridors if an opposing agent has already locked it.

### 2.2 The MARL Wrapper (`env/marl_env.py`)
- **Role**: Acts as the bridge between the physical environment and the MAPPO algorithm, handling Reward Shaping and Episode Management.
- **Reward Shaping Highlights**:
  - **`completion` (+50)** & **`team_bonus` (+30)**: Strongly encourages reaching the goal.
  - **`conflict` (-100)**: Severe penalty for crashing.
  - **`step_penalty` (-0.01)**: Mild penalty to encourage efficiency.
  - **`deadlock` (-0.5/step)**: Punishes agents that get stuck in gridlocks for extended periods.

### 2.3 The Agent & Observation (`agent/aircraft_agent.py`)
- **Role**: Defines what the agent "sees" and what actions it can take.
- **Action Space**: `Discrete(8)`. Actions 0-4 select neighbor nodes (routing). Actions 5-6 are manual speed controls (masked out when `auto_speed=True`). Action 7 is cruise.
- **Observation Space (41-Dim Egocentric)**:
  - **Self State (6 dims)**: Speed, on_edge status, distance to goal, time remaining, priority, etc.
  - **Neighbor Slots (5 slots × 7 dims = 35 dims)**: For each neighbor node, the agent observes:
    1. `is_valid`
    2. `path_cost_norm` (A* shortest path distance)
    3. `on_planned_path` (Soft-Guidance: 1 if this node is recommended by A*)
    4. `immediate_risk` (Is there a physical collision risk here?)
    5. `corridor_locked` (Is the corridor ahead blocked?)
    6. **`downstream_occ`**: (New) Looks 2-hops ahead to see if the next node is occupied, preventing blind entry into deadlocks.
    7. **`corridor_load`**: (New) Global congestion metric for the target corridor.

### 2.4 The Trainer (`trainer/mappo_improved.py`)
- **Role**: Custom PPO implementation tailored for MARL.
- **Curriculum Learning Ready**: The `load()` function is designed to handle changing agent counts. When loading weights from a 4-agent model to train an 8-agent model, it successfully loads the Actor (egocentric) while safely re-initializing the Critic (which relies on global state dimension `num_agents * 8 + 2`).

---

## 3. The "Soft-Guidance" MAPPO Paradigm
This project does NOT use a pure "blank slate" MARL approach (which fails in complex airport topologies). Instead, it uses **Soft-Guidance**:
1. **A* Runs in Background**: Computes the shortest path and flags the best next node via the `on_planned_path` observation bit.
2. **MAPPO Makes the Call**: The neural network looks at the A* recommendation, but also looks at congestion (`corridor_load`) and nearby agents (`immediate_risk`, `downstream_occ`).
3. **Dynamic Routing**: If the A* path is congested or will lead to a deadlock, the MAPPO agent learns to **deviate from the A* path**, wait, or yield. This is what makes it superior to rigid A*+FCFS systems.

## 4. Key Execution Scripts
- **`configs/improved_v2.yaml` / `configs/ablation_no_guide.yaml`**: The master configuration files controlling agent counts, reward weights, and dynamic disturbance probabilities.
- **`run_ablation_quick.py`**: A one-click script that runs both the Guided MAPPO and Pure MAPPO ablation groups and plots the convergence comparison. Supports `--resume` for loading Curriculum weights.
- **`upgrade_weights.py`**: A utility that pads older 31-dim pre-trained `.pt` weights with zeros to make them compatible with the newly upgraded 41-dim observation space without losing learned policies.
