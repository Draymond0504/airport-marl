# TRB Result Visualization

Use the unified plotting pipeline after each baseline or perturbation run:

```powershell
D:\Anaconda\Anaconda_envs\envs\torch_gpu\python.exe plot_trb_results.py `
  --baseline-dir results/exp_4_2 `
  --delay-dir results/exp_4_3_delay `
  --closure-dir results/exp_4_3_closure `
  --priority-dir results/exp_4_3_priority `
  --out-dir results/trb_figures
```

Outputs:

- `fig_scenario_outcomes.png`: safe completion, eventual service,
  deadlocks per 100 flights, and average total taxi time.
- `fig_perturbation_timeseries.png`: cumulative safe completion normalized by
  the actual flight count in each scenario.
- `fig_delay_wait_distribution.png`: delay-scene waiting-time distribution
  without deleting large observations.
- `fig_priority_event_response.png`: event-aligned response of priority and
  normal flights when priority micro-logs are available.
- `scenario_metrics.csv` and `scenario_metrics.md`: table-ready values.

The plotting contract uses `completion_rate_pct` as the strict safe-completion
metric and `eventual_completion_rate_pct` as the eventual service metric. They
must not be merged into one bar or one sentence in the paper.

The experiment runner also writes `experiment_meta.json` and one event file per
algorithm (`MAPPO_events.json`, `FCFS_events.json`). These files allow future
plots to mark the actual delay, closure, or priority event time rather than
assuming a fixed step count.

## Final Paper Suite

Run the complete baseline and perturbation suite with one frozen checkpoint:

```powershell
D:\Anaconda\Anaconda_envs\envs\torch_gpu\python.exe run_paper_final.py `
  --model checkpoints/stage3_from_4agents_v1/stage3_8agents_best_eval.pt `
  --result-root results/paper_final
```

The wrapper writes `baseline`, `delay`, `closure`, and `priority` folders,
validates the result contract, and calls `plot_trb_results.py`. In addition to
the aggregate plots, it creates `fig_delay_mechanism.png` and
`fig_closure_mechanism.png` for event-level explanations.

The deterministic training curve is generated separately:

```powershell
D:\Anaconda\Anaconda_envs\envs\torch_gpu\python.exe plot_training_evaluation.py `
  --files checkpoints/stage3_from_4agents_v1/stage3_8agents_evaluation_log.csv `
  --labels MAPPO-Guided
```

Run the NoGuide ablation after the guided baseline is available:

```powershell
D:\Anaconda\Anaconda_envs\envs\torch_gpu\python.exe run_paper_ablation.py `
  --guided-result results/paper_final/baseline `
  --save-dir checkpoints/paper_ablation_no_guide
```

Then generate `fig_ablation_astar_guidance.png` with:

```powershell
D:\Anaconda\Anaconda_envs\envs\torch_gpu\python.exe plot_ablation_result.py
```
