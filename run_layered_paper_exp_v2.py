"""Use the corrected layered rolling executor entry point."""

import run_layered_paper_exp as base
from layered_rolling_executor_v2 import run_layered_fcfs, run_layered_mappo

base.run_layered_fcfs = run_layered_fcfs
base.run_layered_mappo = run_layered_mappo
base.main()
