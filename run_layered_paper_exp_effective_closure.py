"""Layered rolling runner with closures inside the compressed service horizon."""

from paper_rolling_executor import RollingScenario

_ORIGINAL_INIT = RollingScenario.__init__

def _effective_init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)
    if scenario != "closure":
        return
    service_horizon = max(self.release_steps, default=max_steps)
    self.closure_plan = [
        (max(1, int(service_horizon * 0.35)), 800, (102, 103)),
        (max(2, int(service_horizon * 0.70)), 800, (304, 305)),
    ]

RollingScenario.__init__ = _effective_init

import run_layered_paper_exp as base

base.main()
