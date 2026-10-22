"""Run paper scenarios with closure timing tied to the effective schedule.

The original runner defines closures relative to ``max_steps``. In compressed
rolling schedules the service period can finish before the later closure. This
wrapper keeps the runner intact while ensuring both planned closure events fall
within the active release horizon.
"""

from __future__ import annotations

from paper_rolling_executor import RollingScenario


_ORIGINAL_INIT = RollingScenario.__init__


def _effective_closure_init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)
    if scenario != "closure":
        return
    service_horizon = max(self.release_steps, default=max_steps)
    self.closure_plan = [
        (max(1, int(service_horizon * 0.35)), 800, (102, 103)),
        (max(2, int(service_horizon * 0.70)), 800, (304, 305)),
    ]


RollingScenario.__init__ = _effective_closure_init


if __name__ == "__main__":
    from run_paper_exp_4_2 import main

    main()
