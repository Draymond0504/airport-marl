"""Layered rolling runner with a visible mid-service delay interruption."""

from paper_rolling_executor import RollingScenario

_ORIGINAL_INIT = RollingScenario.__init__

def _midservice_delay(self):
    service_horizon = max(self.release_steps, default=self.max_steps)
    duration = max(1, int(round(20.0 * 60.0 / self.time_scale)))
    midpoint = max(1, int(round(service_horizon * 0.50)))
    start = max(1, midpoint - duration // 2)
    end = min(service_horizon, start + duration)
    affected = [i for i, release in enumerate(self.release_steps) if start <= release < end]
    for i in affected:
        self.release_steps[i] = end
    self.admission_hold_start = start
    self.admission_hold_end = end
    self.metrics.events.append({
        "step": start,
        "type": "delay_interruption",
        "duration": end - start,
        "affected_flights": affected,
        "count": len(affected),
        "resume_step": end,
        "description": "mid-service admission interruption followed by queued-release recovery",
    })

def _init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)
    if scenario == "delay":
        _midservice_delay(self)

_ORIGINAL_ADMIT = None
RollingScenario.__init__ = _init
# Disable the default weather-delay rewrite; this runner uses one explicit
# mid-service admission interruption so the event is unambiguous in plots.
RollingScenario._apply_weather_delay = lambda self: None

import run_layered_paper_exp as base
from paper_rolling_executor import RollingPaperExecutor

_ORIGINAL_ADMIT = RollingPaperExecutor._admit_due_flights
def _admit_with_hold(self, step):
    scenario = self.scenario
    if scenario.scenario == "delay":
        start = getattr(scenario, "admission_hold_start", -1)
        end = getattr(scenario, "admission_hold_end", -1)
        if start <= step < end:
            return
    return _ORIGINAL_ADMIT(self, step)

RollingPaperExecutor._admit_due_flights = _admit_with_hold
base.main()
