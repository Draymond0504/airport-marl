"""Run the paper delay scenario as a mid-service admission interruption.

New admissions are held during the event while aircraft already on the
network continue.  The release queue then resumes at the end of the window.
"""

from __future__ import annotations

from paper_rolling_executor import RollingPaperExecutor, RollingScenario


_ORIGINAL_INIT = RollingScenario.__init__
_ORIGINAL_ADMIT_DUE_FLIGHTS = RollingPaperExecutor._admit_due_flights


def _midservice_delay(self) -> None:
    service_horizon = max(self.release_steps, default=self.max_steps)
    duration = max(1, int(round(20.0 * 60.0 / self.time_scale)))
    midpoint = max(1, int(round(service_horizon * 0.50)))
    start = max(1, midpoint - duration // 2)
    end = min(service_horizon, start + duration)
    affected = [index for index, release in enumerate(self.release_steps) if start <= release < end]
    for index in affected:
        self.release_steps[index] = end
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


def _admit_with_midservice_hold(self, step: int) -> None:
    scenario = self.scenario
    if scenario.scenario == "delay":
        start = getattr(scenario, "admission_hold_start", -1)
        end = getattr(scenario, "admission_hold_end", -1)
        if start <= step < end:
            return
    _ORIGINAL_ADMIT_DUE_FLIGHTS(self, step)


RollingScenario._apply_weather_delay = _midservice_delay
RollingScenario.__init__ = _init
RollingPaperExecutor._admit_due_flights = _admit_with_midservice_hold


if __name__ == "__main__":
    from run_paper_exp_4_2 import main

    main()
