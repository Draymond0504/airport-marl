"""Paper runner with a real gate-admission interruption for delay analysis.

The standard rolling executor can contain a backlog of flights that were
scheduled before a delay.  This wrapper holds new admissions during the
event window, while leaving in-motion aircraft unaffected, so the cumulative
service curve contains an observable interruption and post-event recovery.
"""

from __future__ import annotations

from paper_rolling_executor import RollingPaperExecutor, RollingScenario


_ORIGINAL_INIT = RollingScenario.__init__
_ORIGINAL_ADMIT_DUE_FLIGHTS = RollingPaperExecutor._admit_due_flights


def _delay_interruption(self) -> None:
    service_horizon = max(self.release_steps, default=self.max_steps)
    interruption_steps = max(1, int(round(20.0 * 60.0 / self.time_scale)))
    interruption_start = max(1, int(round(service_horizon * 0.35)))
    interruption_end = min(service_horizon, interruption_start + interruption_steps)
    affected = [
        index for index, release in enumerate(self.release_steps)
        if interruption_start <= release < interruption_end
    ]
    for index in affected:
        self.release_steps[index] = interruption_end
    self.admission_hold_start = interruption_start
    self.admission_hold_end = interruption_end
    self.metrics.events.append({
        "step": interruption_start,
        "type": "delay_interruption",
        "duration": interruption_end - interruption_start,
        "affected_flights": affected,
        "count": len(affected),
        "resume_step": interruption_end,
        "description": "admission interruption followed by queued-release recovery",
    })


def _operational_init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)


def _admit_with_delay_gatehold(self, step: int) -> None:
    scenario = self.scenario
    if (
        scenario.scenario == "delay"
        and getattr(scenario, "admission_hold_start", None) <= step
        and step < getattr(scenario, "admission_hold_end", -1)
    ):
        return
    _ORIGINAL_ADMIT_DUE_FLIGHTS(self, step)


RollingScenario._apply_weather_delay = _delay_interruption
RollingScenario.__init__ = _operational_init
RollingPaperExecutor._admit_due_flights = _admit_with_delay_gatehold


if __name__ == "__main__":
    from run_paper_exp_4_2 import main

    main()
