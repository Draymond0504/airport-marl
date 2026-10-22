"""Unified final rolling protocol for the paper experiments."""

from __future__ import annotations

import paper_rolling_executor as paper
from paper_rolling_executor import RollingPaperExecutor, RollingScenario

# Import applies edge logging, closure duration, and three priority targets.
import run_layered_paper_exp_operational_v2  # noqa: F401


# Replace the legacy weather-delay rewrite with one explicit mid-service
# admission interruption.
RollingScenario._apply_weather_delay = lambda self: None

_ORIGINAL_INIT = RollingScenario.__init__


def _final_scenario_init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)
    if scenario != "delay":
        return

    # The compressed schedule midpoint is not the operational midpoint: FCFS
    # can keep all slots occupied long after the nominal release sequence has
    # passed its midpoint.  Apply the disturbance to the actual admission
    # gate in a common service window instead of rewriting release timestamps.
    max_release = max(self.release_steps, default=0)
    start = max(1, int(round(max_release * 2.8)))
    start = min(start, max_steps - 1201)
    duration = 1200
    end = start + duration
    self.admission_hold_start = start
    self.admission_hold_end = end
    self.delay_affected_flights = set()
    self.delay_event = {
        "step": start,
        "type": "delay_interruption",
        "duration": duration,
        "affected_flights": [],
        "count": 0,
        "resume_step": end,
        "description": "mid-service admission gate interruption followed by queued-release recovery",
    }
    self.metrics.events.append(self.delay_event)


RollingScenario.__init__ = _final_scenario_init

_ORIGINAL_ADMIT = RollingPaperExecutor._admit_due_flights


def _admit_with_delay_hold(self, step):
    if self.scenario.scenario == "delay":
        start = getattr(self.scenario, "admission_hold_start", -1)
        end = getattr(self.scenario, "admission_hold_end", -1)
        if start <= step < end:
            due = sorted(
                i for i, release in enumerate(self.scenario.release_steps)
                if release <= step and self.metrics.records[i].status == "queued"
            )
            for flight_index in due:
                self.metrics.record_admission_hold(flight_index, step, "delay_interruption")
                self.scenario.delay_affected_flights.add(flight_index)
            event = getattr(self.scenario, "delay_event", None)
            if event is not None:
                event["affected_flights"] = sorted(self.scenario.delay_affected_flights)
                event["count"] = len(event["affected_flights"])
            return
    return _ORIGINAL_ADMIT(self, step)


RollingPaperExecutor._admit_due_flights = _admit_with_delay_hold


if __name__ == "__main__":
    import run_layered_paper_exp as base

    base.main()
