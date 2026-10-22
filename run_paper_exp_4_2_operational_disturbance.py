"""Paper runner with operational clock calibration and effective disturbances.

It preserves the existing simulation interface while scheduling disturbances
relative to the compressed rolling-service horizon.
"""

from __future__ import annotations

from paper_rolling_executor import RollingScenario


_ORIGINAL_INIT = RollingScenario.__init__


def _delay_interruption(self) -> None:
    """Hold a scheduled release block, then resume it as a delayed batch."""
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
    self.metrics.events.append({
        "step": interruption_start,
        "type": "delay_interruption",
        "duration": interruption_end - interruption_start,
        "affected_flights": affected,
        "count": len(affected),
        "resume_step": interruption_end,
        "description": "scheduled release interruption followed by batch recovery",
    })


def _operational_init(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression=1.0):
    _ORIGINAL_INIT(self, scenario, flights, max_steps, time_scale, metrics, schedule_compression)
    service_horizon = max(self.release_steps, default=max_steps)
    if scenario == "closure":
        # Preserve a 15 min physical closure under the selected time mapping.
        closure_duration = max(1, int(round(15.0 * 60.0 / time_scale)))
        self.closure_plan = [
            (max(1, int(round(service_horizon * 0.35))), closure_duration, (102, 103)),
            (max(2, int(round(service_horizon * 0.70))), closure_duration, (304, 305)),
        ]


RollingScenario._apply_weather_delay = _delay_interruption
RollingScenario.__init__ = _operational_init


if __name__ == "__main__":
    from run_paper_exp_4_2 import main

    main()
