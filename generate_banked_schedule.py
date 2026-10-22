"""Build a reproducible, gate-feasible airport apron experiment schedule.

The generator uses the airline, stand, and aircraft-type distributions in an
existing timetable, but it does not select flights using any policy result.
It produces a banked peak-period schedule with minute-level timestamps and a
manifest containing one predetermined eight-aircraft visualization case.

The output workbook uses the same columns as ``simulate_continuous.py`` and
can therefore be passed directly to the rolling paper executor.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_loader import STAND_TO_NODE_MAP


COLUMNS = [
    "airline",
    "movement",
    "planned_arrival",
    "actual_arrival",
    "planned_departure",
    "actual_departure",
    "aircraft_type",
    "stand",
    "flight_number",
]

OUTPUT_COLUMNS = {
    "airline": "航空公司",
    "movement": "进出",
    "planned_arrival": "计划到达",
    "actual_arrival": "实际到达",
    "planned_departure": "计划起飞",
    "actual_departure": "实际起飞",
    "aircraft_type": "机型",
    "stand": "机位",
    "flight_number": "航班号",
}


def _column(df: pd.DataFrame, name: str) -> str | None:
    aliases = {
        "airline": ["航空公司"],
        "stand": ["机位", "机位(A)", "机位(D)"],
        "aircraft_type": ["机型"],
    }
    for candidate in aliases[name]:
        if candidate in df.columns:
            return candidate
    return None


def _weighted_choices(values: list[str], fallback: list[str]) -> tuple[list[str], list[float]]:
    counts = Counter(value for value in values if value and value.lower() != "nan")
    if not counts:
        counts = Counter(fallback)
    labels = sorted(counts)
    weights = [counts[label] for label in labels]
    return labels, weights


def _parse_clock(value: str) -> datetime:
    return datetime.strptime(value, "%H:%M")


def _format_clock(value: datetime | None) -> str:
    return value.strftime("%H:%M") if value is not None else ""


def _eligible_weighted_choice(
    rng: random.Random,
    candidates: list[str],
    weights_by_stand: Counter,
) -> str:
    weights = [weights_by_stand.get(stand, 1) for stand in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def build_schedule(
    source: Path,
    output: Path,
    manifest: Path,
    start_clock: str,
    duration_minutes: int,
    movements: int,
    bank_interval_minutes: int,
    min_turnaround_minutes: int,
    max_turnaround_minutes: int,
    visualization_compression: float,
    seed: int,
) -> dict:
    if movements < 8:
        raise ValueError("movements must be at least 8")
    if duration_minutes < bank_interval_minutes:
        raise ValueError("duration must contain at least one traffic bank")
    if min_turnaround_minutes > max_turnaround_minutes:
        raise ValueError("min turnaround cannot exceed max turnaround")
    if visualization_compression <= 0:
        raise ValueError("visualization compression must be positive")

    source_df = pd.read_excel(source)
    airline_col = _column(source_df, "airline")
    stand_col = _column(source_df, "stand")
    type_col = _column(source_df, "aircraft_type")

    valid_stands = set(STAND_TO_NODE_MAP)
    source_stands = []
    if stand_col is not None:
        source_stands = [str(value).strip() for value in source_df[stand_col]]
    stand_weights = Counter(stand for stand in source_stands if stand in valid_stands)
    if not stand_weights:
        stand_weights = Counter({stand: 1 for stand in sorted(valid_stands)})

    airlines, airline_weights = _weighted_choices(
        [] if airline_col is None else [str(value).strip() for value in source_df[airline_col]],
        ["CA", "MU", "CZ", "HU", "ZH"],
    )
    aircraft_types, type_weights = _weighted_choices(
        [] if type_col is None else [str(value).strip() for value in source_df[type_col]],
        ["319", "320", "32N", "738"],
    )

    rng = random.Random(seed)
    start_time = _parse_clock(start_clock)
    end_time = start_time + timedelta(minutes=duration_minutes)
    stand_state = {
        stand: {"occupied": False, "ready_at": start_time}
        for stand in sorted(valid_stands)
    }

    # At the beginning of a peak window, some stands are naturally occupied
    # by aircraft that arrived before the observation period.
    initial_occupied = rng.sample(sorted(valid_stands), k=min(16, len(valid_stands)))
    for rank, stand in enumerate(initial_occupied):
        stand_state[stand] = {
            "occupied": True,
            "ready_at": start_time + timedelta(minutes=2 * (rank % 8)),
        }

    num_banks = max(1, (duration_minutes + bank_interval_minutes - 1) // bank_interval_minutes)
    per_bank = [movements // num_banks] * num_banks
    for index in range(movements % num_banks):
        per_bank[index] += 1

    records: list[dict] = []
    flight_index = 1
    for bank_index, count in enumerate(per_bank):
        bank_start = start_time + timedelta(minutes=bank_index * bank_interval_minutes)
        for within_bank in range(count):
            # Spread movements through the bank at minute resolution.  The
            # alternating request order creates mixed arrival/departure waves.
            offset = round((within_bank + 0.5) * bank_interval_minutes / count)
            event_time = min(bank_start + timedelta(minutes=offset), end_time - timedelta(minutes=1))
            request_departure = (within_bank + bank_index) % 2 == 0

            ready_departures = [
                stand for stand, state in stand_state.items()
                if state["occupied"] and state["ready_at"] <= event_time
            ]
            free_arrivals = [
                stand for stand, state in stand_state.items()
                if not state["occupied"]
            ]

            if request_departure and ready_departures:
                movement = "DEP"
                stand = _eligible_weighted_choice(rng, ready_departures, stand_weights)
                stand_state[stand] = {"occupied": False, "ready_at": event_time}
            elif free_arrivals:
                movement = "ARR"
                stand = _eligible_weighted_choice(rng, free_arrivals, stand_weights)
                turnaround = rng.randint(min_turnaround_minutes, max_turnaround_minutes)
                # The executor injects arrivals into the local graph five
                # minutes after the timetable arrival and releases departures
                # ten minutes before the timetable departure.  Keep the
                # requested gate turnaround valid under that execution clock.
                execution_offset = 15
                stand_state[stand] = {
                    "occupied": True,
                    "ready_at": event_time + timedelta(
                        minutes=turnaround + execution_offset
                    ),
                }
            elif ready_departures:
                movement = "DEP"
                stand = _eligible_weighted_choice(rng, ready_departures, stand_weights)
                stand_state[stand] = {"occupied": False, "ready_at": event_time}
            else:
                raise RuntimeError("No feasible stand assignment was available")

            airline = rng.choices(airlines, weights=airline_weights, k=1)[0]
            aircraft_type = rng.choices(aircraft_types, weights=type_weights, k=1)[0]
            stand_node = STAND_TO_NODE_MAP[stand]
            if movement == "ARR":
                start_node = 100 if (flight_index + bank_index) % 2 == 0 else 200
                goal_node = stand_node
                row = {
                    "airline": airline,
                    "movement": "进港",
                    "planned_arrival": _format_clock(event_time),
                    "actual_arrival": _format_clock(event_time),
                    "planned_departure": "",
                    "actual_departure": "",
                }
            else:
                start_node = stand_node
                goal_node = 105 if stand_node < 15 else 205
                row = {
                    "airline": airline,
                    "movement": "出港",
                    "planned_arrival": "",
                    "actual_arrival": "",
                    "planned_departure": _format_clock(event_time),
                    "actual_departure": _format_clock(event_time),
                }

            row.update({
                "aircraft_type": aircraft_type,
                "stand": stand,
                "flight_number": f"{airline}{9000 + flight_index:04d}",
            })
            row["release_time"] = event_time
            row["execution_release_time"] = event_time + timedelta(
                minutes=5 if movement == "ARR" else -10
            )
            row["start_node"] = start_node
            row["goal_node"] = goal_node
            row["movement_code"] = movement
            records.append(row)
            flight_index += 1

    records.sort(key=lambda item: (item["release_time"], item["flight_number"]))
    execution_order = sorted(
        records,
        key=lambda item: (item["execution_release_time"], item["flight_number"]),
    )
    visual_records = execution_order[:8]
    visual_base = visual_records[0]["execution_release_time"]
    time_scale_seconds = 1.127
    visualization_case = {
        "selection_rule": (
            "first eight movements after applying the executor's local-network "
            "release offsets; no policy-outcome selection"
        ),
        "schedule_compression": visualization_compression,
        "starts": [item["start_node"] for item in visual_records],
        "goals": [item["goal_node"] for item in visual_records],
        "delays": [
            int(round(
                (item["execution_release_time"] - visual_base).total_seconds()
                / time_scale_seconds * visualization_compression
            ))
            for item in visual_records
        ],
        "flights": [
            {
                "flight_number": item["flight_number"],
                "movement": item["movement_code"],
                "stand": item["stand"],
                "release_time": _format_clock(item["release_time"]),
                "execution_release_time": _format_clock(item["execution_release_time"]),
                "start": item["start_node"],
                "goal": item["goal_node"],
            }
            for item in visual_records
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame([
        {OUTPUT_COLUMNS[key]: item[key] for key in COLUMNS}
        for item in records
    ])
    output_df.to_excel(output, index=False)

    manifest_data = {
        "generator": "generate_banked_schedule.py",
        "source_timetable": str(source),
        "seed": seed,
        "start_clock": start_clock,
        "duration_minutes": duration_minutes,
        "movement_count": len(records),
        "bank_interval_minutes": bank_interval_minutes,
        "turnaround_minutes": [min_turnaround_minutes, max_turnaround_minutes],
        "design_note": (
            "The schedule preserves observed airline/stand/type distributions and "
            "enforces one aircraft per stand. It is a reproducible coordinated "
            "peak-demand scenario, not a schedule selected by MAPPO performance. "
            "A 15-minute release buffer preserves the stated gate turnaround after "
            "the executor's +5-minute arrival and -10-minute departure offsets."
        ),
        "visualization_case": visualization_case,
        "records": [
            {
                "flight_number": item["flight_number"],
                "movement": item["movement_code"],
                "stand": item["stand"],
                "release_time": _format_clock(item["release_time"]),
                "execution_release_time": _format_clock(item["execution_release_time"]),
                "start": item["start_node"],
                "goal": item["goal_node"],
            }
            for item in records
        ],
    }
    manifest.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible banked apron schedule")
    parser.add_argument("--source", default="data.xlsx", help="Reference timetable workbook")
    parser.add_argument("--output", default="data/paper_schedule_banked_v1.xlsx")
    parser.add_argument("--manifest", default="data/paper_schedule_banked_v1_manifest.json")
    parser.add_argument("--start", default="08:00", help="Peak window start in HH:MM")
    parser.add_argument("--duration-minutes", type=int, default=240)
    parser.add_argument("--movements", type=int, default=96)
    parser.add_argument("--bank-interval-minutes", type=int, default=10)
    parser.add_argument("--min-turnaround-minutes", type=int, default=45)
    parser.add_argument("--max-turnaround-minutes", type=int, default=75)
    parser.add_argument(
        "--visualization-compression",
        type=float,
        default=0.10,
        help="Common time-compression factor for the controlled stress scenario",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_schedule(
        source=Path(args.source),
        output=Path(args.output),
        manifest=Path(args.manifest),
        start_clock=args.start,
        duration_minutes=args.duration_minutes,
        movements=args.movements,
        bank_interval_minutes=args.bank_interval_minutes,
        min_turnaround_minutes=args.min_turnaround_minutes,
        max_turnaround_minutes=args.max_turnaround_minutes,
        visualization_compression=args.visualization_compression,
        seed=args.seed,
    )
    print(f"Schedule written: {args.output}")
    print(f"Manifest written: {args.manifest}")
    print(
        "Visualization case: "
        f"{len(manifest['visualization_case']['starts'])} aircraft, "
        f"delays={manifest['visualization_case']['delays']}"
    )


if __name__ == "__main__":
    main()
