from datetime import datetime, timedelta

from apronops.models import Task, TaskType, VehicleType
from apronops.scheduler import assign_nearest, pick_task_order


def test_assign_nearest_picks_closest_compatible_vehicle(simple_layout, make_vehicle):
    now = datetime.utcnow()
    far = make_vehicle("TRACTOR_FAR", VehicleType.TRACTOR, "A", now)
    near = make_vehicle("TRACTOR_NEAR", VehicleType.TRACTOR, "B", now)
    task = Task(
        id="T1", type=TaskType.BAGGAGE_TOW, origin="B", destination="C",
        created_at=now,
    )

    chosen = assign_nearest(task, [far, near], simple_layout)

    assert chosen is not None
    assert chosen.id == "TRACTOR_NEAR"


def test_assign_nearest_returns_none_when_no_compatible_vehicle(simple_layout, make_vehicle):
    now = datetime.utcnow()
    catering = make_vehicle("CATERING_01", VehicleType.CATERING, "A", now)
    task = Task(
        id="T2", type=TaskType.BAGGAGE_TOW, origin="A", destination="C",
        created_at=now,
    )

    assert assign_nearest(task, [catering], simple_layout) is None


def test_pick_task_order_by_priority():
    now = datetime.utcnow()
    low = Task(id="LOW", type=TaskType.INSPECTION, origin="A", destination="B", priority=1, created_at=now)
    high = Task(id="HIGH", type=TaskType.INSPECTION, origin="A", destination="B", priority=5, created_at=now)

    ordered = pick_task_order([low, high])

    assert [t.id for t in ordered] == ["HIGH", "LOW"]


def test_pick_task_order_by_deadline():
    now = datetime.utcnow()
    soon = Task(
        id="SOON", type=TaskType.INSPECTION, origin="A", destination="B",
        priority=1, deadline=now + timedelta(seconds=60), created_at=now,
    )
    later = Task(
        id="LATER", type=TaskType.INSPECTION, origin="A", destination="B",
        priority=5, deadline=now + timedelta(seconds=600), created_at=now,
    )

    ordered = pick_task_order([later, soon], strategy="deadline")

    assert [t.id for t in ordered] == ["SOON", "LATER"]
