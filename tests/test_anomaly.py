from datetime import datetime, timedelta

from apronops import anomaly
from apronops.models import VehicleType


def test_check_heartbeat_ok_within_window(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.TRACTOR, "A", now)
    assert anomaly.check_heartbeat(v, now + timedelta(seconds=5)) is None


def test_check_heartbeat_triggers_after_threshold(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.TRACTOR, "A", now)
    later = now + timedelta(seconds=anomaly.HEARTBEAT_PERIOD_S * (anomaly.HEARTBEAT_MISS_THRESHOLD + 1))

    event = anomaly.check_heartbeat(v, later)

    assert event is not None
    assert event.event_type == "communication_lost"
    assert event.severity.value == "CRITICAL"


def test_position_quality_streak_triggers_warning(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.AMR_INSPECT, "A", now)

    event = None
    for _ in range(anomaly.POSITION_QUALITY_STREAK_THRESHOLD):
        event = anomaly.check_position_quality(v, True, now)

    assert event is not None
    assert event.event_type == "position_quality_degraded"


def test_position_quality_resets_on_good_reading(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.AMR_INSPECT, "A", now)

    anomaly.check_position_quality(v, True, now)
    anomaly.check_position_quality(v, False, now)

    assert v.position_quality_streak_low == 0


def test_check_battery_insufficient(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.TRACTOR, "A", now)
    v.battery = 10.0

    event = anomaly.check_battery(v, required_pct=5.0, now=now)

    assert event is not None
    assert event.event_type == "battery_insufficient"


def test_check_battery_sufficient(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.TRACTOR, "A", now)
    v.battery = 100.0

    assert anomaly.check_battery(v, required_pct=5.0, now=now) is None


def test_check_route_block_triggers(make_vehicle):
    now = datetime.utcnow()
    v = make_vehicle("V1", VehicleType.TRACTOR, "A", now)

    event = anomaly.check_route_block(v, eta_s=200, base_travel_s=60, now=now)

    assert event is not None
    assert event.event_type == "route_blocked"
