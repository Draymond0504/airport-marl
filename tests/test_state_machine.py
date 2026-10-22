import pytest

from apronops.state_machine import (
    InvalidTransition,
    transition_task,
    transition_vehicle,
)
from apronops.models import TaskStatus, VehicleState


def test_vehicle_valid_transition():
    assert transition_vehicle(VehicleState.IDLE, VehicleState.ASSIGNED) == VehicleState.ASSIGNED
    assert transition_vehicle(VehicleState.ASSIGNED, VehicleState.EN_ROUTE) == VehicleState.EN_ROUTE
    assert transition_vehicle(VehicleState.EN_ROUTE, VehicleState.OPERATING) == VehicleState.OPERATING
    assert transition_vehicle(VehicleState.OPERATING, VehicleState.RETURNING) == VehicleState.RETURNING
    assert transition_vehicle(VehicleState.RETURNING, VehicleState.IDLE) == VehicleState.IDLE


def test_vehicle_invalid_transition_raises():
    with pytest.raises(InvalidTransition):
        transition_vehicle(VehicleState.IDLE, VehicleState.OPERATING)


def test_vehicle_degraded_recovery_paths():
    assert transition_vehicle(VehicleState.DEGRADED, VehicleState.SAFE_STOP) == VehicleState.SAFE_STOP
    assert transition_vehicle(VehicleState.SAFE_STOP, VehicleState.IDLE) == VehicleState.IDLE
    with pytest.raises(InvalidTransition):
        transition_vehicle(VehicleState.SAFE_STOP, VehicleState.OPERATING)


def test_task_valid_transition():
    assert transition_task(TaskStatus.PENDING, TaskStatus.ASSIGNED) == TaskStatus.ASSIGNED
    assert transition_task(TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS) == TaskStatus.IN_PROGRESS
    assert transition_task(TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED) == TaskStatus.COMPLETED


def test_task_invalid_transition_raises():
    with pytest.raises(InvalidTransition):
        transition_task(TaskStatus.COMPLETED, TaskStatus.PENDING)


def test_task_reassignment_path():
    assert transition_task(TaskStatus.IN_PROGRESS, TaskStatus.REASSIGNED) == TaskStatus.REASSIGNED
    assert transition_task(TaskStatus.REASSIGNED, TaskStatus.PENDING) == TaskStatus.PENDING
