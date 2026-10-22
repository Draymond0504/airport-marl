"""核心数据模型：Vehicle / Task / Event / Alert。"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VehicleType(str, Enum):
    TRACTOR = "tractor"
    SHUTTLE = "shuttle"
    CATERING = "catering"
    AMR_INSPECT = "amr_inspect"


class VehicleState(str, Enum):
    IDLE = "IDLE"
    ASSIGNED = "ASSIGNED"
    EN_ROUTE = "EN_ROUTE"
    OPERATING = "OPERATING"
    RETURNING = "RETURNING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    REMOTE_ASSIST = "REMOTE_ASSIST"
    SAFE_STOP = "SAFE_STOP"


class TaskType(str, Enum):
    BAGGAGE_TOW = "baggage_tow"
    CATERING_SUPPLY = "catering_supply"
    INSPECTION = "inspection"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REASSIGNED = "REASSIGNED"


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Vehicle(BaseModel):
    id: str
    type: VehicleType
    node_id: str
    x: float
    y: float
    speed_mps: float
    current_speed_mps: float = 0.0
    planned_path: list[str] = Field(default_factory=list)
    state: VehicleState = VehicleState.IDLE
    battery: float = 100.0
    current_task_id: Optional[str] = None
    target_node_id: Optional[str] = None
    next_node_id: Optional[str] = None
    edge_progress_m: float = 0.0
    task_phase: Optional[str] = None  # to_origin / to_destination / servicing / returning
    service_remaining_s: float = 0.0
    last_heartbeat: datetime
    position_quality_streak_low: int = 0


class Task(BaseModel):
    id: str
    type: TaskType
    origin: str
    destination: str
    priority: int = 3
    deadline: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_vehicle_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Event(BaseModel):
    id: str
    timestamp: datetime
    source: str
    event_type: str
    severity: EventSeverity
    related_task_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)


class Alert(BaseModel):
    id: str
    event_id: str
    anomaly_type: str
    impact_scope: list[str] = Field(default_factory=list)
    suggested_action: str
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
