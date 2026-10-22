"""API 请求 / 响应 schema。"""
from __future__ import annotations

from pydantic import BaseModel


class CreateTaskRequest(BaseModel):
    id: str
    type: str
    origin: str
    destination: str
    priority: int = 3
    deadline_offset_s: int = 1800


class ResolveAlertRequest(BaseModel):
    resolution: str
