import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apronops.layout import Layout, Node  # noqa: E402
from apronops.models import Vehicle, VehicleType  # noqa: E402


@pytest.fixture
def simple_layout() -> Layout:
    layout = Layout()
    layout.nodes = {
        "A": Node(id="A", label="A", type="stand", x=0, y=0),
        "B": Node(id="B", label="B", type="stand", x=100, y=0),
        "C": Node(id="C", label="C", type="cargo", x=200, y=0),
    }
    layout.adjacency = {
        "A": [("B", 100.0, 50.0)],
        "B": [("A", 100.0, 50.0), ("C", 100.0, 50.0)],
        "C": [("B", 100.0, 50.0)],
    }
    return layout


@pytest.fixture
def make_vehicle():
    def _make(id_: str, type_: VehicleType, node_id: str, now: datetime) -> Vehicle:
        return Vehicle(
            id=id_,
            type=type_,
            node_id=node_id,
            x=0.0,
            y=0.0,
            speed_mps=4.0,
            last_heartbeat=now,
        )

    return _make
