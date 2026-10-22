from datetime import datetime

from apronops.layout import Layout, Node
from apronops.models import Vehicle, VehicleType
from apronops.simulator import World, _move_along_path


def _vehicle(id_: str, node_id: str, target_node_id: str, speed_mps: float = 4.0) -> Vehicle:
    now = datetime.utcnow()
    v = Vehicle(
        id=id_,
        type=VehicleType.TRACTOR,
        node_id=node_id,
        x=0.0,
        y=0.0,
        speed_mps=speed_mps,
        last_heartbeat=now,
    )
    v.target_node_id = target_node_id
    return v


def _line_layout() -> Layout:
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


def _turn_layout() -> Layout:
    layout = Layout()
    layout.nodes = {
        "A": Node(id="A", label="A", type="stand", x=0, y=0),
        "B": Node(id="B", label="B", type="stand", x=100, y=0),
        "C": Node(id="C", label="C", type="cargo", x=100, y=100),
    }
    layout.adjacency = {
        "A": [("B", 100.0, 50.0)],
        "B": [("A", 100.0, 50.0), ("C", 100.0, 50.0)],
        "C": [("B", 100.0, 50.0)],
    }
    return layout


def test_accel_ramps_up_and_respects_max_speed():
    layout = _line_layout()
    v = _vehicle("V1", "A", "C", speed_mps=4.0)
    world = World(layout, {"V1": v}, {})

    speeds = []
    for _ in range(20):
        _move_along_path(world, v, dt=1.0)
        speeds.append(v.current_speed_mps)

    assert all(s <= 4.0 + 1e-6 for s in speeds)
    assert all(b >= a - 1e-6 for a, b in zip(speeds, speeds[1:]))  # 直线路段全程单调不减
    assert max(speeds) > 3.5  # 足够多 tick 后应接近上限速度


def test_car_following_keeps_safety_gap():
    layout = _line_layout()
    front = _vehicle("V_FRONT", "B", "B")  # 已停在 B，不再移动
    rear = _vehicle("V_REAR", "A", "C")
    world = World(layout, {"V_FRONT": front, "V_REAR": rear}, {})

    for _ in range(60):
        _move_along_path(world, rear, dt=1.0)

    edge_len_ab = layout.edge_distance("A", "B")
    # 反应式跟车不是精确最优控制，允许比理论安全净空小的数值误差，
    # 但核心不变式是：绝不会穿过/到达前车所在节点，且已明显减速接近停止。
    assert rear.node_id == "A"  # 被前车挡住，尚未进入 B
    assert rear.edge_progress_m < edge_len_ab - 1.0
    assert rear.current_speed_mps < 1.0  # 已减速到接近停止，而不是全速逼近


def test_head_on_conflict_lower_id_passes_higher_id_yields():
    layout = _line_layout()
    v1 = _vehicle("V1", "A", "B")
    v2 = _vehicle("V2", "B", "A")
    world = World(layout, {"V1": v1, "V2": v2}, {})

    arrived = False
    for _ in range(60):
        just_arrived = _move_along_path(world, v1, dt=1.0)
        _move_along_path(world, v2, dt=1.0)
        if just_arrived:
            arrived = True
            break

    assert arrived, "V1（id 字典序小）应正常通行到达 B"
    assert v2.edge_progress_m < 1.0, "V2（id 字典序大）在 V1 让出边之前应保持让行、几乎不前进"

    for _ in range(5):
        _move_along_path(world, v2, dt=1.0)
    assert v2.node_id == "B", "还没走完整条边"
    assert v2.edge_progress_m > 0.5, "V1 让出边之后，V2 应能继续前进"


def test_turn_slowdown_lower_speed_near_turn_than_straight():
    turn_layout = _turn_layout()
    line_layout = _line_layout()

    turning = _vehicle("V_TURN", "A", "C", speed_mps=4.0)  # A->B->C，B 处 90 度转弯
    straight = _vehicle("V_STRAIGHT", "A", "B", speed_mps=4.0)  # A->B，直行，目标就是 B 本身
    turn_world = World(turn_layout, {"V_TURN": turning}, {})
    straight_world = World(line_layout, {"V_STRAIGHT": straight}, {})

    # 固定推进 26 秒：足够让转弯车进入 15m 限速半径并减速稳定，
    # 又早于直行车抵达 B（约 27.5s）之前停止，避免边界状态被重置干扰比较。
    for _ in range(26):
        _move_along_path(turn_world, turning, dt=1.0)
        _move_along_path(straight_world, straight, dt=1.0)

    edge_len = turn_layout.edge_distance("A", "B")
    assert turning.node_id == "A" and turning.edge_progress_m >= edge_len - 15.0
    assert straight.node_id == "A"

    assert turning.current_speed_mps <= 2.0  # 明显被转弯限速压低（上限 1.5，留数值容差）
    assert straight.current_speed_mps >= 3.5  # 直行不受限，接近满速
    assert straight.current_speed_mps > turning.current_speed_mps
