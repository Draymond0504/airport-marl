import asyncio

from starlette.testclient import TestClient

from apronops.api import main


def test_ws_state_pushes_monotonic_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("APRONOPS_DB_PATH", str(tmp_path / "test.db"))

    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/state") as websocket:
            for _ in range(2):
                with main._lock:
                    main.simulator.tick(main._world, dt=main.SIM_DT_S)
                    payload = main._snapshot_payload(main._world)
                future = asyncio.run_coroutine_threadsafe(
                    main._manager.broadcast(payload), main._loop
                )
                future.result(timeout=5)

            first = websocket.receive_json()
            second = websocket.receive_json()

    assert second["tick"] > first["tick"]
    assert len(first["vehicles"]) == 5
    assert {"id", "state", "x", "y", "current_task_id"} <= first["vehicles"][0].keys()
    assert "unresolved_alerts" in first
