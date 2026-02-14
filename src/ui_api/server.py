"""FastAPI + WebSocket server for live AGV simulation visualization."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.simulation.engine import SimulationEngine
from src.warehouse.config import SimulationConfig, WarehouseConfig, load_config

app = FastAPI(title="AGV Fleet Simulator API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    """Basic readiness endpoint."""

    return {"status": "ok"}


def _load_runtime_config(config_path: str, hours: float | None, seed: int | None) -> WarehouseConfig:
    path = Path(config_path)
    if path.exists():
        base = load_config(path)
    else:
        base = WarehouseConfig()

    if hours is None and seed is None:
        return base

    sim = SimulationConfig(
        duration_hours=hours if hours is not None else base.simulation.duration_hours,
        dispatch_interval_s=base.simulation.dispatch_interval_s,
        metrics_interval_s=base.simulation.metrics_interval_s,
        random_seed=seed if seed is not None else base.simulation.random_seed,
    )
    return WarehouseConfig(
        warehouse=base.warehouse,
        stations=base.stations,
        agv=base.agv,
        orders=base.orders,
        pick_station=base.pick_station,
        simulation=sim,
    )


def _metrics_summary(engine: SimulationEngine) -> dict:
    metrics = engine.metrics
    if metrics is None:
        return {}
    return {
        "total_orders_generated": metrics.total_orders_generated,
        "total_orders_completed": metrics.total_orders_completed,
        "total_simulation_time_s": metrics.total_simulation_time_s,
        "avg_throughput_per_hour": metrics.avg_throughput_per_hour,
        "avg_cycle_time_s": metrics.avg_cycle_time_s,
        "p95_cycle_time_s": metrics.p95_cycle_time_s,
        "avg_agv_utilization_pct": metrics.avg_agv_utilization_pct,
        "avg_station_utilization_pct": metrics.avg_station_utilization_pct,
        "total_distance_traveled_m": metrics.total_distance_traveled_m,
    }


@app.websocket("/api/simulation/ws")
async def simulation_ws(
    websocket: WebSocket,
    n_agvs: int = 50,
    hours: float = 0.25,
    seed: int = 42,
    strategy: Literal["cpsat", "greedy"] = "cpsat",
    hz: float = 10.0,
    config_path: str = "config/default_warehouse.yaml",
) -> None:
    """Run one simulation per websocket connection and stream live snapshots."""

    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=512)
    stop_event = threading.Event()

    def emit(event_type: str, payload: dict) -> None:
        def _enqueue() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait({"type": event_type, "payload": payload})

        loop.call_soon_threadsafe(_enqueue)

    def _run() -> None:
        try:
            cfg = _load_runtime_config(config_path=config_path, hours=hours, seed=seed)
            engine = SimulationEngine(cfg, n_agvs=n_agvs, assignment_strategy=strategy)

            snapshot_interval = 1.0 / max(1.0, hz)
            engine.run(
                runtime_interval_s=snapshot_interval,
                runtime_callback=lambda snap: emit("snapshot", snap),
                layout_callback=lambda layout: emit("layout", layout),
                stop_requested=stop_event.is_set,
            )
            emit("completed", {"metrics": _metrics_summary(engine), "stopped": stop_event.is_set()})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            emit("error", {"message": str(exc)})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    emit(
        "started",
        {
            "n_agvs": n_agvs,
            "hours": hours,
            "seed": seed,
            "strategy": strategy,
            "hz": hz,
            "config_path": config_path,
        },
    )

    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["type"] in {"completed", "error"}:
                break
    except WebSocketDisconnect:
        stop_event.set()
    finally:
        stop_event.set()
        thread.join(timeout=3.0)
