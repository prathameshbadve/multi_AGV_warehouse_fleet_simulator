"""
Cost matrix computation for task assignment.

Precomputes the full cost matrix and battery feasibility mask for all
(AGV, Task) pairs. Distances come from the warehouse graph's precomputed
shortest paths. All values are integer-scaled (×100) for CP-SAT compatibility.

Usage:
    matrix = compute_cost_matrix(tasks, agvs, warehouse, agv_config)
    # matrix.trip_cost[a][t] = integer-scaled travel cost
    # matrix.feasible[a][t]  = True if AGV a can complete task t on battery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.agv import AGV
    from src.simulation.orders import Task
    from src.warehouse.config import AGVConfig
    from src.warehouse.graph import WarehouseGraph


COST_SCALE = 100  # Multiply float distances by this for CP-SAT integer math


@dataclass
class CostMatrix:
    """Precomputed assignment costs and feasibility.

    All distance/cost fields are integer-scaled (×COST_SCALE).
    Indexed as [agv_index][task_index].

    Attributes:
        n_agvs: Number of AGVs.
        n_tasks: Number of tasks.
        agv_ids: AGV ID at each index.
        task_ids: Task ID at each index.
        pickup_dist: Distance from AGV position to pod location.
        deliver_dist: Distance from pod location to pick station.
        return_dist: Distance from pick station to return location.
        charger_dist: Distance from return location to nearest charger.
        trip_cost: pickup_dist + deliver_dist (the assignment-relevant cost).
        feasible: Whether AGV can complete the full round trip on battery.
        station_index: Maps task index → station index (for load balancing).
        station_ids: Unique station IDs in order.
    """

    n_agvs: int
    n_tasks: int
    agv_ids: list[str]
    task_ids: list[str]
    pickup_dist: list[list[int]]
    deliver_dist: list[list[int]]
    return_dist: list[list[int]]
    charger_dist: list[list[int]]
    trip_cost: list[list[int]]
    feasible: list[list[bool]]
    station_index: list[int]
    station_ids: list[str]


def compute_cost_matrix(
    tasks: list[Task],
    agvs: list[AGV],
    warehouse: WarehouseGraph,
    agv_config: AGVConfig,
) -> CostMatrix:
    """Build the full cost matrix for assignment optimization.

    Args:
        tasks: Pending tasks to assign.
        agvs: Available (idle) AGVs.
        warehouse: Warehouse graph with precomputed distances.
        agv_config: AGV configuration (battery parameters).

    Returns:
        CostMatrix with all costs integer-scaled and feasibility computed.
    """
    from src.warehouse.graph import NodeType

    n_agvs = len(agvs)
    n_tasks = len(tasks)

    # Map station IDs to indices
    unique_stations = sorted(set(t.pick_station for t in tasks))
    station_id_to_idx = {sid: i for i, sid in enumerate(unique_stations)}
    station_index = [station_id_to_idx[t.pick_station] for t in tasks]

    # Precompute nearest charger from each unique return location
    charger_nodes = warehouse.nodes_by_type(NodeType.CHARGING)
    _nearest_charger_cache: dict[str, float] = {}

    def _nearest_charger_dist(from_node: str) -> float:
        if from_node not in _nearest_charger_cache:
            min_d = float("inf")
            for cn in charger_nodes:
                try:
                    d = warehouse.shortest_path_distance(from_node, cn)
                    min_d = min(min_d, d)
                except Exception:
                    continue
            _nearest_charger_cache[from_node] = min_d
        return _nearest_charger_cache[from_node]

    # Build cost arrays
    pickup = [[0] * n_tasks for _ in range(n_agvs)]
    deliver = [[0] * n_tasks for _ in range(n_agvs)]
    ret = [[0] * n_tasks for _ in range(n_agvs)]
    charger = [[0] * n_tasks for _ in range(n_agvs)]
    trip = [[0] * n_tasks for _ in range(n_agvs)]
    feas = [[False] * n_tasks for _ in range(n_agvs)]

    # Precompute per-task distances that don't depend on AGV
    task_deliver_dist: list[float] = []
    task_return_dist: list[float] = []
    task_charger_dist: list[float] = []

    for t in tasks:
        try:
            dd = warehouse.shortest_path_distance(t.pod_location, t.pick_station)
        except Exception:
            dd = float("inf")
        task_deliver_dist.append(dd)

        try:
            rd = warehouse.shortest_path_distance(t.pick_station, t.return_location)
        except Exception:
            rd = float("inf")
        task_return_dist.append(rd)

        task_charger_dist.append(_nearest_charger_dist(t.return_location))

    drain_per_m = agv_config.battery_drain_per_meter
    drain_per_lift = agv_config.battery_drain_per_lift
    battery_threshold = agv_config.battery_threshold

    for a_idx, agv in enumerate(agvs):
        for t_idx, task in enumerate(tasks):
            # Pickup distance (AGV-dependent)
            try:
                pd = warehouse.shortest_path_distance(agv.position, task.pod_location)
            except Exception:
                pd = float("inf")

            dd = task_deliver_dist[t_idx]
            rd = task_return_dist[t_idx]
            cd = task_charger_dist[t_idx]

            # Integer-scale
            pickup[a_idx][t_idx] = _scale(pd)
            deliver[a_idx][t_idx] = _scale(dd)
            ret[a_idx][t_idx] = _scale(rd)
            charger[a_idx][t_idx] = _scale(cd)
            trip[a_idx][t_idx] = _scale(pd) + _scale(dd)

            # Battery feasibility: can the AGV complete the full cycle?
            total_dist = pd + dd + rd + cd
            if total_dist == float("inf"):
                feas[a_idx][t_idx] = False
            else:
                total_drain = total_dist * drain_per_m + 2 * drain_per_lift
                remaining = agv.battery_pct - total_drain
                feas[a_idx][t_idx] = remaining >= battery_threshold

    return CostMatrix(
        n_agvs=n_agvs,
        n_tasks=n_tasks,
        agv_ids=[a.id for a in agvs],
        task_ids=[t.id for t in tasks],
        pickup_dist=pickup,
        deliver_dist=deliver,
        return_dist=ret,
        charger_dist=charger,
        trip_cost=trip,
        feasible=feas,
        station_index=station_index,
        station_ids=unique_stations,
    )


def _scale(value: float) -> int:
    """Scale a float distance to integer for CP-SAT. Inf → large sentinel."""
    if value == float("inf"):
        return 999_999_999
    return int(round(value * COST_SCALE))
