"""
AGV (Automated Guided Vehicle) state machine.

Each AGV runs as an independent SimPy process, cycling through states:
IDLE → TRAVELING_TO_POD → PICKING_UP → TRAVELING_TO_STATION →
WAITING_AT_STATION → PROCESSING → RETURNING_POD → (CHECK_BATTERY) → IDLE

Module 1 MVP uses "teleportation" for movement — the AGV instantly
computes the shortest-path distance and waits for the corresponding
travel time. Module 3 replaces this with real pathfinding via a
pluggable movement strategy.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

import simpy

from src.simulation.stations import pick_processing_time
from src.warehouse.graph import NodeType
from src.simulation.orders import TaskStatus, OrderStatus

if TYPE_CHECKING:
    from src.warehouse.config import AGVConfig
    from src.warehouse.graph import WarehouseGraph
    from src.simulation.orders import Task


class AGVState(Enum):
    """Valid AGV States"""

    IDLE = auto()
    TRAVELING_TO_POD = auto()
    PICKING_UP_POD = auto()
    TRAVELING_TO_STATION = auto()
    WAITING_AT_STATION = auto()
    PROCESSING = auto()
    RETURNING_POD = auto()
    TRAVELING_TO_CHARGER = auto()
    CHARGING = auto()


class MovementStrategy(Protocol):
    """Protocol for AGV movement. Module 1 uses teleportation;
    Module 3 swaps in real pathfinding."""

    def compute_travel_time(self, agv_id: str, from_node: str, to_node: str) -> float:
        """Return travel time in seconds from from_node to to_node."""

    def move(self, env: simpy.Environment, agv_id: str, from_node: str, to_node: str):
        """SimPy generator: yield for the travel duration, update position."""


class TeleportMovement:
    """MVP movement: instant shortest-path lookup, wait for travel time.

    No conflict detection, no reservation. The AGV "teleports" along the
    shortest path in the time it would take to physically traverse it.
    """

    def __init__(self, warehouse: WarehouseGraph, speed_mps: float) -> None:
        self.warehouse = warehouse
        self.speed_mps = speed_mps

    # pylint: disable=unused-argument

    def compute_travel_time(self, agv_id: str, from_node: str, to_node: str) -> float:
        """Computes travel time"""

        if from_node == to_node:
            return 0.0
        distance = self.warehouse.shortest_path_distance(from_node, to_node)
        return distance / self.speed_mps

    def compute_travel_distance(self, agv_id: str, from_node: str, to_node: str) -> float:
        """Computes travel distance"""

        if from_node == to_node:
            return 0.0
        return self.warehouse.shortest_path_distance(from_node, to_node)

    # pylint: enable=unused-argument

    def move(self, env: simpy.Environment, agv_id: str, from_node: str, to_node: str):
        """SimPy generator: yield for travel time."""

        travel_time = self.compute_travel_time(agv_id, from_node, to_node)
        yield env.timeout(travel_time)


@dataclass
class AGVMetrics:
    """Running counters for a single AGV."""

    total_distance_m: float = 0.0
    total_travel_time_s: float = 0.0
    total_idle_time_s: float = 0.0
    total_charging_time_s: float = 0.0
    total_wait_time_s: float = 0.0
    tasks_completed: int = 0
    state_log: list[tuple[float, AGVState]] = field(default_factory=list)

    def log_state(self, time: float, state: AGVState) -> None:
        """Log Simulation State"""

        self.state_log.append((time, state))


@dataclass
class ActiveTravel:
    """In-flight movement metadata used for live interpolation."""

    from_node: str
    to_node: str
    path_nodes: list[str]
    edge_distances_m: list[float]
    cumulative_distances_m: list[float]
    start_time_s: float
    end_time_s: float


class AGV:
    """A single Automated Guided Vehicle running as a SimPy process.

    Attributes:
        id: Unique identifier (e.g., "AGV_001").
        position: Current node in the warehouse graph.
        state: Current lifecycle state.
        battery_pct: Current battery level (0–100).
        current_task: The task being worked on, or None.
        metrics: Running performance counters.
    """

    def __init__(
        self,
        agv_id: str,
        start_position: str,
        env: simpy.Environment,
        config: AGVConfig,
        warehouse: WarehouseGraph,
        movement: MovementStrategy,
        pick_stations: dict[str, simpy.Resource],
        charging_stations: dict[str, simpy.Resource],
    ) -> None:
        self.id = agv_id
        self.position = start_position
        self.env = env
        self.config = config
        self.warehouse = warehouse
        self.movement = movement
        self.pick_stations = pick_stations
        self.charging_stations = charging_stations

        self.state = AGVState.IDLE
        self.battery_pct = config.battery_capacity_pct
        self.current_task: Task | None = None
        self.metrics = AGVMetrics()
        self._active_travel: ActiveTravel | None = None

        # SimPy event: dispatcher sets this to wake up an idle AGV
        self.task_assigned_event = env.event()

    def _set_state(self, new_state: AGVState) -> None:
        """Update state and log the transition."""

        self.state = new_state
        self.metrics.log_state(self.env.now, new_state)

    def _drain_battery(self, distance_m: float) -> None:
        """Reduce battery based on distance traveled."""

        drain = distance_m * self.config.battery_drain_per_meter
        self.battery_pct = max(0.0, self.battery_pct - drain)

    def _drain_battery_lift(self) -> None:
        """Reduce battery for pod lift/place operation."""

        self.battery_pct = max(0.0, self.battery_pct - self.config.battery_drain_per_lift)

    def _needs_charging(self) -> bool:
        """Check if battery is below threshold."""

        return self.battery_pct < self.config.battery_threshold

    def assign_task(self, task: Task) -> None:
        """Called by the dispatcher to assign a task to this AGV.

        Triggers the task_assigned_event to wake the AGV from IDLE.
        """

        self.current_task = task
        task.assigned_agv = self.id
        task.assigned_time = self.env.now

        # Wake the AGV if it's waiting
        if not self.task_assigned_event.triggered:
            self.task_assigned_event.succeed()

    def _start_active_travel(self, to_node: str, distance_m: float) -> None:
        """Store path/time metadata for live interpolation during travel."""

        from_node = self.position
        if from_node == to_node:
            self._active_travel = None
            return

        path_nodes = self.warehouse.shortest_path(from_node, to_node)
        edge_distances = [
            self.warehouse.edge_distance(path_nodes[i], path_nodes[i + 1])
            for i in range(len(path_nodes) - 1)
        ]
        cumulative = [0.0]
        for edge_dist in edge_distances:
            cumulative.append(cumulative[-1] + edge_dist)

        travel_time_s = distance_m / self.config.speed_mps if self.config.speed_mps > 0 else 0.0
        self._active_travel = ActiveTravel(
            from_node=from_node,
            to_node=to_node,
            path_nodes=path_nodes,
            edge_distances_m=edge_distances,
            cumulative_distances_m=cumulative,
            start_time_s=self.env.now,
            end_time_s=self.env.now + travel_time_s,
        )

    def _clear_active_travel(self) -> None:
        self._active_travel = None

    def current_edge(self, now_s: float) -> tuple[str, str] | None:
        """Return the directed edge currently traversed, if any."""

        travel = self._active_travel
        if travel is None or len(travel.path_nodes) < 2:
            return None

        total_dist = travel.cumulative_distances_m[-1]
        if total_dist <= 0:
            return None

        duration = travel.end_time_s - travel.start_time_s
        if duration <= 0:
            return None

        progress = max(0.0, min(1.0, (now_s - travel.start_time_s) / duration))
        if progress >= 1.0:
            return None

        dist_along_path = progress * total_dist
        seg_idx = bisect.bisect_right(travel.cumulative_distances_m, dist_along_path) - 1
        seg_idx = max(0, min(seg_idx, len(travel.path_nodes) - 2))
        return travel.path_nodes[seg_idx], travel.path_nodes[seg_idx + 1]

    def live_position(self, now_s: float) -> tuple[float, float]:
        """Interpolated XY position for UI rendering at simulation time now_s."""

        travel = self._active_travel
        if travel is None or len(travel.path_nodes) < 2:
            node = self.warehouse.get_node(self.position)
            return float(node["x"]), float(node["y"])

        total_dist = travel.cumulative_distances_m[-1]
        duration = travel.end_time_s - travel.start_time_s
        if total_dist <= 0 or duration <= 0:
            node = self.warehouse.get_node(travel.to_node)
            return float(node["x"]), float(node["y"])

        progress = max(0.0, min(1.0, (now_s - travel.start_time_s) / duration))
        if progress >= 1.0:
            node = self.warehouse.get_node(travel.to_node)
            return float(node["x"]), float(node["y"])

        dist_along_path = progress * total_dist
        seg_idx = bisect.bisect_right(travel.cumulative_distances_m, dist_along_path) - 1
        seg_idx = max(0, min(seg_idx, len(travel.path_nodes) - 2))

        seg_start = travel.cumulative_distances_m[seg_idx]
        seg_len = travel.edge_distances_m[seg_idx]
        seg_ratio = 0.0 if seg_len <= 0 else (dist_along_path - seg_start) / seg_len

        from_node = self.warehouse.get_node(travel.path_nodes[seg_idx])
        to_node = self.warehouse.get_node(travel.path_nodes[seg_idx + 1])

        x = float(from_node["x"]) + seg_ratio * (float(to_node["x"]) - float(from_node["x"]))
        y = float(from_node["y"]) + seg_ratio * (float(to_node["y"]) - float(from_node["y"]))
        return x, y

    def run(self):
        """Main SimPy process loop. Runs for the lifetime of the simulation."""

        self._set_state(AGVState.IDLE)

        while True:
            # ── IDLE: Wait for task assignment ───────────────────────
            idle_start = self.env.now

            if self.current_task is None:
                yield self.task_assigned_event
                # Reset event for next cycle
                self.task_assigned_event = self.env.event()

            self.metrics.total_idle_time_s += self.env.now - idle_start
            task = self.current_task
            if task is None:
                continue  # spurious wake-up

            # ── TRAVELING TO POD ─────────────────────────────────────
            self._set_state(AGVState.TRAVELING_TO_POD)
            distance = self.movement.compute_travel_distance(
                self.id, self.position, task.pod_location
            )
            self._start_active_travel(task.pod_location, distance)
            yield from self.movement.move(self.env, self.id, self.position, task.pod_location)
            self.position = task.pod_location
            self._clear_active_travel()
            self._drain_battery(distance)
            self.metrics.total_distance_m += distance
            self.metrics.total_travel_time_s += distance / self.config.speed_mps

            # ── PICKING UP POD ───────────────────────────────────────
            self._set_state(AGVState.PICKING_UP_POD)
            yield self.env.timeout(self.config.pod_pickup_time_s)
            self._drain_battery_lift()

            # ── TRAVELING TO PICK STATION ────────────────────────────
            self._set_state(AGVState.TRAVELING_TO_STATION)
            distance = self.movement.compute_travel_distance(
                self.id, self.position, task.pick_station
            )
            self._start_active_travel(task.pick_station, distance)
            yield from self.movement.move(self.env, self.id, self.position, task.pick_station)
            self.position = task.pick_station
            self._clear_active_travel()
            self._drain_battery(distance)
            self.metrics.total_distance_m += distance
            self.metrics.total_travel_time_s += distance / self.config.speed_mps

            # ── WAITING AT STATION (queue for pick station resource) ─
            self._set_state(AGVState.WAITING_AT_STATION)
            station_resource = self.pick_stations[task.pick_station]
            wait_start = self.env.now

            with station_resource.request() as req:
                yield req
                self.metrics.total_wait_time_s += self.env.now - wait_start

                # ── PROCESSING (human picks items from pod) ──────────
                self._set_state(AGVState.PROCESSING)
                # Processing time is yielded by the station (see stations.py)
                # Here we just yield the pick time for this order

                process_time = pick_processing_time(
                    task,
                    self.config,
                    self.env.now,  # we pass env.now as seed proxy
                )
                yield self.env.timeout(process_time)

            # ── RETURNING POD TO STORAGE ─────────────────────────────
            self._set_state(AGVState.RETURNING_POD)
            distance = self.movement.compute_travel_distance(
                self.id, self.position, task.return_location
            )
            self._start_active_travel(task.return_location, distance)
            yield from self.movement.move(self.env, self.id, self.position, task.return_location)
            self.position = task.return_location
            self._clear_active_travel()
            self._drain_battery(distance)
            self.metrics.total_distance_m += distance
            self.metrics.total_travel_time_s += distance / self.config.speed_mps

            # Pod drop-off
            yield self.env.timeout(self.config.pod_dropoff_time_s)
            self._drain_battery_lift()

            # Mark task complete
            task.status = TaskStatus.COMPLETE
            task.completed_time = self.env.now
            self.metrics.tasks_completed += 1

            # Propagate completion to parent order
            if task.order_ref is not None:
                task.order_ref.status = OrderStatus.COMPLETE
                task.order_ref.completion_time = self.env.now

            self.current_task = None

            # ── CHECK BATTERY ────────────────────────────────────────
            if self._needs_charging():
                yield from self._charge()

            # Loop back to IDLE
            self._set_state(AGVState.IDLE)

    def _charge(self):
        """Navigate to nearest charger, charge to full, return."""

        # Find nearest charging station
        charger_node, charge_dist = self.warehouse.nearest_node_of_type(
            self.position, NodeType.CHARGING
        )

        # Travel to charger
        self._set_state(AGVState.TRAVELING_TO_CHARGER)
        self._start_active_travel(charger_node, charge_dist)
        yield from self.movement.move(self.env, self.id, self.position, charger_node)
        self.position = charger_node
        self._clear_active_travel()
        self._drain_battery(charge_dist)
        self.metrics.total_distance_m += charge_dist

        # Charge (queue for charging station resource)
        self._set_state(AGVState.CHARGING)
        charger_resource = self.charging_stations[charger_node]
        with charger_resource.request() as req:
            yield req
            # Charge to full
            deficit = self.config.battery_capacity_pct - self.battery_pct
            charge_time = deficit / self.config.battery_charge_rate
            yield self.env.timeout(charge_time)
            self.battery_pct = self.config.battery_capacity_pct
            self.metrics.total_charging_time_s += charge_time
