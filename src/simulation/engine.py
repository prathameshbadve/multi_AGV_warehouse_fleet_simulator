"""
Simulation engine — the top-level orchestrator.

Wires together the warehouse graph, AGV fleet, order generator, stations,
and metrics collector into a single runnable simulation.

Usage:
    config = load_config("config/default_warehouse.yaml")
    engine = SimulationEngine(config, n_agvs=50)
    results = engine.run()
    print(f"Throughput: {results.avg_throughput_per_hour:.0f} orders/hour")

The engine uses a task dispatcher that runs as a SimPy process,
periodically checking for pending tasks and assigning them to idle AGVs.

Module 1: greedy nearest-AGV assignment.
Module 2: CP-SAT optimizer (default, with greedy fallback).
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import simpy

from src.warehouse.config import WarehouseConfig, load_config
from src.warehouse.graph import WarehouseGraph, NodeType
from src.warehouse.layout import GridLayoutGenerator
from src.simulation.agv import AGV, AGVState, TeleportMovement
from src.simulation.orders import OrderGenerator, Task, TaskStatus, OrderStatus
from src.simulation.stations import (
    create_pick_station_resources,
    create_charging_station_resources,
    set_station_rng,
)
from src.simulation.metrics import MetricsCollector, SimulationMetrics


class SimulationEngine:
    """Top-level simulation orchestrator.

    Args:
        config: Full warehouse configuration.
        n_agvs: Number of AGVs to deploy.
        assignment_strategy: Which assignment solver to use.
            "cpsat" (default) uses Module 2 CP-SAT optimizer with greedy fallback.
            "greedy" uses Module 1 nearest-neighbor baseline.
        assignment_fn: Optional custom assignment function (overrides strategy).
            Signature: (pending_tasks, idle_agvs, warehouse) → list[(AGV, Task)]
    """

    def __init__(
        self,
        config: WarehouseConfig,
        n_agvs: int = 50,
        assignment_strategy: Literal["cpsat", "greedy"] = "cpsat",
        assignment_fn: Callable | None = None,
    ) -> None:
        self.config = config
        self.n_agvs = n_agvs
        self.assignment_strategy = assignment_strategy
        self._custom_assignment_fn = assignment_fn

        self.warehouse: WarehouseGraph | None = None
        self.agvs: list[AGV] = []
        self.metrics: SimulationMetrics | None = None

        # Tracks AGVs assigned to each station but not yet arrived/processing.
        # Updated by the dispatcher; read by the CP-SAT solver for balancing.
        self._inflight_station_counts: dict[str, int] = {}

    def run(self) -> SimulationMetrics:
        """Execute the full simulation and return results.

        Returns:
            SimulationMetrics with time-series and summary statistics.
        """
        # ── Build warehouse ──────────────────────────────────────────
        layout_gen = GridLayoutGenerator(self.config)
        self.warehouse = layout_gen.generate()

        # Precompute distances for assignment heuristic
        self.warehouse.precompute_distances()  # all-pairs for MVP scale

        # ── Setup SimPy environment ──────────────────────────────────
        env = simpy.Environment()
        rng = np.random.default_rng(self.config.simulation.random_seed)
        set_station_rng(self.config.simulation.random_seed + 1)

        # ── Create station resources ─────────────────────────────────
        pick_station_ids = self.warehouse.nodes_by_type(NodeType.PICK_STATION)
        charging_station_ids = self.warehouse.nodes_by_type(NodeType.CHARGING)

        pick_resources = create_pick_station_resources(env, pick_station_ids)
        charging_resources = create_charging_station_resources(env, charging_station_ids)

        # Initialize inflight counter
        self._inflight_station_counts = {sid: 0 for sid in pick_station_ids}

        # ── Create assignment function ────────────────────────────────
        assignment_fn = self._build_assignment_fn(pick_resources)

        # ── Create movement strategy ─────────────────────────────────
        movement = TeleportMovement(self.warehouse, self.config.agv.speed_mps)

        # ── Create AGV fleet ─────────────────────────────────────────
        # Start AGVs at parking spots, then overflow to intersections
        start_positions = self.warehouse.nodes_by_type(NodeType.PARKING)
        intersection_starts = self.warehouse.nodes_by_type(NodeType.INTERSECTION)
        while len(start_positions) < self.n_agvs:
            start_positions.extend(intersection_starts)

        self.agvs = []
        for i in range(self.n_agvs):
            start_pos = start_positions[i % len(start_positions)]
            agv = AGV(
                agv_id=f"AGV_{i:03d}",
                start_position=start_pos,
                env=env,
                config=self.config.agv,
                warehouse=self.warehouse,
                movement=movement,
                pick_stations=pick_resources,
                charging_stations=charging_resources,
            )
            self.agvs.append(agv)
            env.process(agv.run())

        # ── Create order generator ───────────────────────────────────
        order_gen = OrderGenerator(
            env=env,
            config=self.config.orders,
            warehouse=self.warehouse,
            rng=rng,
        )
        env.process(order_gen.run())

        # ── Create metrics collector ─────────────────────────────────
        collector = MetricsCollector(
            env=env,
            agvs=self.agvs,
            order_generator=order_gen,
            pick_station_resources=pick_resources,
            charging_station_resources=charging_resources,
            interval_s=self.config.simulation.metrics_interval_s,
        )
        env.process(collector.run())

        # ── Create task dispatcher ───────────────────────────────────
        env.process(
            self._dispatcher_process(env, order_gen, assignment_fn)
        )

        # ── Run! ─────────────────────────────────────────────────────
        duration = self.config.simulation.duration_s
        print(f"Starting simulation: {self.n_agvs} AGVs, "
              f"{self.config.stations.n_pick_stations} pick stations, "
              f"{duration/3600:.1f} hours, "
              f"assignment={self.assignment_strategy}")

        env.run(until=duration)

        # ── Compute final metrics ────────────────────────────────────
        self.metrics = collector.compute_final_metrics()

        print(f"\nSimulation complete:")
        print(f"  Orders generated: {self.metrics.total_orders_generated}")
        print(f"  Orders completed: {self.metrics.total_orders_completed}")
        print(f"  Avg throughput:   {self.metrics.avg_throughput_per_hour:.0f} orders/hour")
        print(f"  Avg cycle time:   {self.metrics.avg_cycle_time_s:.1f}s")
        print(f"  P95 cycle time:   {self.metrics.p95_cycle_time_s:.1f}s")
        print(f"  AGV utilization:  {self.metrics.avg_agv_utilization_pct:.1f}%")
        print(f"  Station util:     {self.metrics.avg_station_utilization_pct:.1f}%")

        return self.metrics

    # ── Assignment function factory ───────────────────────────────

    def _build_assignment_fn(
        self, pick_resources: dict[str, simpy.Resource]
    ) -> Callable:
        """Build the assignment function based on the selected strategy.

        If a custom assignment_fn was provided, use it directly.
        Otherwise, construct the appropriate solver.
        """
        if self._custom_assignment_fn is not None:
            return self._custom_assignment_fn

        if self.assignment_strategy == "cpsat":
            from src.assignment.solver import CPSATAssignmentSolver

            solver = CPSATAssignmentSolver(
                pick_station_resources=pick_resources,
                inflight_station_counts=self._inflight_station_counts,
            )
            return solver.solve

        return self._greedy_assignment

    # ── Dispatcher ────────────────────────────────────────────────

    def _dispatcher_process(
        self,
        env: simpy.Environment,
        order_gen: OrderGenerator,
        assignment_fn: Callable,
    ):
        """SimPy process: periodically assign pending tasks to idle AGVs.

        Runs every `dispatch_interval_s` seconds. Collects pending tasks
        from the order generator, finds idle AGVs, and calls the
        assignment function.
        """
        interval = self.config.simulation.dispatch_interval_s

        # Small initial delay to let orders accumulate
        yield env.timeout(interval)

        while True:
            # Update inflight station counts from currently assigned tasks
            self._update_inflight_counts()

            # Gather pending tasks
            new_tasks = order_gen.get_and_clear_pending()

            # Also include previously unassigned tasks that are still pending
            # (tasks where no AGV was available last cycle)
            pending_tasks = [
                t for t in order_gen.tasks
                if t.status == TaskStatus.UNASSIGNED
            ]

            if pending_tasks:
                # Find idle AGVs
                idle_agvs = [a for a in self.agvs if a.state == AGVState.IDLE and a.current_task is None]

                if idle_agvs:
                    assignments = assignment_fn(pending_tasks, idle_agvs, self.warehouse)
                    for agv, task in assignments:
                        task.status = TaskStatus.ASSIGNED
                        if task.order_ref is not None:
                            task.order_ref.status = OrderStatus.ASSIGNED
                        agv.assign_task(task)

            yield env.timeout(interval)

    def _update_inflight_counts(self) -> None:
        """Count AGVs heading to each station but not yet processing.

        An AGV is "inflight" to station S if it has been assigned a task
        targeting S and is in state TRAVELING_TO_POD, PICKING_UP_POD,
        or TRAVELING_TO_STATION.
        """
        inflight_states = {
            AGVState.TRAVELING_TO_POD,
            AGVState.PICKING_UP_POD,
            AGVState.TRAVELING_TO_STATION,
        }

        # Reset counts
        for sid in self._inflight_station_counts:
            self._inflight_station_counts[sid] = 0

        for agv in self.agvs:
            if agv.state in inflight_states and agv.current_task is not None:
                station = agv.current_task.pick_station
                if station in self._inflight_station_counts:
                    self._inflight_station_counts[station] += 1

    @staticmethod
    def _greedy_assignment(
        pending_tasks: list[Task],
        idle_agvs: list[AGV],
        warehouse: WarehouseGraph,
    ) -> list[tuple[AGV, Task]]:
        """Greedy nearest-neighbor assignment.

        For each task (in priority order), assign the closest idle AGV.
        This is the Module 1 baseline.

        Returns:
            List of (AGV, Task) pairs.
        """
        from src.simulation.orders import OrderPriority

        # Sort tasks: express first, then by creation time
        sorted_tasks = sorted(
            pending_tasks,
            key=lambda t: (0 if t.priority == OrderPriority.EXPRESS else 1, t.created_time),
        )

        available_agvs = set(a.id for a in idle_agvs)
        agv_map = {a.id: a for a in idle_agvs}
        assignments = []

        for task in sorted_tasks:
            if not available_agvs:
                break

            # Find nearest available AGV to the pod location
            best_agv_id = None
            best_dist = float("inf")

            for agv_id in available_agvs:
                agv = agv_map[agv_id]
                try:
                    dist = warehouse.shortest_path_distance(agv.position, task.pod_location)
                    if dist < best_dist:
                        best_dist = dist
                        best_agv_id = agv_id
                except Exception:
                    continue

            if best_agv_id is not None:
                assignments.append((agv_map[best_agv_id], task))
                available_agvs.remove(best_agv_id)

        return assignments
