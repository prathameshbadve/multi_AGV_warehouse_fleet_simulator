# """
# Simulation engine — the top-level orchestrator.

# Wires together the warehouse graph, AGV fleet, order generator, stations,
# and metrics collector into a single runnable simulation.

# Usage:
#     config = load_config("config/default_warehouse.yaml")
#     engine = SimulationEngine(config, n_agvs=50)
#     results = engine.run()
#     print(f"Throughput: {results.avg_throughput_per_hour:.0f} orders/hour")

# The engine uses a task dispatcher that runs as a SimPy process,
# periodically checking for pending tasks and assigning them to idle AGVs.

# Module 1: greedy nearest-AGV assignment.
# Module 2: CP-SAT optimizer (default, with greedy fallback).
# """

# from __future__ import annotations

# from collections import Counter
# from typing import Callable, Literal

# import numpy as np
# import simpy

# from src.warehouse.config import WarehouseConfig
# from src.warehouse.graph import WarehouseGraph, NodeType
# from src.warehouse.layout import GridLayoutGenerator
# from src.simulation.agv import AGV, AGVState, TeleportMovement
# from src.simulation.orders import OrderGenerator, Task, TaskStatus, OrderStatus, OrderPriority
# from src.simulation.stations import (
#     create_pick_station_resources,
#     create_charging_station_resources,
#     set_station_rng,
# )
# from src.simulation.metrics import MetricsCollector, SimulationMetrics
# from src.assignment.solver import CPSATAssignmentSolver


# class SimulationEngine:
#     """Top-level simulation orchestrator.

#     Args:
#         config: Full warehouse configuration.
#         n_agvs: Number of AGVs to deploy.
#         assignment_strategy: Which assignment solver to use.
#             "cpsat" (default) uses Module 2 CP-SAT optimizer with greedy fallback.
#             "greedy" uses Module 1 nearest-neighbor baseline.
#         assignment_fn: Optional custom assignment function (overrides strategy).
#             Signature: (pending_tasks, idle_agvs, warehouse) → list[(AGV, Task)]
#     """

#     def __init__(
#         self,
#         config: WarehouseConfig,
#         n_agvs: int = 50,
#         assignment_strategy: Literal["cpsat", "greedy"] = "cpsat",
#         assignment_fn: Callable | None = None,
#     ) -> None:
#         self.config = config
#         self.n_agvs = n_agvs
#         self.assignment_strategy = assignment_strategy
#         self._custom_assignment_fn = assignment_fn

#         self.warehouse: WarehouseGraph | None = None
#         self.agvs: list[AGV] = []
#         self.metrics: SimulationMetrics | None = None

#         # Tracks AGVs assigned to each station but not yet arrived/processing.
#         # Updated by the dispatcher; read by the CP-SAT solver for balancing.
#         self._inflight_station_counts: dict[str, int] = {}

#     def run(
#         self,
#         runtime_interval_s: float | None = None,
#         runtime_callback: Callable[[dict], None] | None = None,
#         layout_callback: Callable[[dict], None] | None = None,
#         stop_requested: Callable[[], bool] | None = None,
#     ) -> SimulationMetrics:
#         """Execute the full simulation and return results.

#         Args:
#             runtime_interval_s: Snapshot interval (seconds) for live streaming.
#             runtime_callback: Called with runtime snapshot payloads if provided.
#             layout_callback: Called once with static warehouse layout payload.
#             stop_requested: Optional cancellation predicate for long-running UI sessions.

#         Returns:
#             SimulationMetrics with time-series and summary statistics.
#         """
#         # ── Build warehouse ──────────────────────────────────────────
#         layout_gen = GridLayoutGenerator(self.config)
#         self.warehouse = layout_gen.generate()

#         # Precompute distances for assignment heuristic
#         self.warehouse.precompute_distances()  # all-pairs for MVP scale
#         if layout_callback is not None:
#             layout_callback(self._build_layout_payload())

#         # ── Setup SimPy environment ──────────────────────────────────
#         env = simpy.Environment()
#         rng = np.random.default_rng(self.config.simulation.random_seed)
#         set_station_rng(self.config.simulation.random_seed + 1)

#         # ── Create station resources ─────────────────────────────────
#         pick_station_ids = self.warehouse.nodes_by_type(NodeType.PICK_STATION)
#         charging_station_ids = self.warehouse.nodes_by_type(NodeType.CHARGING)

#         pick_resources = create_pick_station_resources(env, pick_station_ids)
#         charging_resources = create_charging_station_resources(env, charging_station_ids)

#         # Initialize inflight counter
#         self._inflight_station_counts = {sid: 0 for sid in pick_station_ids}

#         # ── Create assignment function ────────────────────────────────
#         assignment_fn = self._build_assignment_fn(pick_resources)

#         # ── Create movement strategy ─────────────────────────────────
#         movement = TeleportMovement(self.warehouse, self.config.agv.speed_mps)

#         # ── Create AGV fleet ─────────────────────────────────────────
#         # Start AGVs at parking spots, then overflow to intersections
#         start_positions = self.warehouse.nodes_by_type(NodeType.PARKING)
#         intersection_starts = self.warehouse.nodes_by_type(NodeType.INTERSECTION)
#         while len(start_positions) < self.n_agvs:
#             start_positions.extend(intersection_starts)

#         self.agvs = []
#         for i in range(self.n_agvs):
#             start_pos = start_positions[i % len(start_positions)]
#             agv = AGV(
#                 agv_id=f"AGV_{i:03d}",
#                 start_position=start_pos,
#                 env=env,
#                 config=self.config.agv,
#                 warehouse=self.warehouse,
#                 movement=movement,
#                 pick_stations=pick_resources,
#                 charging_stations=charging_resources,
#             )
#             self.agvs.append(agv)
#             env.process(agv.run())

#         # ── Create order generator ───────────────────────────────────
#         order_gen = OrderGenerator(
#             env=env,
#             config=self.config.orders,
#             warehouse=self.warehouse,
#             rng=rng,
#         )
#         env.process(order_gen.run())

#         # ── Create metrics collector ─────────────────────────────────
#         collector = MetricsCollector(
#             env=env,
#             agvs=self.agvs,
#             order_generator=order_gen,
#             pick_station_resources=pick_resources,
#             charging_station_resources=charging_resources,
#             interval_s=self.config.simulation.metrics_interval_s,
#         )
#         env.process(collector.run())

#         # ── Create task dispatcher ───────────────────────────────────
#         env.process(self._dispatcher_process(env, order_gen, assignment_fn))

#         # ── Optional runtime stream (for UI) ────────────────────────
#         if runtime_callback is not None and runtime_interval_s is not None and runtime_interval_s > 0:
#             env.process(
#                 self._runtime_monitor(
#                     env,
#                     order_gen,
#                     pick_resources,
#                     runtime_interval_s,
#                     runtime_callback,
#                 )
#             )

#         # ── Run! ─────────────────────────────────────────────────────
#         duration = self.config.simulation.duration_s
#         print(
#             f"Starting simulation: {self.n_agvs} AGVs, "
#             f"{self.config.stations.n_pick_stations} pick stations, "
#             f"{duration / 3600:.1f} hours, "
#             f"assignment={self.assignment_strategy}"
#         )

#         if stop_requested is None:
#             env.run(until=duration)
#         else:
#             step_s = max(0.1, min(1.0, self.config.simulation.dispatch_interval_s))
#             while env.now < duration:
#                 if stop_requested():
#                     break
#                 env.run(until=min(duration, env.now + step_s))

#         # ── Compute final metrics ────────────────────────────────────
#         self.metrics = collector.compute_final_metrics()
#         if runtime_callback is not None:
#             runtime_callback(
#                 self._build_runtime_snapshot(
#                     env=env,
#                     order_gen=order_gen,
#                     pick_resources=pick_resources,
#                     final=True,
#                 )
#             )

#         print("\nSimulation complete:")
#         print(f"  Orders generated: {self.metrics.total_orders_generated}")
#         print(f"  Orders completed: {self.metrics.total_orders_completed}")
#         print(f"  Avg throughput:   {self.metrics.avg_throughput_per_hour:.0f} orders/hour")
#         print(f"  Avg cycle time:   {self.metrics.avg_cycle_time_s:.1f}s")
#         print(f"  P95 cycle time:   {self.metrics.p95_cycle_time_s:.1f}s")
#         print(f"  AGV utilization:  {self.metrics.avg_agv_utilization_pct:.1f}%")
#         print(f"  Station util:     {self.metrics.avg_station_utilization_pct:.1f}%")

#         return self.metrics

#     # ── Runtime snapshots for UI ──────────────────────────────────

#     def _runtime_monitor(
#         self,
#         env: simpy.Environment,
#         order_gen: OrderGenerator,
#         pick_resources: dict[str, simpy.Resource],
#         interval_s: float,
#         runtime_callback: Callable[[dict], None],
#     ):
#         """SimPy process: emit runtime snapshots at fixed intervals."""

#         interval_s = max(0.05, interval_s)
#         while True:
#             runtime_callback(
#                 self._build_runtime_snapshot(
#                     env=env,
#                     order_gen=order_gen,
#                     pick_resources=pick_resources,
#                     final=False,
#                 )
#             )
#             yield env.timeout(interval_s)

#     @staticmethod
#     def _node_capacity(node_type: NodeType) -> int:
#         """Simple per-node occupancy baseline for congestion coloring."""

#         capacities = {
#             NodeType.INTERSECTION: 2,
#             NodeType.PICK_STATION: 2,
#             NodeType.CHARGING: 2,
#             NodeType.AISLE_POINT: 1,
#             NodeType.STORAGE: 1,
#             NodeType.PARKING: 1,
#             NodeType.DEPOT: 1,
#         }
#         return capacities.get(node_type, 1)

#     def _build_layout_payload(self) -> dict:
#         """Create static graph payload for the UI."""

#         if self.warehouse is None:
#             raise RuntimeError("Warehouse is not initialized")

#         g = self.warehouse.graph
#         nodes = []
#         xs: list[float] = []
#         ys: list[float] = []

#         for node_id, attrs in g.nodes(data=True):
#             node_type = attrs["node_type"]
#             x = float(attrs["x"])
#             y = float(attrs["y"])
#             xs.append(x)
#             ys.append(y)
#             nodes.append(
#                 {
#                     "id": node_id,
#                     "x": x,
#                     "y": y,
#                     "type": node_type.name,
#                     "capacity": self._node_capacity(node_type),
#                 }
#             )

#         edges = []
#         for from_node, to_node, attrs in g.edges(data=True):
#             edge_id = f"{from_node}->{to_node}"
#             edges.append(
#                 {
#                     "id": edge_id,
#                     "from": from_node,
#                     "to": to_node,
#                     "type": attrs["edge_type"].name,
#                     "capacity": int(attrs.get("capacity", 1)),
#                     "distance_m": float(attrs.get("distance", 0.0)),
#                 }
#             )

#         padding = 2.0
#         bounds = {
#             "min_x": (min(xs) - padding) if xs else -1.0,
#             "max_x": (max(xs) + padding) if xs else 1.0,
#             "min_y": (min(ys) - padding) if ys else -1.0,
#             "max_y": (max(ys) + padding) if ys else 1.0,
#         }

#         return {"nodes": nodes, "edges": edges, "bounds": bounds}

#     def _build_runtime_snapshot(
#         self,
#         env: simpy.Environment,
#         order_gen: OrderGenerator,
#         pick_resources: dict[str, simpy.Resource],
#         final: bool,
#     ) -> dict:
#         """Create a live snapshot for front-end visualization."""

#         from src.simulation.orders import OrderStatus, TaskStatus

#         now = float(env.now)
#         duration = float(self.config.simulation.duration_s)
#         completed_orders = [o for o in order_gen.orders if o.status == OrderStatus.COMPLETE]
#         pending_orders = [o for o in order_gen.orders if o.status != OrderStatus.COMPLETE]

#         edge_occupancy: Counter[str] = Counter()
#         node_occupancy: Counter[str] = Counter()
#         agv_states: Counter[str] = Counter()
#         agv_payload = []

#         for agv in sorted(self.agvs, key=lambda a: a.id):
#             agv_states[agv.state.name] += 1
#             x, y = agv.live_position(now)
#             edge = agv.current_edge(now)
#             edge_id = None
#             if edge is None:
#                 node_occupancy[agv.position] += 1
#             else:
#                 edge_id = f"{edge[0]}->{edge[1]}"
#                 edge_occupancy[edge_id] += 1

#             agv_payload.append(
#                 {
#                     "id": agv.id,
#                     "state": agv.state.name,
#                     "node_id": agv.position,
#                     "battery_pct": round(float(agv.battery_pct), 2),
#                     "task_id": agv.current_task.id if agv.current_task is not None else None,
#                     "x": x,
#                     "y": y,
#                     "current_edge_id": edge_id,
#                 }
#             )

#         station_queues = {
#             sid: len(resource.queue) + len(resource.users) for sid, resource in pick_resources.items()
#         }

#         n_idle = agv_states.get(AGVState.IDLE.name, 0)
#         n_charging = agv_states.get(AGVState.CHARGING.name, 0) + agv_states.get(
#             AGVState.TRAVELING_TO_CHARGER.name, 0
#         )
#         n_active = len(self.agvs) - n_idle - n_charging
#         utilization_pct = (n_active / len(self.agvs) * 100.0) if self.agvs else 0.0

#         throughput = len(completed_orders) / (now / 3600.0) if now > 0 else 0.0
#         tasks_unassigned = sum(1 for t in order_gen.tasks if t.status == TaskStatus.UNASSIGNED)
#         tasks_active = sum(
#             1 for t in order_gen.tasks if t.status in {TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}
#         )

#         return {
#             "time_s": now,
#             "final": final,
#             "progress_pct": min(100.0, (now / duration * 100.0)) if duration > 0 else 0.0,
#             "agvs": agv_payload,
#             "congestion": {
#                 "nodes": dict(node_occupancy),
#                 "edges": dict(edge_occupancy),
#             },
#             "metrics": {
#                 "orders_generated": len(order_gen.orders),
#                 "orders_completed": len(completed_orders),
#                 "orders_pending": len(pending_orders),
#                 "tasks_unassigned": tasks_unassigned,
#                 "tasks_active": tasks_active,
#                 "agv_utilization_pct": utilization_pct,
#                 "throughput_per_hour": throughput,
#                 "agv_states": dict(agv_states),
#                 "station_queues": station_queues,
#             },
#         }

#     # ── Assignment function factory ───────────────────────────────

#     def _build_assignment_fn(self, pick_resources: dict[str, simpy.Resource]) -> Callable:
#         """Build the assignment function based on the selected strategy.

#         If a custom assignment_fn was provided, use it directly.
#         Otherwise, construct the appropriate solver.
#         """
#         if self._custom_assignment_fn is not None:
#             return self._custom_assignment_fn

#         if self.assignment_strategy == "cpsat":
#             solver = CPSATAssignmentSolver(
#                 pick_station_resources=pick_resources,
#                 inflight_station_counts=self._inflight_station_counts,
#             )
#             return solver.solve

#         return self._greedy_assignment

#     # ── Dispatcher ────────────────────────────────────────────────

#     def _dispatcher_process(
#         self,
#         env: simpy.Environment,
#         order_gen: OrderGenerator,
#         assignment_fn: Callable,
#     ):
#         """SimPy process: periodically assign pending tasks to idle AGVs.

#         Runs every `dispatch_interval_s` seconds. Collects pending tasks
#         from the order generator, finds idle AGVs, and calls the
#         assignment function.
#         """
#         interval = self.config.simulation.dispatch_interval_s

#         # Small initial delay to let orders accumulate
#         yield env.timeout(interval)

#         while True:
#             # Update inflight station counts from currently assigned tasks
#             self._update_inflight_counts()

#             # Gather pending tasks
#             _ = order_gen.get_and_clear_pending()

#             # Also include previously unassigned tasks that are still pending
#             # (tasks where no AGV was available last cycle)
#             pending_tasks = [t for t in order_gen.tasks if t.status == TaskStatus.UNASSIGNED]

#             if pending_tasks:
#                 # Find idle AGVs
#                 idle_agvs = [
#                     a for a in self.agvs if a.state == AGVState.IDLE and a.current_task is None
#                 ]

#                 if idle_agvs:
#                     assignments = assignment_fn(pending_tasks, idle_agvs, self.warehouse)
#                     for agv, task in assignments:
#                         task.status = TaskStatus.ASSIGNED
#                         if task.order_ref is not None:
#                             task.order_ref.status = OrderStatus.ASSIGNED
#                         agv.assign_task(task)

#             yield env.timeout(interval)

#     def _update_inflight_counts(self) -> None:
#         """Count AGVs heading to each station but not yet processing.

#         An AGV is "inflight" to station S if it has been assigned a task
#         targeting S and is in state TRAVELING_TO_POD, PICKING_UP_POD,
#         or TRAVELING_TO_STATION.
#         """
#         inflight_states = {
#             AGVState.TRAVELING_TO_POD,
#             AGVState.PICKING_UP_POD,
#             AGVState.TRAVELING_TO_STATION,
#         }

#         # Reset counts
#         for sid in self._inflight_station_counts:
#             self._inflight_station_counts[sid] = 0

#         for agv in self.agvs:
#             if agv.state in inflight_states and agv.current_task is not None:
#                 station = agv.current_task.pick_station
#                 if station in self._inflight_station_counts:
#                     self._inflight_station_counts[station] += 1

#     @staticmethod
#     def _greedy_assignment(
#         pending_tasks: list[Task],
#         idle_agvs: list[AGV],
#         warehouse: WarehouseGraph,
#     ) -> list[tuple[AGV, Task]]:
#         """Greedy nearest-neighbor assignment.

#         For each task (in priority order), assign the closest idle AGV.
#         This is the Module 1 baseline.

#         Returns:
#             List of (AGV, Task) pairs.
#         """

#         # Sort tasks: express first, then by creation time
#         sorted_tasks = sorted(
#             pending_tasks,
#             key=lambda t: (0 if t.priority == OrderPriority.EXPRESS else 1, t.created_time),
#         )

#         available_agvs = set(a.id for a in idle_agvs)
#         agv_map = {a.id: a for a in idle_agvs}
#         assignments = []

#         for task in sorted_tasks:
#             if not available_agvs:
#                 break

#             # Find nearest available AGV to the pod location
#             best_agv_id = None
#             best_dist = float("inf")

#             for agv_id in available_agvs:
#                 agv = agv_map[agv_id]
#                 try:
#                     dist = warehouse.shortest_path_distance(agv.position, task.pod_location)
#                     if dist < best_dist:
#                         best_dist = dist
#                         best_agv_id = agv_id
#                 except Exception:  # pylint: disable=broad-exception-caught
#                     continue

#             if best_agv_id is not None:
#                 assignments.append((agv_map[best_agv_id], task))
#                 available_agvs.remove(best_agv_id)

#         return assignments


"""
src/simulation/engine.py
──────────────────────────────────────────────────────────────────────────────
Simulation engine — top-level orchestrator.

Wires together the warehouse graph, AGV fleet, order generator, stations,
and metrics collector into a single runnable simulation.

Usage:
    config = load_config("config/default_warehouse.yaml")
    engine = SimulationEngine(config, n_agvs=50)
    results = engine.run()
    print(f"Throughput: {results.avg_throughput_per_hour:.0f} orders/hour")

Assignment strategies (--strategy CLI flag / assignment_strategy constructor):
    "greedy"    Nearest-neighbour baseline (Module 1)
    "cpsat"     Improved CP-SAT with k-nearest filter + warm hint  ← old default
    "scipy"     scipy.optimize.linear_sum_assignment (< 1 ms)
    "hungarian" Pure-Python Kuhn-Munkres (zero extra deps)
    "hybrid"    Hungarian primary, CP-SAT escalation on hard cases  ← recommended

The engine uses a task dispatcher that runs as a SimPy process, periodically
checking for pending tasks and assigning them to idle AGVs.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import simpy

from src.warehouse.config import WarehouseConfig
from src.warehouse.graph import WarehouseGraph, NodeType
from src.warehouse.layout import GridLayoutGenerator
from src.simulation.agv import AGV, AGVState, TeleportMovement
from src.simulation.orders import OrderGenerator, Task, TaskStatus, OrderStatus, OrderPriority
from src.simulation.stations import (
    create_pick_station_resources,
    create_charging_station_resources,
    set_station_rng,
)
from src.simulation.metrics import MetricsCollector, SimulationMetrics
from src.assignment.solver import create_solver

# Type alias for the extended strategy literal
AssignmentStrategy = Literal["greedy", "cpsat", "scipy", "hungarian", "hybrid"]


class SimulationEngine:
    """Top-level simulation orchestrator.

    Args:
        config: Full warehouse configuration.
        n_agvs: Number of AGVs to deploy.
        assignment_strategy: Which assignment solver to use.
            "greedy"    — nearest-neighbour baseline (Module 1)
            "cpsat"     — improved CP-SAT solver     (backward-compat default)
            "scipy"     — scipy LAP solver (fastest, < 1 ms)
            "hungarian" — pure-Python Kuhn-Munkres (zero extra deps)
            "hybrid"    — Hungarian + CP-SAT escalation (recommended)
        assignment_fn: Optional custom assignment function (overrides strategy).
            Signature: (pending_tasks, idle_agvs, warehouse) → list[(AGV, Task)]
    """

    def __init__(
        self,
        config: WarehouseConfig,
        n_agvs: int = 50,
        assignment_strategy: AssignmentStrategy = "cpsat",
        assignment_fn: Callable | None = None,
    ) -> None:
        self.config = config
        self.n_agvs = n_agvs
        self.assignment_strategy = assignment_strategy
        self._custom_assignment_fn = assignment_fn

        self.warehouse: WarehouseGraph | None = None
        self.agvs: list[AGV] = []
        self.metrics: SimulationMetrics | None = None

        # Tracks AGVs heading to each station but not yet processing.
        # Updated by the dispatcher; read by optimised solvers for balancing.
        self._inflight_station_counts: dict[str, int] = {}

    def run(self) -> SimulationMetrics:
        """Execute the full simulation and return results."""
        env = simpy.Environment()
        rng = np.random.default_rng(self.config.simulation.random_seed)

        # ── Build warehouse ───────────────────────────────────────────────────
        self.warehouse = GridLayoutGenerator(self.config).generate()
        self.warehouse.precompute_distances()

        # ── Create station resources ──────────────────────────────────────────
        pick_resources = create_pick_station_resources(env, self.warehouse.graph, self.config)
        charging_resources = create_charging_station_resources(env, self.warehouse, self.config)
        set_station_rng(rng)

        # ── Initialise inflight counts (one entry per pick station) ───────────
        for ps_id in self.warehouse.nodes_by_type(NodeType.PICK_STATION):
            self._inflight_station_counts[ps_id] = 0

        # ── Build assignment function ─────────────────────────────────────────
        assignment_fn = self._build_assignment_fn(pick_resources)

        # ── Create AGV fleet ──────────────────────────────────────────────────
        parking_nodes = self.warehouse.nodes_by_type(NodeType.PARKING)
        intersection_nodes = self.warehouse.nodes_by_type(NodeType.INTERSECTION)
        start_nodes = parking_nodes + intersection_nodes
        movement = TeleportMovement(self.warehouse, self.config.agv.speed_mps)

        self.agvs = []
        for i in range(self.n_agvs):
            start_pos = start_nodes[i % len(start_nodes)]
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

        # ── Order generator ───────────────────────────────────────────────────
        order_gen = OrderGenerator(
            env=env,
            config=self.config.orders,
            warehouse=self.warehouse,
            rng=rng,
        )
        env.process(order_gen.run())

        # ── Metrics collector ─────────────────────────────────────────────────
        collector = MetricsCollector(
            env=env,
            agvs=self.agvs,
            order_generator=order_gen,
            pick_station_resources=pick_resources,
            charging_station_resources=charging_resources,
            interval_s=self.config.simulation.metrics_interval_s,
        )
        env.process(collector.run())

        # ── Task dispatcher ───────────────────────────────────────────────────
        env.process(self._dispatcher_process(env, order_gen, assignment_fn))

        # ── Run ───────────────────────────────────────────────────────────────
        duration = self.config.simulation.duration_s
        print(
            f"Starting simulation: {self.n_agvs} AGVs, "
            f"{self.config.stations.n_pick_stations} pick stations, "
            f"{duration / 3600:.1f} hours, "
            f"assignment={self.assignment_strategy}"
        )
        env.run(until=duration)

        # ── Final metrics ─────────────────────────────────────────────────────
        self.metrics = collector.compute_final_metrics()
        print("\nSimulation complete:")
        print(f"  Orders generated: {self.metrics.total_orders_generated}")
        print(f"  Orders completed: {self.metrics.total_orders_completed}")
        print(f"  Avg throughput:   {self.metrics.avg_throughput_per_hour:.0f} orders/hour")
        print(f"  Avg cycle time:   {self.metrics.avg_cycle_time_s:.1f}s")
        print(f"  P95 cycle time:   {self.metrics.p95_cycle_time_s:.1f}s")
        print(f"  AGV utilization:  {self.metrics.avg_agv_utilization_pct:.1f}%")
        print(f"  Station util:     {self.metrics.avg_station_utilization_pct:.1f}%")
        return self.metrics

    # ── Assignment function factory ───────────────────────────────────────────

    def _build_assignment_fn(self, pick_resources: dict[str, simpy.Resource]) -> Callable:
        """Return the callable that the dispatcher will invoke each cycle.

        "greedy"    → static method (no solver object needed)
        all others  → create_solver() returns an instance whose .solve()
                      method is the assignment function

        The returned callable always has signature:
            fn(pending_tasks, idle_agvs, warehouse) → list[(AGV, Task)]
        """
        if self._custom_assignment_fn is not None:
            return self._custom_assignment_fn

        if self.assignment_strategy == "greedy":
            return self._greedy_assignment

        # All optimised solvers go through create_solver()
        solver = create_solver(
            strategy=self.assignment_strategy,
            pick_station_resources=pick_resources,
            inflight_station_counts=self._inflight_station_counts,
        )
        return solver.solve

    # ── Dispatcher SimPy process ──────────────────────────────────────────────

    def _dispatcher_process(
        self,
        env: simpy.Environment,
        order_gen: OrderGenerator,
        assignment_fn: Callable,
    ):
        """Periodically assign pending tasks to idle AGVs."""
        interval = self.config.simulation.dispatch_interval_s
        yield env.timeout(interval)  # small initial delay for orders to accumulate

        while True:
            self._update_inflight_counts()

            # Collect all currently unassigned tasks
            _ = order_gen.get_and_clear_pending()
            pending_tasks = [t for t in order_gen.tasks if t.status == TaskStatus.UNASSIGNED]

            if pending_tasks:
                idle_agvs = [
                    a for a in self.agvs if a.state == AGVState.IDLE and a.current_task is None
                ]
                if idle_agvs:
                    assignments = assignment_fn(pending_tasks, idle_agvs, self.warehouse)
                    for agv, task in assignments:
                        task.status = TaskStatus.ASSIGNED
                        if task.order_ref is not None:
                            task.order_ref.status = OrderStatus.ASSIGNED
                        agv.assign_task(task)

            yield env.timeout(interval)

    def _update_inflight_counts(self) -> None:
        """Count AGVs heading to each station but not yet processing."""
        inflight_states = {
            AGVState.TRAVELING_TO_POD,
            AGVState.PICKING_UP_POD,
            AGVState.TRAVELING_TO_STATION,
        }
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
        """Greedy nearest-neighbour assignment — Module 1 baseline.

        For each task (express first, then FIFO), assign the closest idle AGV.
        """
        sorted_tasks = sorted(
            pending_tasks,
            key=lambda t: (0 if t.priority == OrderPriority.EXPRESS else 1, t.created_time),
        )
        available = {a.id: a for a in idle_agvs}
        assignments: list[tuple[AGV, Task]] = []

        for task in sorted_tasks:
            if not available:
                break
            best_id, best_dist = None, float("inf")
            for agv_id, agv in available.items():
                try:
                    d = warehouse.shortest_path_distance(agv.position, task.pod_location)
                    if d < best_dist:
                        best_dist, best_id = d, agv_id
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
            if best_id is not None:
                assignments.append((available.pop(best_id), task))

        return assignments
