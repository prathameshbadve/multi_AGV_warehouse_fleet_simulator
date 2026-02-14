"""
Task assignment solver using Google OR-Tools CP-SAT.

Solves the AGV-to-task assignment problem as a constrained binary optimization:
- Minimize total travel distance (AGV → pod → station)
- Respect battery feasibility (hard pre-filter)
- Balance pick station load (soft penalty in objective)
- Prioritize express orders (hard constraint: must assign if possible)
- Prefer making assignments over leaving tasks idle (soft bonus)

Falls back to greedy nearest-neighbor if CP-SAT is unavailable or times out.

Usage:
    from src.assignment.solver import CPSATAssignmentSolver

    solver = CPSATAssignmentSolver(pick_station_resources)
    assignments = solver.solve(pending_tasks, idle_agvs, warehouse)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import simpy
from ortools.sat.python import cp_model

if TYPE_CHECKING:
    from src.simulation.agv import AGV
    from src.simulation.orders import Task
    from src.warehouse.graph import WarehouseGraph

from src.assignment.cost_matrix import CostMatrix, compute_cost_matrix


# ── Result dataclass ──────────────────────────────────────────────


class SolverStatus(Enum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    FALLBACK_GREEDY = auto()


@dataclass
class AssignmentResult:
    """Output of the assignment solver.

    Attributes:
        assignments: List of (AGV, Task) pairs to execute.
        unassigned_tasks: Tasks that could not be assigned this cycle.
        solver_status: How the solution was obtained.
        solve_time_ms: Wall-clock solve time in milliseconds.
        total_estimated_distance: Sum of trip costs for all assignments (meters).
        battery_rejections: Number of (AGV, Task) pairs pruned by battery check.
    """

    assignments: list[tuple[AGV, Task]]
    unassigned_tasks: list[Task]
    solver_status: SolverStatus
    solve_time_ms: float
    total_estimated_distance: float = 0.0
    battery_rejections: int = 0


# ── Solver configuration ──────────────────────────────────────────


@dataclass(frozen=True)
class SolverConfig:
    """Tunable parameters for the CP-SAT solver.

    Attributes:
        time_limit_ms: Maximum solve time before fallback.
        priority_bonus: Cost bonus (negative) for express tasks.
            Must be large enough to dominate over distance differences.
        balance_weight: Penalty weight on station load imbalance.
        assign_bonus: Small bonus per assignment to prefer assigning
            over leaving tasks idle.
    """

    time_limit_ms: int = 1000
    priority_bonus: int = 10_000
    balance_weight: int = 500
    assign_bonus: int = 100


# ── Main solver class ─────────────────────────────────────────────


class CPSATAssignmentSolver:
    """CP-SAT-based task assignment with greedy fallback.

    Instantiate once with pick station resources (for queue depth reading),
    then call `solve()` every dispatch cycle.

    Args:
        pick_station_resources: Dict mapping station node ID → SimPy Resource.
            Used to read current queue depth for station balancing.
        solver_config: Tunable solver parameters.
        inflight_station_counts: Optional dict tracking AGVs en route to each
            station but not yet arrived. Updated externally by the dispatcher.
    """

    def __init__(
        self,
        pick_station_resources: dict[str, simpy.Resource] | None = None,
        solver_config: SolverConfig | None = None,
        inflight_station_counts: dict[str, int] | None = None,
    ) -> None:
        self.pick_resources = pick_station_resources or {}
        self.config = solver_config or SolverConfig()
        self.inflight_counts = inflight_station_counts or {}

        # Track solver statistics across cycles
        self.total_solves: int = 0
        self.total_fallbacks: int = 0
        self.total_solve_time_ms: float = 0.0

    def solve(
        self,
        pending_tasks: list[Task],
        idle_agvs: list[AGV],
        warehouse: WarehouseGraph,
    ) -> list[tuple[AGV, Task]]:
        """Solve the assignment problem and return (AGV, Task) pairs.

        This method has the same signature as the greedy baseline in
        engine.py, making it a drop-in replacement via assignment_fn.

        Args:
            pending_tasks: Tasks awaiting AGV assignment.
            idle_agvs: AGVs currently idle and available.
            warehouse: Warehouse graph with precomputed distances.

        Returns:
            List of (AGV, Task) assignment pairs.
        """
        if not pending_tasks or not idle_agvs:
            return []

        result = self._solve_cpsat(pending_tasks, idle_agvs, warehouse)

        # Update running stats
        self.total_solves += 1
        self.total_solve_time_ms += result.solve_time_ms
        if result.solver_status == SolverStatus.FALLBACK_GREEDY:
            self.total_fallbacks += 1

        return result.assignments

    def solve_with_diagnostics(
        self,
        pending_tasks: list[Task],
        idle_agvs: list[AGV],
        warehouse: WarehouseGraph,
    ) -> AssignmentResult:
        """Like solve(), but returns the full AssignmentResult with diagnostics."""
        if not pending_tasks or not idle_agvs:
            return AssignmentResult(
                assignments=[],
                unassigned_tasks=list(pending_tasks),
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=0.0,
            )

        result = self._solve_cpsat(pending_tasks, idle_agvs, warehouse)
        self.total_solves += 1
        self.total_solve_time_ms += result.solve_time_ms
        if result.solver_status == SolverStatus.FALLBACK_GREEDY:
            self.total_fallbacks += 1
        return result

    # ── CP-SAT implementation ─────────────────────────────────────

    def _solve_cpsat(
        self,
        tasks: list[Task],
        agvs: list[AGV],
        warehouse: WarehouseGraph,
    ) -> AssignmentResult:
        """Build and solve the CP-SAT model. Falls back to greedy on failure."""
        from src.warehouse.config import AGVConfig

        # Infer AGV config from the first AGV (all share the same config)
        agv_config = agvs[0].config

        # Step 1: Build cost matrix
        t0 = time.perf_counter()
        matrix = compute_cost_matrix(tasks, agvs, warehouse, agv_config)

        # Count battery rejections
        battery_rejections = sum(
            1
            for a in range(matrix.n_agvs)
            for t in range(matrix.n_tasks)
            if not matrix.feasible[a][t]
        )

        # Step 2: Try CP-SAT
        try:
            model = cp_model.CpModel()
        except ImportError:
            # OR-Tools not installed — fall back to greedy
            result = self._greedy_fallback(tasks, agvs, warehouse)
            result.battery_rejections = battery_rejections
            result.solve_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # Step 3: Create decision variables (only for feasible pairs)
        x: dict[tuple[int, int], cp_model.IntVar] = {}
        for a in range(matrix.n_agvs):
            for t in range(matrix.n_tasks):
                if matrix.feasible[a][t]:
                    x[a, t] = model.new_bool_var(f"x_{a}_{t}")

        if not x:
            # No feasible assignment exists
            return AssignmentResult(
                assignments=[],
                unassigned_tasks=list(tasks),
                solver_status=SolverStatus.INFEASIBLE,
                solve_time_ms=(time.perf_counter() - t0) * 1000,
                battery_rejections=battery_rejections,
            )

        # Step 4: Constraints

        # C1: Each task assigned to at most one AGV
        for t in range(matrix.n_tasks):
            task_vars = [x[a, t] for a in range(matrix.n_agvs) if (a, t) in x]
            if task_vars:
                model.add(sum(task_vars) <= 1)

        # C2: Each AGV assigned to at most one task
        for a in range(matrix.n_agvs):
            agv_vars = [x[a, t] for t in range(matrix.n_tasks) if (a, t) in x]
            if agv_vars:
                model.add(sum(agv_vars) <= 1)

        # C5: Express priority — must assign if any AGV is feasible
        from src.simulation.orders import OrderPriority

        for t_idx, task in enumerate(tasks):
            if task.priority == OrderPriority.EXPRESS:
                express_vars = [x[a, t_idx] for a in range(matrix.n_agvs) if (a, t_idx) in x]
                if express_vars:
                    model.add(sum(express_vars) == 1)

        # Step 5: Objective

        # 5a: Travel cost
        cost_terms = []
        for (a, t), var in x.items():
            cost_terms.append(var * matrix.trip_cost[a][t])

        # 5b: Express priority bonus (negative cost = incentive)
        priority_terms = []
        for (a, t), var in x.items():
            if tasks[t].priority == OrderPriority.EXPRESS:
                priority_terms.append(var * (-self.config.priority_bonus))

        # 5c: Assignment bonus (prefer assigning over leaving idle)
        assign_terms = []
        for var in x.values():
            assign_terms.append(var * (-self.config.assign_bonus))

        # 5d: Station load balancing
        balance_terms = []
        if len(matrix.station_ids) > 1:
            balance_terms = self._add_station_balance(model, x, matrix, tasks)

        all_terms = cost_terms + priority_terms + assign_terms + balance_terms
        model.minimize(sum(all_terms))

        # Step 6: Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit_ms / 1000.0

        status = solver.solve(model)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Step 7: Extract solution
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solver_status = (
                SolverStatus.OPTIMAL if status == cp_model.OPTIMAL else SolverStatus.FEASIBLE
            )

            assignments = []
            assigned_task_indices: set[int] = set()
            total_dist = 0.0

            for (a, t), var in x.items():
                if solver.value(var) == 1:
                    assignments.append((agvs[a], tasks[t]))
                    assigned_task_indices.add(t)
                    total_dist += matrix.trip_cost[a][t] / 100.0  # un-scale

            unassigned = [tasks[t] for t in range(matrix.n_tasks) if t not in assigned_task_indices]

            return AssignmentResult(
                assignments=assignments,
                unassigned_tasks=unassigned,
                solver_status=solver_status,
                solve_time_ms=elapsed_ms,
                total_estimated_distance=total_dist,
                battery_rejections=battery_rejections,
            )
        else:
            # Solver failed — fallback to greedy
            result = self._greedy_fallback(tasks, agvs, warehouse)
            result.battery_rejections = battery_rejections
            result.solve_time_ms = elapsed_ms
            return result

    # ── Station balancing ─────────────────────────────────────────

    def _add_station_balance(
        self,
        model,
        x: dict[tuple[int, int], any],
        matrix: CostMatrix,
        tasks: list[Task],
    ) -> list:
        """Add station load balancing as a linearized min-max penalty.

        Returns objective terms to include in the minimize() call.
        """
        from ortools.sat.python import cp_model

        n_stations = len(matrix.station_ids)

        # Current queue depth at each station
        current_queue = []
        for sid in matrix.station_ids:
            queue_len = 0
            if sid in self.pick_resources:
                res = self.pick_resources[sid]
                queue_len = res.count + len(res.queue)
            # Add in-flight AGVs heading to this station
            queue_len += self.inflight_counts.get(sid, 0)
            current_queue.append(queue_len)

        # station_load[s] = current_queue[s] + sum of x[a,t] where station[t]==s
        station_load_vars = []
        max_possible = matrix.n_agvs + max(current_queue, default=0) + 1

        for s_idx in range(n_stations):
            load_var = model.new_int_var(0, max_possible, f"load_{s_idx}")
            assigned_to_s = [x[a, t] for (a, t) in x if matrix.station_index[t] == s_idx]
            if assigned_to_s:
                model.add(load_var == current_queue[s_idx] + sum(assigned_to_s))
            else:
                model.add(load_var == current_queue[s_idx])
            station_load_vars.append(load_var)

        # Linearize: max_load >= all loads, min_load <= all loads
        max_load = model.new_int_var(0, max_possible, "max_load")
        min_load = model.new_int_var(0, max_possible, "min_load")

        for load_var in station_load_vars:
            model.add(max_load >= load_var)
            model.add(min_load <= load_var)

        # Imbalance = max_load - min_load, penalized in objective
        imbalance = model.new_int_var(0, max_possible, "imbalance")
        model.add(imbalance == max_load - min_load)

        return [imbalance * self.config.balance_weight]

    # ── Greedy fallback ───────────────────────────────────────────

    def _greedy_fallback(
        self,
        tasks: list[Task],
        agvs: list[AGV],
        warehouse: WarehouseGraph,
    ) -> AssignmentResult:
        """Greedy nearest-neighbor assignment (Module 1 baseline).

        Used as fallback when CP-SAT is unavailable or times out.
        """
        from src.simulation.orders import OrderPriority

        sorted_tasks = sorted(
            tasks,
            key=lambda t: (
                0 if t.priority == OrderPriority.EXPRESS else 1,
                t.created_time,
            ),
        )

        available = {a.id: a for a in agvs}
        assignments: list[tuple[AGV, Task]] = []
        assigned_task_ids: set[str] = set()

        for task in sorted_tasks:
            if not available:
                break

            best_id = None
            best_dist = float("inf")

            for agv_id, agv in available.items():
                try:
                    d = warehouse.shortest_path_distance(agv.position, task.pod_location)
                    if d < best_dist:
                        best_dist = d
                        best_id = agv_id
                except Exception:
                    continue

            if best_id is not None:
                assignments.append((available[best_id], task))
                assigned_task_ids.add(task.id)
                del available[best_id]

        unassigned = [t for t in tasks if t.id not in assigned_task_ids]

        return AssignmentResult(
            assignments=assignments,
            unassigned_tasks=unassigned,
            solver_status=SolverStatus.FALLBACK_GREEDY,
            solve_time_ms=0.0,
            total_estimated_distance=sum(
                warehouse.shortest_path_distance(a.position, t.pod_location) for a, t in assignments
            ),
        )
