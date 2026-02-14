"""
Benchmark: Greedy vs CP-SAT task assignment.

Generates random assignment scenarios on a real warehouse graph and compares
the greedy nearest-neighbor baseline against the CP-SAT optimizer.

Measures:
- Total trip distance (sum across all assignments)
- Station load imbalance (std dev of station loads)
- Solve time (wall-clock ms)
- Assignment rate (fraction of tasks assigned)

Usage:
    python -m src.assignment.benchmark                     # 50 scenarios
    python -m src.assignment.benchmark --scenarios 200
    python -m src.assignment.benchmark --agvs 30 --tasks 20
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.warehouse.config import (
    WarehouseConfig,
    WarehouseLayoutConfig,
    StationConfig,
    AGVConfig,
)
from src.warehouse.graph import WarehouseGraph, NodeType
from src.warehouse.layout import GridLayoutGenerator
from src.simulation.orders import Task, Order, OrderPriority, OrderStatus, TaskStatus
from src.assignment.cost_matrix import compute_cost_matrix
from src.assignment.solver import (
    CPSATAssignmentSolver,
    SolverStatus,
    AssignmentResult,
)


# ── Mock AGV for standalone benchmarking ──────────────────────────


class MockAGV:
    """Minimal AGV stand-in for benchmarking without SimPy."""

    def __init__(self, agv_id: str, position: str, battery_pct: float, config: AGVConfig):
        self.id = agv_id
        self.position = position
        self.battery_pct = battery_pct
        self.config = config
        self.state = "IDLE"
        self.current_task = None


# ── Scenario generation ───────────────────────────────────────────


@dataclass
class BenchmarkScenario:
    """A single random assignment scenario."""

    agvs: list[MockAGV]
    tasks: list[Task]


def generate_scenario(
    warehouse: WarehouseGraph,
    agv_config: AGVConfig,
    n_agvs: int,
    n_tasks: int,
    rng: np.random.Generator,
    low_battery_fraction: float = 0.15,
    express_fraction: float = 0.1,
) -> BenchmarkScenario:
    """Generate a random assignment scenario.

    Places AGVs at random intersections/parking spots. Creates tasks
    with random pod locations and pick stations. Some AGVs get low
    battery; some tasks are express priority.
    """
    storage_nodes = warehouse.nodes_by_type(NodeType.STORAGE)
    pick_stations = warehouse.nodes_by_type(NodeType.PICK_STATION)
    intersection_nodes = warehouse.nodes_by_type(NodeType.INTERSECTION)
    parking_nodes = warehouse.nodes_by_type(NodeType.PARKING)
    agv_start_nodes = intersection_nodes + parking_nodes

    # Create AGVs at random positions
    agvs = []
    for i in range(n_agvs):
        pos = agv_start_nodes[rng.integers(len(agv_start_nodes))]
        battery = (
            rng.uniform(15.0, 30.0)
            if rng.random() < low_battery_fraction
            else rng.uniform(50.0, 100.0)
        )
        agvs.append(MockAGV(f"AGV_{i:03d}", pos, battery, agv_config))

    # Create tasks
    tasks = []
    for j in range(n_tasks):
        pod_loc = storage_nodes[rng.integers(len(storage_nodes))]
        station = pick_stations[rng.integers(len(pick_stations))]
        priority = (
            OrderPriority.EXPRESS
            if rng.random() < express_fraction
            else OrderPriority.STANDARD
        )
        order = Order(
            id=f"ORD_{j:03d}",
            arrival_time=0,
            n_items=rng.integers(1, 9),
            priority=priority,
        )
        task = Task(
            id=f"TASK_{j:03d}",
            order_id=order.id,
            pod_location=pod_loc,
            pick_station=station,
            return_location=pod_loc,
            order_ref=order,
            priority=priority,
            created_time=0.0,
        )
        tasks.append(task)

    return BenchmarkScenario(agvs=agvs, tasks=tasks)


# ── Greedy solver (standalone, no SimPy) ──────────────────────────


def greedy_solve(
    tasks: list[Task],
    agvs: list[MockAGV],
    warehouse: WarehouseGraph,
) -> AssignmentResult:
    """Greedy nearest-neighbor assignment for benchmarking."""
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (
            0 if t.priority == OrderPriority.EXPRESS else 1,
            t.created_time,
        ),
    )

    available = {a.id: a for a in agvs}
    assignments = []
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

    total_dist = 0.0
    for agv, task in assignments:
        try:
            total_dist += warehouse.shortest_path_distance(
                agv.position, task.pod_location
            ) + warehouse.shortest_path_distance(
                task.pod_location, task.pick_station
            )
        except Exception:
            pass

    return AssignmentResult(
        assignments=assignments,
        unassigned_tasks=unassigned,
        solver_status=SolverStatus.FALLBACK_GREEDY,
        solve_time_ms=0.0,
        total_estimated_distance=total_dist,
    )


# ── Metrics computation ───────────────────────────────────────────


def station_imbalance(assignments: list[tuple], n_stations: int) -> float:
    """Compute std deviation of station loads from assignments."""
    loads = [0] * n_stations
    station_ids = sorted(set(t.pick_station for _, t in assignments)) if assignments else []
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    for _, task in assignments:
        if task.pick_station in sid_to_idx:
            loads[sid_to_idx[task.pick_station]] += 1

    if not loads:
        return 0.0
    return float(np.std(loads))


# ── Main benchmark loop ──────────────────────────────────────────


def run_benchmark(
    n_scenarios: int = 50,
    n_agvs: int = 30,
    n_tasks: int = 20,
    seed: int = 42,
) -> None:
    """Run the full benchmark and print comparison table."""
    print("=" * 70)
    print("  AGV Task Assignment Benchmark: Greedy vs CP-SAT")
    print("=" * 70)
    print(f"  Scenarios: {n_scenarios}  |  AGVs: {n_agvs}  |  Tasks: {n_tasks}")
    print(f"  Seed: {seed}")
    print()

    # Build warehouse
    config = WarehouseConfig(
        warehouse=WarehouseLayoutConfig(
            n_aisles=10,
            bays_per_aisle=20,
            inner_pair_gap_m=2.0,
            pair_spacing_m=7.0,
            rack_offset_m=1.5,
            uturn_interval=5,
        ),
        stations=StationConfig(
            n_pick_stations=4,
            n_charging_stations=4,
            n_parking_spots=12,
        ),
    )
    wh = GridLayoutGenerator(config).generate()
    wh.precompute_distances()
    agv_config = AGVConfig()

    n_stations = len(wh.nodes_by_type(NodeType.PICK_STATION))
    print(f"  Warehouse: {wh.n_nodes} nodes, {wh.n_edges} edges, {n_stations} stations")
    print()

    # Initialize CP-SAT solver (no SimPy resources for standalone benchmark)
    cpsat_solver = CPSATAssignmentSolver()

    rng = np.random.default_rng(seed)

    # Accumulators
    greedy_dists = []
    cpsat_dists = []
    greedy_imbalances = []
    cpsat_imbalances = []
    greedy_assigned = []
    cpsat_assigned = []
    cpsat_times = []
    cpsat_statuses: dict[str, int] = {}

    for i in range(n_scenarios):
        scenario = generate_scenario(wh, agv_config, n_agvs, n_tasks, rng)

        # Greedy
        t0 = time.perf_counter()
        g_result = greedy_solve(scenario.tasks, scenario.agvs, wh)
        g_time = (time.perf_counter() - t0) * 1000

        # CP-SAT
        c_result = cpsat_solver.solve_with_diagnostics(
            scenario.tasks, scenario.agvs, wh
        )

        greedy_dists.append(g_result.total_estimated_distance)
        cpsat_dists.append(c_result.total_estimated_distance)
        greedy_imbalances.append(station_imbalance(g_result.assignments, n_stations))
        cpsat_imbalances.append(station_imbalance(c_result.assignments, n_stations))
        greedy_assigned.append(len(g_result.assignments))
        cpsat_assigned.append(len(c_result.assignments))
        cpsat_times.append(c_result.solve_time_ms)

        status_name = c_result.solver_status.name
        cpsat_statuses[status_name] = cpsat_statuses.get(status_name, 0) + 1

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_scenarios} scenarios...")

    # ── Print results ─────────────────────────────────────────────
    print()
    print("-" * 70)
    print(f"{'Metric':<35} {'Greedy':>15} {'CP-SAT':>15}")
    print("-" * 70)

    def fmt_row(label, g_vals, c_vals, fmt=".1f"):
        g_mean = np.mean(g_vals)
        c_mean = np.mean(c_vals)
        delta = ((c_mean - g_mean) / g_mean * 100) if g_mean != 0 else 0
        sign = "+" if delta > 0 else ""
        print(
            f"  {label:<33} {g_mean:>14{fmt}} {c_mean:>14{fmt}}  ({sign}{delta:.1f}%)"
        )

    fmt_row("Avg total trip distance (m)", greedy_dists, cpsat_dists)
    fmt_row("Avg station imbalance (std)", greedy_imbalances, cpsat_imbalances, ".2f")
    fmt_row("Avg tasks assigned", greedy_assigned, cpsat_assigned, ".1f")

    print(f"  {'Avg solve time (ms)':<33} {'N/A':>15} {np.mean(cpsat_times):>14.1f}")
    print(f"  {'Max solve time (ms)':<33} {'N/A':>15} {np.max(cpsat_times):>14.1f}")
    print(f"  {'P95 solve time (ms)':<33} {'N/A':>15} {np.percentile(cpsat_times, 95):>14.1f}")

    print()
    print("  CP-SAT solver status distribution:")
    for status, count in sorted(cpsat_statuses.items()):
        print(f"    {status:<20} {count:>5} ({count / n_scenarios * 100:.0f}%)")

    # Distance improvement distribution
    improvements = [
        (g - c) / g * 100 if g > 0 else 0
        for g, c in zip(greedy_dists, cpsat_dists)
    ]
    print()
    print(f"  Distance improvement (CP-SAT vs Greedy):")
    print(f"    Mean:   {np.mean(improvements):.1f}%")
    print(f"    Median: {np.median(improvements):.1f}%")
    print(f"    Min:    {np.min(improvements):.1f}%")
    print(f"    Max:    {np.max(improvements):.1f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark greedy vs CP-SAT assignment")
    parser.add_argument("--scenarios", type=int, default=50, help="Number of random scenarios")
    parser.add_argument("--agvs", type=int, default=30, help="AGVs per scenario")
    parser.add_argument("--tasks", type=int, default=20, help="Tasks per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_benchmark(args.scenarios, args.agvs, args.tasks, args.seed)
