"""
src/assignment/benchmark.py
──────────────────────────────────────────────────────────────────────────────
Benchmark: five assignment strategies head-to-head.

Compares the greedy nearest-neighbour baseline against the four optimised
solvers on random assignment scenarios drawn from a real warehouse graph.

Metrics per scenario:
  • Total trip distance   (pickup_dist + deliver_dist, metres)
  • Station load imbalance (std-dev across pick stations)
  • Solve time            (wall-clock, ms)
  • Assignment rate       (tasks assigned / tasks available)
  • Battery rejections    (pairs pruned by battery pre-filter)

HybridSolver additionally tracks its escalation rate.

Usage:
    python -m src.assignment.benchmark                    # 50 scenarios, defaults
    python -m src.assignment.benchmark --scenarios 200
    python -m src.assignment.benchmark --agvs 50 --tasks 30
    python -m src.assignment.benchmark --solvers greedy scipy hungarian
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from src.warehouse.config import (
    WarehouseConfig,
    WarehouseLayoutConfig,
    StationConfig,
    AGVConfig,
)
from src.warehouse.graph import WarehouseGraph, NodeType
from src.warehouse.layout import GridLayoutGenerator
from src.simulation.orders import Task, Order, OrderPriority
from src.assignment.solver import (
    HybridSolver,
    SolverStatus,
    AssignmentResult,
    create_solver,
)


# ── Mock AGV (no SimPy required for standalone benchmarking) ──────────────────


class MockAGV:
    """Minimal AGV stand-in. Mirrors the real AGV public interface used by solvers."""

    def __init__(self, agv_id: str, position: str, battery_pct: float, config: AGVConfig):
        self.id = agv_id
        self.position = position
        self.battery_pct = battery_pct
        self.config = config
        self.state = "IDLE"
        self.current_task = None


# ── Scenario generation ───────────────────────────────────────────────────────


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
    express_fraction: float = 0.10,
) -> BenchmarkScenario:
    """Generate a random dispatch scenario.

    Spawns n_agvs at random intersection/parking nodes with realistic battery
    levels (15 % get low battery to stress-test infeasibility handling).
    Creates n_tasks with random pod locations and pick stations.
    """
    storage_nodes = warehouse.nodes_by_type(NodeType.STORAGE)
    pick_stations = warehouse.nodes_by_type(NodeType.PICK_STATION)
    intersection_nodes = warehouse.nodes_by_type(NodeType.INTERSECTION)
    parking_nodes = warehouse.nodes_by_type(NodeType.PARKING)
    start_nodes = intersection_nodes + parking_nodes

    agvs = []
    for i in range(n_agvs):
        pos = start_nodes[rng.integers(len(start_nodes))]
        battery = (
            rng.uniform(15.0, 30.0)
            if rng.random() < low_battery_fraction
            else rng.uniform(50.0, 100.0)
        )
        agvs.append(MockAGV(f"AGV_{i:03d}", pos, battery, agv_config))

    tasks = []
    for j in range(n_tasks):
        pod_loc = storage_nodes[rng.integers(len(storage_nodes))]
        station = pick_stations[rng.integers(len(pick_stations))]
        priority = (
            OrderPriority.EXPRESS if rng.random() < express_fraction else OrderPriority.STANDARD
        )
        order = Order(
            id=f"ORD_{j:03d}",
            arrival_time=0,
            n_items=int(rng.integers(1, 9)),
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


# ── Greedy solver (standalone, no SimPy) ─────────────────────────────────────


def greedy_solve(
    tasks: list[Task],
    agvs: list[MockAGV],
    warehouse: WarehouseGraph,
) -> AssignmentResult:
    """Nearest-neighbour greedy. Express tasks first. Benchmark baseline."""
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (0 if t.priority == OrderPriority.EXPRESS else 1, t.created_time),
    )
    available = {a.id: a for a in agvs}
    assignments = []
    assigned_ids: set[str] = set()

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
            assigned_ids.add(task.id)

    total_dist = 0.0
    for agv, task in assignments:
        try:
            total_dist += warehouse.shortest_path_distance(
                agv.position, task.pod_location
            ) + warehouse.shortest_path_distance(task.pod_location, task.pick_station)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    return AssignmentResult(
        assignments=assignments,
        unassigned_tasks=[t for t in tasks if t.id not in assigned_ids],
        solver_status=SolverStatus.FALLBACK_GREEDY,
        solve_time_ms=0.0,
        total_estimated_distance=total_dist,
    )


# ── Metric helpers ────────────────────────────────────────────────────────────


def station_imbalance(assignments: list[tuple], n_stations: int) -> float:
    """Std-dev of station loads across all assigned tasks."""
    if not assignments:
        return 0.0
    station_ids = sorted({t.pick_station for _, t in assignments})
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    loads = [0] * max(n_stations, len(station_ids))
    for _, task in assignments:
        loads[sid_to_idx[task.pick_station]] += 1
    return float(np.std(loads))


# ── Main benchmark loop ───────────────────────────────────────────────────────


def run_benchmark(
    n_scenarios: int = 50,
    n_agvs: int = 30,
    n_tasks: int = 20,
    seed: int = 42,
    solver_names: list[str] | None = None,
) -> None:
    """Run scenarios and print a comparison table."""

    all_solvers = ["greedy", "scipy", "hungarian", "cpsat", "hybrid"]
    active = solver_names or all_solvers

    print("=" * 80)
    print("  AGV Task Assignment Benchmark")
    print("=" * 80)
    print(f"  Scenarios: {n_scenarios}  |  AGVs: {n_agvs}  |  Tasks: {n_tasks}  |  Seed: {seed}")
    print(f"  Solvers:   {', '.join(active)}")
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
    print(f"  Warehouse: {wh.n_nodes} nodes, {wh.n_edges} edges, {n_stations} stations\n")

    # Instantiate solvers (greedy is a standalone function)
    solver_instances: dict[str, object] = {}
    for name in active:
        if name != "greedy":
            solver_instances[name] = create_solver(name)

    rng = np.random.default_rng(seed)

    # Per-solver accumulators
    results: dict[str, dict[str, list]] = {
        name: {"dist": [], "imbal": [], "rate": [], "time_ms": [], "status": []} for name in active
    }

    for _ in range(n_scenarios):
        scenario = generate_scenario(wh, agv_config, n_agvs, n_tasks, rng)

        for name in active:
            t0 = time.perf_counter()
            if name == "greedy":
                r = greedy_solve(scenario.tasks, scenario.agvs, wh)
                r.solve_time_ms = (time.perf_counter() - t0) * 1e3
            else:
                solver = solver_instances[name]
                r = solver.solve_with_diagnostics(scenario.tasks, scenario.agvs, wh)

            dist = r.total_estimated_distance
            imbal = station_imbalance(r.assignments, n_stations)
            rate = len(r.assignments) / max(min(n_agvs, n_tasks), 1)

            results[name]["dist"].append(dist)
            results[name]["imbal"].append(imbal)
            results[name]["rate"].append(rate * 100)
            results[name]["time_ms"].append(r.solve_time_ms)
            results[name]["status"].append(r.solver_status.name)

    # ── Print results ─────────────────────────────────────────────────────────
    col_w = 16

    def hdr(label: str) -> str:
        return f"{label:>{col_w}}"

    def val(v: float, fmt: str = ".1f") -> str:
        return f"{v:{col_w}{fmt}}"

    # Header row
    print(f"  {'Metric':<30}" + "".join(hdr(n) for n in active))
    print("  " + "─" * (30 + col_w * len(active)))

    metrics_cfg = [
        ("Avg trip distance (m)", "dist", ".1f"),
        ("Avg station imbalance", "imbal", ".2f"),
        ("Avg assignment rate (%)", "rate", ".1f"),
        ("Avg solve time (ms)", "time_ms", ".2f"),
        ("P95 solve time (ms)", "time_ms", ".2f"),
        ("Max solve time (ms)", "time_ms", ".2f"),
    ]

    fn_map = {
        "Avg trip distance (m)": lambda d: np.mean(d["dist"]),
        "Avg station imbalance": lambda d: np.mean(d["imbal"]),
        "Avg assignment rate (%)": lambda d: np.mean(d["rate"]),
        "Avg solve time (ms)": lambda d: np.mean(d["time_ms"]),
        "P95 solve time (ms)": lambda d: np.percentile(d["time_ms"], 95),
        "Max solve time (ms)": lambda d: np.max(d["time_ms"]),
    }

    for label, _, fmt in metrics_cfg:
        row = f"  {label:<30}"
        for name in active:
            row += val(fn_map[label](results[name]), fmt)
        print(row)

    # Distance improvement vs greedy
    if "greedy" in active:
        print()
        print(f"  {'Distance improvement vs greedy':<30}", end="")
        g_dists = results["greedy"]["dist"]
        for name in active:
            if name == "greedy":
                print(f"{'baseline':>{col_w}}", end="")
            else:
                improvements = [
                    (g - c) / g * 100 if g > 0 else 0.0
                    for g, c in zip(g_dists, results[name]["dist"])
                ]
                print(val(np.mean(improvements), ".1f") + "%", end="")
        print()

    # HybridSolver escalation rate
    if "hybrid" in active and isinstance(solver_instances.get("hybrid"), HybridSolver):
        h = solver_instances["hybrid"]
        print(f"\n  Hybrid escalation rate: {h.escalation_rate * 100:.1f}%")

    # Solver status distribution
    print()
    print("  Solver status distribution:")
    for name in active:
        counts: dict[str, int] = {}
        for s in results[name]["status"]:
            counts[s] = counts.get(s, 0) + 1
        dist_str = "  ".join(f"{s}={c}" for s, c in sorted(counts.items()))
        print(f"    {name:<12}: {dist_str}")

    print("\n" + "=" * 80)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark AGV assignment strategies")
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--agvs", type=int, default=30)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["greedy", "scipy", "hungarian", "cpsat", "hybrid"],
        default=None,
        help="Subset of solvers to benchmark (default: all five)",
    )
    args = parser.parse_args()
    run_benchmark(args.scenarios, args.agvs, args.tasks, args.seed, args.solvers)
