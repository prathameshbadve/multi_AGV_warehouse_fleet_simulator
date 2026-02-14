"""
run_simulation.py
──────────────────────────────────────────────────────────────────────────────
Quick-run script for the AGV fleet simulator.

Usage:
    python run_simulation.py                              # defaults: 50 AGVs, cpsat
    python run_simulation.py --n-agvs 80 --hours 2
    python run_simulation.py --strategy scipy             # fastest solver
    python run_simulation.py --strategy hybrid            # recommended for production
    python run_simulation.py --strategy greedy            # Module 1 baseline
    python run_simulation.py --config config/default_warehouse.yaml

Assignment strategy options:
    greedy    Nearest-neighbour baseline (Module 1)         no dependencies
    cpsat     Improved CP-SAT: k-nearest + warm hint        requires or-tools
    scipy     scipy.optimize.linear_sum_assignment < 1 ms   requires scipy
    hungarian Pure-Python Kuhn-Munkres, zero extra deps     ~5 ms
    hybrid    Hungarian primary + CP-SAT escalation         recommended
"""

import argparse
from pathlib import Path

from src.warehouse.config import load_config, WarehouseConfig, SimulationConfig
from src.simulation.engine import SimulationEngine


def main():
    """Main"""

    parser = argparse.ArgumentParser(description="Run AGV fleet simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_warehouse.yaml",
        help="Path to warehouse config YAML",
    )
    parser.add_argument("--n-agvs", type=int, default=50, help="Number of AGVs to deploy")
    parser.add_argument(
        "--hours", type=float, default=None, help="Simulation duration in hours (overrides config)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="cpsat",
        choices=["greedy", "cpsat", "scipy", "hungarian", "hybrid"],
        help=(
            "Assignment strategy:\n"
            "  greedy    nearest-neighbour baseline\n"
            "  cpsat     improved CP-SAT (k-nearest + warm hint) [default]\n"
            "  scipy     scipy LAP solver (fastest, < 1 ms)\n"
            "  hungarian pure-Python Kuhn-Munkres (zero extra deps)\n"
            "  hybrid    Hungarian + CP-SAT escalation (recommended)"
        ),
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config {config_path} not found, using defaults")
        config = WarehouseConfig()

    # Apply CLI overrides
    if args.hours is not None or args.seed is not None:
        sim_kwargs = {
            "duration_hours": args.hours or config.simulation.duration_hours,
            "dispatch_interval_s": config.simulation.dispatch_interval_s,
            "metrics_interval_s": config.simulation.metrics_interval_s,
            "random_seed": args.seed if args.seed is not None else config.simulation.random_seed,
        }
        config = WarehouseConfig(
            warehouse=config.warehouse,
            stations=config.stations,
            agv=config.agv,
            orders=config.orders,
            pick_station=config.pick_station,
            simulation=SimulationConfig(**sim_kwargs),
        )

    # Run
    engine = SimulationEngine(config, n_agvs=args.n_agvs, assignment_strategy=args.strategy)
    results = engine.run()

    # Detailed AGV fleet summary
    print(f"\n{'=' * 60}")
    print("AGV Fleet Summary:")
    print(f"{'=' * 60}")
    print(f"{'AGV ID':<10} {'Tasks':>6} {'Dist(m)':>8} {'Util%':>6} {'Battery':>8}")
    print(f"{'-' * 10} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 8}")
    for agv_id, summary in sorted(results.agv_summaries.items()):
        print(
            f"{agv_id:<10} {summary['tasks_completed']:>6} "
            f"{summary['distance_m']:>8.0f} {summary['utilization_pct']:>5.1f}% "
            f"{summary['final_battery_pct']:>7.1f}%"
        )


if __name__ == "__main__":
    main()
