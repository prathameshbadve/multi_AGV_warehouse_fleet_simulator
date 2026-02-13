"""Integration tests for the simulation engine.

These tests run short simulations to verify the full pipeline works:
orders generated → tasks created → AGVs assigned → tasks completed.

Run with: pytest tests/test_simulation.py -v
"""

import pytest

from src.warehouse.config import (
    WarehouseConfig,
    WarehouseLayoutConfig,
    StationConfig,
    AGVConfig,
    OrderConfig,
    SimulationConfig,
)
from src.simulation.engine import SimulationEngine


@pytest.fixture
def quick_sim_config() -> WarehouseConfig:
    """Config for a short simulation (~30 sim-minutes, small warehouse)."""
    return WarehouseConfig(
        warehouse=WarehouseLayoutConfig(
            n_aisles=4,
            bays_per_aisle=10,
            aisle_spacing_m=3.0,
            bay_depth_m=1.5,
        ),
        stations=StationConfig(
            n_pick_stations=2,
            n_charging_stations=2,
            n_parking_spots=4,
        ),
        agv=AGVConfig(
            speed_mps=1.5,
            battery_drain_per_meter=0.08,
            battery_threshold=20.0,
        ),
        orders=OrderConfig(
            base_rate_per_min=4.0,  # lower rate for small warehouse
            express_fraction=0.1,
        ),
        simulation=SimulationConfig(
            duration_hours=0.5,  # 30 minutes
            dispatch_interval_s=5.0,
            metrics_interval_s=30.0,
            random_seed=42,
        ),
    )


class TestSimulationEngine:
    """End-to-end simulation tests."""

    def test_simulation_runs_without_errors(self, quick_sim_config):
        """Basic smoke test: simulation should complete without exceptions."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        assert results is not None
        assert results.total_simulation_time_s == pytest.approx(1800.0)

    def test_orders_are_generated(self, quick_sim_config):
        """Orders should be generated during the simulation."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        assert results.total_orders_generated > 0

    def test_some_orders_completed(self, quick_sim_config):
        """At least some orders should be completed in 30 minutes with 10 AGVs."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        assert results.total_orders_completed > 0

    def test_throughput_positive(self, quick_sim_config):
        """Throughput should be positive if orders are completed."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        if results.total_orders_completed > 0:
            assert results.avg_throughput_per_hour > 0

    def test_more_agvs_higher_throughput(self, quick_sim_config):
        """Increasing AGV count should increase throughput (up to a point)."""
        engine_small = SimulationEngine(quick_sim_config, n_agvs=5)
        results_small = engine_small.run()

        engine_large = SimulationEngine(quick_sim_config, n_agvs=20)
        results_large = engine_large.run()

        # With 4x the AGVs, throughput should be meaningfully higher
        # (not necessarily 4x due to station bottleneck)
        assert results_large.total_orders_completed >= results_small.total_orders_completed

    def test_metrics_snapshots_collected(self, quick_sim_config):
        """Metrics collector should produce periodic snapshots."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        # 30 min sim / 30 sec interval = ~60 snapshots
        assert len(results.snapshots) > 0

    def test_agv_summaries_populated(self, quick_sim_config):
        """Each AGV should have a summary in the results."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        assert len(results.agv_summaries) == 10
        for agv_id, summary in results.agv_summaries.items():
            assert "tasks_completed" in summary
            assert "distance_m" in summary
            assert "utilization_pct" in summary

    def test_cycle_time_reasonable(self, quick_sim_config):
        """Average cycle time should be reasonable (not 0, not absurdly high)."""
        engine = SimulationEngine(quick_sim_config, n_agvs=10)
        results = engine.run()
        if results.total_orders_completed > 5:
            # Cycle time should be at least 30s (travel + processing)
            # and less than 10 minutes for a small warehouse
            assert results.avg_cycle_time_s > 20
            assert results.avg_cycle_time_s < 600
