"""
Metrics collection and KPI computation.

Runs as a SimPy process that snapshots system state at regular intervals.
After the simulation, produces a structured SimulationMetrics object with
time-series data and summary statistics.

KPIs tracked:
- Throughput (orders completed per hour)
- Order cycle time (arrival to completion)
- AGV utilization (% time actively working)
- Pick station utilization (% time processing)
- Queue lengths at pick stations and charging stations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import simpy

if TYPE_CHECKING:
    from src.simulation.agv import AGV
    from src.simulation.orders import OrderGenerator


@dataclass
class Snapshot:
    """A point-in-time snapshot of system state."""

    time_s: float
    orders_completed: int
    orders_pending: int
    throughput_per_hour: float  # based on completed orders so far
    avg_cycle_time_s: float  # of completed orders
    agv_utilization_pct: float  # avg across fleet
    n_agvs_idle: int
    n_agvs_traveling: int
    n_agvs_charging: int
    n_agvs_waiting: int
    station_queue_lengths: dict[str, int]  # station_id → queue length


@dataclass
class SimulationMetrics:
    """Final simulation results with time-series and summary stats."""

    # Time-series
    snapshots: list[Snapshot] = field(default_factory=list)

    # Summary stats (computed at end)
    total_orders_generated: int = 0
    total_orders_completed: int = 0
    total_simulation_time_s: float = 0.0
    avg_throughput_per_hour: float = 0.0
    avg_cycle_time_s: float = 0.0
    median_cycle_time_s: float = 0.0
    p95_cycle_time_s: float = 0.0
    avg_agv_utilization_pct: float = 0.0
    avg_station_utilization_pct: float = 0.0
    total_distance_traveled_m: float = 0.0

    # Per-AGV summaries
    agv_summaries: dict[str, dict] = field(default_factory=dict)


class MetricsCollector:
    """SimPy process that periodically snapshots the system and computes final KPIs.

    Usage:
        collector = MetricsCollector(env, agvs, order_gen, stations, interval=60)
        env.process(collector.run())
        # ... run simulation ...
        results = collector.compute_final_metrics()
    """

    def __init__(
        self,
        env: simpy.Environment,
        agvs: list[AGV],
        order_generator: OrderGenerator,
        pick_station_resources: dict[str, simpy.Resource],
        charging_station_resources: dict[str, simpy.Resource],
        interval_s: float = 60.0,
    ) -> None:
        self.env = env
        self.agvs = agvs
        self.order_generator = order_generator
        self.pick_stations = pick_station_resources
        self.charging_stations = charging_station_resources
        self.interval_s = interval_s

        self._snapshots: list[Snapshot] = []

    def run(self):
        """SimPy generator: take snapshots at regular intervals."""
        while True:
            yield self.env.timeout(self.interval_s)
            snapshot = self._take_snapshot()
            self._snapshots.append(snapshot)

    def _take_snapshot(self) -> Snapshot:
        """Capture current system state."""

        from src.simulation.agv import AGVState
        from src.simulation.orders import OrderStatus

        now = self.env.now

        # Count completed orders
        completed_orders = [
            o for o in self.order_generator.orders if o.status == OrderStatus.COMPLETE
        ]
        pending_orders = [o for o in self.order_generator.orders if o.status == OrderStatus.PENDING]

        n_completed = len(completed_orders)
        n_pending = len(pending_orders)

        # Throughput: orders completed so far / elapsed hours
        elapsed_hours = now / 3600.0
        throughput = n_completed / elapsed_hours if elapsed_hours > 0 else 0.0

        # Average cycle time of completed orders
        cycle_times = [o.cycle_time for o in completed_orders if o.cycle_time is not None]
        avg_cycle = float(np.mean(cycle_times)) if cycle_times else 0.0

        # AGV states
        idle_states = {AGVState.IDLE}
        travel_states = {
            AGVState.TRAVELING_TO_POD,
            AGVState.TRAVELING_TO_STATION,
            AGVState.RETURNING_POD,
        }
        charging_states = {AGVState.TRAVELING_TO_CHARGER, AGVState.CHARGING}
        waiting_states = {AGVState.WAITING_AT_STATION}

        n_idle = sum(1 for a in self.agvs if a.state in idle_states)
        n_traveling = sum(1 for a in self.agvs if a.state in travel_states)
        n_charging = sum(1 for a in self.agvs if a.state in charging_states)
        n_waiting = sum(1 for a in self.agvs if a.state in waiting_states)

        # AGV utilization: % of AGVs not idle and not charging
        n_active = len(self.agvs) - n_idle - n_charging
        utilization = (n_active / len(self.agvs) * 100) if self.agvs else 0.0

        # Station queue lengths
        station_queues = {}
        for station_id, resource in self.pick_stations.items():
            station_queues[station_id] = len(resource.queue) + len(resource.users)

        return Snapshot(
            time_s=now,
            orders_completed=n_completed,
            orders_pending=n_pending,
            throughput_per_hour=throughput,
            avg_cycle_time_s=avg_cycle,
            agv_utilization_pct=utilization,
            n_agvs_idle=n_idle,
            n_agvs_traveling=n_traveling,
            n_agvs_charging=n_charging,
            n_agvs_waiting=n_waiting,
            station_queue_lengths=station_queues,
        )

    def compute_final_metrics(self) -> SimulationMetrics:
        """Compute summary statistics after the simulation completes.

        Returns:
            SimulationMetrics with both time-series snapshots and summary stats.
        """
        from src.simulation.orders import OrderStatus

        metrics = SimulationMetrics(snapshots=self._snapshots)

        # Order-level stats
        all_orders = self.order_generator.orders
        completed = [o for o in all_orders if o.status == OrderStatus.COMPLETE]
        cycle_times = [o.cycle_time for o in completed if o.cycle_time is not None]

        metrics.total_orders_generated = len(all_orders)
        metrics.total_orders_completed = len(completed)
        metrics.total_simulation_time_s = self.env.now

        elapsed_hours = self.env.now / 3600.0
        metrics.avg_throughput_per_hour = (
            len(completed) / elapsed_hours if elapsed_hours > 0 else 0.0
        )

        if cycle_times:
            metrics.avg_cycle_time_s = float(np.mean(cycle_times))
            metrics.median_cycle_time_s = float(np.median(cycle_times))
            metrics.p95_cycle_time_s = float(np.percentile(cycle_times, 95))

        # AGV-level stats
        total_dist = 0.0
        utilizations = []
        for agv in self.agvs:
            total_active = agv.metrics.total_travel_time_s + agv.metrics.total_wait_time_s
            total_time = self.env.now
            util = (total_active / total_time * 100) if total_time > 0 else 0.0
            utilizations.append(util)
            total_dist += agv.metrics.total_distance_m

            metrics.agv_summaries[agv.id] = {
                "tasks_completed": agv.metrics.tasks_completed,
                "distance_m": agv.metrics.total_distance_m,
                "utilization_pct": util,
                "idle_time_s": agv.metrics.total_idle_time_s,
                "charging_time_s": agv.metrics.total_charging_time_s,
                "wait_time_s": agv.metrics.total_wait_time_s,
                "final_battery_pct": agv.battery_pct,
            }

        metrics.total_distance_traveled_m = total_dist
        metrics.avg_agv_utilization_pct = float(np.mean(utilizations)) if utilizations else 0.0

        # Station utilization (approximate: fraction of time at least 1 AGV was processing)
        # This is a rough estimate from snapshots
        if self._snapshots:
            station_busy_counts = {s: 0 for s in self.pick_stations}
            for snap in self._snapshots:
                for station_id, q_len in snap.station_queue_lengths.items():
                    if q_len > 0:
                        station_busy_counts[station_id] += 1
            n_snaps = len(self._snapshots)
            avg_station_util = np.mean(
                [count / n_snaps * 100 for count in station_busy_counts.values()]
            )
            metrics.avg_station_utilization_pct = float(avg_station_util)

        return metrics
