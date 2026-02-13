"""
Order and Task models, plus the stochastic order generator.

Design decisions:
- Each order maps to exactly ONE task (one pod retrieval). This is an
  explicit MVP simplification — real systems need pod selection to minimize
  multi-pod orders (a set cover problem). Flag this in interviews.
- Orders arrive via a time-varying Poisson process. The rate multiplier
  models intraday demand patterns (surge at 10 AM, dip at 2 PM, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
import simpy

from src.warehouse.graph import NodeType

if TYPE_CHECKING:
    from src.warehouse.config import OrderConfig
    from src.warehouse.graph import WarehouseGraph


class OrderStatus(Enum):
    """Valid order status"""

    PENDING = auto()  # Waiting for task assignment
    ASSIGNED = auto()  # Task assigned to an AGV
    PICKING = auto()  # Being processed at pick station
    COMPLETE = auto()  # Delivered


class TaskStatus(Enum):
    """Valid task status"""

    UNASSIGNED = auto()  # Waiting for AGV assignment
    ASSIGNED = auto()  # AGV assigned, not yet started
    IN_PROGRESS = auto()  # AGV is actively working on this
    COMPLETE = auto()  # Pod returned to storage


class OrderPriority(Enum):
    """Valid order priorities"""

    STANDARD = auto()
    EXPRESS = auto()


@dataclass
class Order:
    """A customer order requiring fulfillment.

    Attributes:
        id: Unique order identifier.
        arrival_time: Simulation time when order entered the system.
        n_items: Number of items in this order.
        priority: Express or standard.
        status: Current lifecycle status.
        completion_time: Sim time when order was fully processed (None if incomplete).
    """

    id: str
    arrival_time: float
    n_items: int
    priority: OrderPriority = OrderPriority.STANDARD
    status: OrderStatus = OrderStatus.PENDING
    completion_time: float | None = None

    @property
    def cycle_time(self) -> float | None:
        """Total time from arrival to completion, or None if incomplete."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


@dataclass
class Task:
    """A single pod-retrieval task derived from an order.

    MVP simplification: 1 order = 1 task = 1 pod retrieval.

    Attributes:
        id: Unique task identifier.
        order_id: Parent order.
        pod_location: Storage node where the pod currently sits.
        pick_station: Destination pick station node.
        return_location: Where to return the pod after processing.
        assigned_agv: AGV ID if assigned, None otherwise.
        status: Current lifecycle status.
        created_time: Sim time when task was created.
        assigned_time: Sim time when AGV was assigned.
        completed_time: Sim time when task finished.
        priority: Inherited from parent order.
    """

    id: str
    order_id: str
    pod_location: str  # Node ID
    pick_station: str  # Node ID
    return_location: str  # Node ID (may differ from pod_location for re-slotting)
    order_ref: Order | None = None  # Back-reference to parent order for status updates
    priority: OrderPriority = OrderPriority.STANDARD
    assigned_agv: str | None = None
    status: TaskStatus = TaskStatus.UNASSIGNED
    created_time: float = 0.0
    assigned_time: float | None = None
    completed_time: float | None = None


class OrderGenerator:
    """SimPy process that generates orders at a time-varying Poisson rate.

    The intraday demand profile is modeled as a piecewise multiplier on the
    base rate. This is more realistic than a flat Poisson rate and produces
    the peak/off-peak patterns that stress-test the system.

    Demand profile (multiplier on base_rate):
        Hour 0-1: 0.5  (early morning ramp-up)
        Hour 1-2: 1.0  (steady state)
        Hour 2-3: 1.5  (peak)
        Hour 3-4: 0.8  (tapering)

    These are configurable but defaulted to something realistic.
    """

    # Hourly demand multipliers (index = hour of simulation)
    DEFAULT_DEMAND_PROFILE = [0.5, 1.0, 1.5, 0.8, 1.0, 1.2, 0.7, 0.5]

    def __init__(
        self,
        env: simpy.Environment,
        config: OrderConfig,
        warehouse: WarehouseGraph,
        rng: np.random.Generator,
        demand_profile: list[float] | None = None,
    ) -> None:
        self.env = env
        self.config = config
        self.warehouse = warehouse
        self.rng = rng
        self.demand_profile = demand_profile or self.DEFAULT_DEMAND_PROFILE

        self.orders: list[Order] = []
        self.tasks: list[Task] = []
        self.pending_tasks: list[Task] = []  # tasks waiting for assignment

        self._order_counter = 0
        self._storage_nodes: list[str] | None = None
        self._pick_stations: list[str] | None = None

    def _get_storage_nodes(self) -> list[str]:
        """Lazy-load storage nodes from warehouse graph."""

        if self._storage_nodes is None:
            self._storage_nodes = self.warehouse.nodes_by_type(NodeType.STORAGE)
        return self._storage_nodes

    def _get_pick_stations(self) -> list[str]:
        """Lazy-load pick station nodes."""
        if self._pick_stations is None:
            self._pick_stations = self.warehouse.nodes_by_type(NodeType.PICK_STATION)
        return self._pick_stations

    def _demand_multiplier(self, sim_time: float) -> float:
        """Get the demand rate multiplier for the current simulation time."""
        hour_idx = int(sim_time / 3600) % len(self.demand_profile)
        return self.demand_profile[hour_idx]

    def _create_order_and_task(self) -> Task:
        """Create an order and its corresponding task."""
        self._order_counter += 1
        order_id = f"ORD_{self._order_counter:05d}"
        task_id = f"TSK_{self._order_counter:05d}"

        # Determine priority
        is_express = self.rng.random() < self.config.express_fraction
        priority = OrderPriority.EXPRESS if is_express else OrderPriority.STANDARD

        # Random number of items (geometric-ish, clamped)
        n_items = int(
            np.clip(
                self.rng.geometric(1.0 / self.config.items_per_order_mean),
                self.config.items_per_order_min,
                self.config.items_per_order_max,
            )
        )

        order = Order(
            id=order_id,
            arrival_time=self.env.now,
            n_items=n_items,
            priority=priority,
        )
        self.orders.append(order)

        # Assign a random pod location (in reality, this depends on which SKUs
        # are needed — MVP uses uniform random across storage nodes)
        storage_nodes = self._get_storage_nodes()
        pod_location = self.rng.choice(storage_nodes)

        # Assign to least-queued pick station (simple load balancing)
        pick_stations = self._get_pick_stations()
        pick_station = self.rng.choice(pick_stations)

        task = Task(
            id=task_id,
            order_id=order_id,
            pod_location=pod_location,
            pick_station=pick_station,
            return_location=pod_location,  # MVP: return to same location
            order_ref=order,  # Back-reference for status propagation
            priority=priority,
            created_time=self.env.now,
        )
        self.tasks.append(task)
        self.pending_tasks.append(task)

        return task

    def run(self):
        """SimPy generator process: yield inter-arrival times, create orders."""

        while True:
            # Compute current rate with demand profile
            multiplier = self._demand_multiplier(self.env.now)
            current_rate = self.config.base_rate_per_min * multiplier  # orders/min
            if current_rate <= 0:
                yield self.env.timeout(60)  # wait a minute if rate is zero
                continue

            # Inter-arrival time: Exponential(1/rate), converted to seconds
            mean_interval_s = 60.0 / current_rate
            interval = self.rng.exponential(mean_interval_s)
            yield self.env.timeout(interval)

            self._create_order_and_task()

    def get_and_clear_pending(self) -> list[Task]:
        """Return pending tasks and clear the buffer. Called by the dispatcher."""

        pending = list(self.pending_tasks)
        self.pending_tasks.clear()
        return pending
