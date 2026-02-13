"""
Pick station and charging station modeling.

Pick stations are modeled as SimPy Resources with capacity 1 (one AGV
processed at a time). Processing time follows a Gamma distribution —
right-skewed, realistic for manual picking where most picks are fast
but some complex orders take much longer.

Charging stations are also SimPy Resources. Charging time is proportional
to the battery deficit.

Design note: We model stations as shared SimPy Resources rather than
independent processes. The AGV process requests the resource, gets
exclusive access, and yields for the processing duration. This is
cleaner than message-passing between station and AGV processes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import simpy

if TYPE_CHECKING:
    from src.warehouse.config import AGVConfig
    from src.simulation.orders import Task

# pylint: disable=global-statement

# Module-level RNG for processing time — set seed from simulation config
_station_rng: np.random.Generator | None = None


def set_station_rng(seed: int) -> None:
    """Initialize the station-level RNG. Called once by the engine."""
    global _station_rng
    _station_rng = np.random.default_rng(seed)


# pylint: disable=unused-argument


def pick_processing_time(
    task: Task,
    config: AGVConfig,
    current_time: float,
    shape: float = 4.0,
    scale: float = 7.5,
) -> float:
    """Sample processing time for a pick station visit.

    The time is drawn from a Gamma(shape, scale) distribution, representing
    the time a human picker takes to retrieve items from the pod.

    Args:
        task: The task being processed (could scale by n_items in future).
        config: AGV config (unused in MVP, placeholder for item-count scaling).
        current_time: Current sim time (unused, placeholder).
        shape: Gamma distribution shape parameter.
        scale: Gamma distribution scale parameter.

    Returns:
        Processing time in seconds.
    """
    global _station_rng
    if _station_rng is None:
        _station_rng = np.random.default_rng(42)

    # Base processing time from Gamma distribution
    base_time = _station_rng.gamma(shape, scale)

    # Floor at 5 seconds (even the fastest pick takes some time)
    return max(5.0, base_time)


def create_pick_station_resources(
    env: simpy.Environment,
    station_node_ids: list[str],
    capacity: int = 1,
) -> dict[str, simpy.Resource]:
    """Create SimPy Resource objects for each pick station.

    Args:
        env: SimPy environment.
        station_node_ids: List of pick station node IDs from the warehouse graph.
        capacity: Number of AGVs a station can process simultaneously.

    Returns:
        Dict mapping station node ID → SimPy Resource.
    """
    return {node_id: simpy.Resource(env, capacity=capacity) for node_id in station_node_ids}


def create_charging_station_resources(
    env: simpy.Environment,
    charger_node_ids: list[str],
    capacity: int = 1,
) -> dict[str, simpy.Resource]:
    """Create SimPy Resource objects for each charging station.

    Args:
        env: SimPy environment.
        charger_node_ids: List of charging station node IDs.
        capacity: Number of AGVs that can charge simultaneously per station.

    Returns:
        Dict mapping charger node ID → SimPy Resource.
    """
    return {node_id: simpy.Resource(env, capacity=capacity) for node_id in charger_node_ids}
