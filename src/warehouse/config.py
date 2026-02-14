"""
Warehouse configuration dataclasses and YAML loader.

All simulation parameters live here as typed, validated dataclasses.
Load from YAML with `load_config()` or construct directly for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class WarehouseLayoutConfig:
    """Physical warehouse dimensions.

    Aisles are paired: each pair has a south-bound (left) and north-bound
    (right) one-way aisle. Storage racks flank each pair on both outer sides.
    U-turn cross-links between paired aisles allow AGVs to switch direction
    without traveling to a highway.
    """

    n_aisles: int = 10  # Total one-way aisles (must be even; forms n/2 pairs)
    bays_per_aisle: int = 20
    bay_depth_m: float = 1.5  # Meters between bays along the aisle
    inner_pair_gap_m: float = 2.0  # Center-to-center between paired aisles
    pair_spacing_m: float = 7.0  # Center-to-center between left aisles of consecutive pairs
    rack_offset_m: float = 1.5  # Storage offset from aisle center
    uturn_interval: int = 5  # U-turn cross-link every N bays
    highway_capacity: int = 2


@dataclass(frozen=True)
class StationConfig:
    """Pick station, charging, and parking layout."""

    n_pick_stations: int = 4
    n_charging_stations: int = 4
    n_parking_spots: int = 15
    pick_station_position: Literal["south", "central"] = "south"


@dataclass(frozen=True)
class AGVConfig:
    """AGV physical and operational parameters."""

    speed_mps: float = 1.5
    battery_capacity_pct: float = 100.0
    battery_drain_per_meter: float = 0.08
    battery_drain_per_lift: float = 0.5
    battery_charge_rate: float = 0.5  # % per second
    battery_threshold: float = 20.0
    pod_pickup_time_s: float = 5.0
    pod_dropoff_time_s: float = 5.0


@dataclass(frozen=True)
class OrderConfig:
    """Order generation parameters."""

    base_rate_per_min: float = 8.0
    items_per_order_min: int = 1
    items_per_order_max: int = 8
    items_per_order_mean: float = 4.0
    express_fraction: float = 0.1   # Fraction of the total orders with express priority


@dataclass(frozen=True)
class PickStationConfig:
    """Pick station processing time distribution (Gamma)."""

    processing_shape: float = 4.0
    processing_scale: float = 7.5  # mean = shape * scale = 30s


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation runtime parameters."""

    duration_hours: float = 4.0         # total simulation duration
    dispatch_interval_s: float = 5.0    # time between consecutive dispatch decisions
    metrics_interval_s: float = 60.0    # KPI aggregation interval
    random_seed: int = 42               # random seed for reproducibility

    @property
    def duration_s(self) -> float:
        """Returns the simulation duration in seconds"""

        return self.duration_hours * 3600.0


@dataclass(frozen=True)
class WarehouseConfig:
    """Top-level configuration aggregating all sub-configs."""

    warehouse: WarehouseLayoutConfig = field(default_factory=WarehouseLayoutConfig)
    stations: StationConfig = field(default_factory=StationConfig)
    agv: AGVConfig = field(default_factory=AGVConfig)
    orders: OrderConfig = field(default_factory=OrderConfig)
    pick_station: PickStationConfig = field(default_factory=PickStationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def load_config(path: str | Path) -> WarehouseConfig:
    """Load a WarehouseConfig from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Fully constructed WarehouseConfig with all sub-configs.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return WarehouseConfig(
        warehouse=WarehouseLayoutConfig(**raw.get("warehouse", {})),
        stations=StationConfig(**raw.get("stations", {})),
        agv=AGVConfig(**raw.get("agv", {})),
        orders=OrderConfig(**raw.get("orders", {})),
        pick_station=PickStationConfig(**raw.get("pick_station", {})),
        simulation=SimulationConfig(**raw.get("simulation", {})),
    )
