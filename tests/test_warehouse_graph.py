"""Tests for warehouse graph construction and layout generation.

Run with: pytest tests/test_warehouse_graph.py -v
"""

import pytest

from src.warehouse.config import WarehouseConfig, WarehouseLayoutConfig, StationConfig
from src.warehouse.graph import WarehouseGraph, NodeType, EdgeType
from src.warehouse.layout import GridLayoutGenerator


@pytest.fixture
def small_config() -> WarehouseConfig:
    """A minimal warehouse config for fast tests."""
    return WarehouseConfig(
        warehouse=WarehouseLayoutConfig(
            n_aisles=4,
            bays_per_aisle=10,
            aisle_spacing_m=3.0,
            bay_depth_m=1.5,
            highway_capacity=2,
        ),
        stations=StationConfig(
            n_pick_stations=2,
            n_charging_stations=2,
            n_parking_spots=4,
        ),
    )


@pytest.fixture
def small_warehouse(small_config) -> WarehouseGraph:
    """A built warehouse graph from the small config."""
    gen = GridLayoutGenerator(small_config)
    return gen.generate()


class TestWarehouseGraph:
    """Unit tests for the WarehouseGraph data structure."""

    def test_add_and_retrieve_node(self):
        g = WarehouseGraph()
        g.add_node("N1", NodeType.INTERSECTION, x=0.0, y=0.0)
        assert g.n_nodes == 1
        attrs = g.get_node("N1")
        assert attrs["node_type"] == NodeType.INTERSECTION
        assert attrs["x"] == 0.0

    def test_add_edge(self):
        g = WarehouseGraph()
        g.add_node("A", NodeType.INTERSECTION, x=0, y=0)
        g.add_node("B", NodeType.INTERSECTION, x=3, y=0)
        g.add_edge("A", "B", EdgeType.HIGHWAY, distance=3.0, capacity=2)
        assert g.n_edges == 1
        assert g.edge_distance("A", "B") == 3.0

    def test_bidirectional_edge(self):
        g = WarehouseGraph()
        g.add_node("A", NodeType.INTERSECTION, x=0, y=0)
        g.add_node("B", NodeType.INTERSECTION, x=3, y=0)
        g.add_bidirectional_edge("A", "B", EdgeType.HIGHWAY, 3.0, capacity=2)
        assert g.n_edges == 2
        assert g.edge_distance("A", "B") == 3.0
        assert g.edge_distance("B", "A") == 3.0

    def test_nodes_by_type(self):
        g = WarehouseGraph()
        g.add_node("S1", NodeType.STORAGE, x=0, y=0)
        g.add_node("S2", NodeType.STORAGE, x=1, y=0)
        g.add_node("I1", NodeType.INTERSECTION, x=2, y=0)
        assert len(g.nodes_by_type(NodeType.STORAGE)) == 2
        assert len(g.nodes_by_type(NodeType.INTERSECTION)) == 1
        assert len(g.nodes_by_type(NodeType.PICK_STATION)) == 0

    def test_shortest_path(self):
        g = WarehouseGraph()
        g.add_node("A", NodeType.INTERSECTION, x=0, y=0)
        g.add_node("B", NodeType.INTERSECTION, x=3, y=0)
        g.add_node("C", NodeType.INTERSECTION, x=6, y=0)
        g.add_edge("A", "B", EdgeType.HIGHWAY, 3.0)
        g.add_edge("B", "C", EdgeType.HIGHWAY, 3.0)
        assert g.shortest_path_distance("A", "C") == 6.0
        assert g.shortest_path("A", "C") == ["A", "B", "C"]


class TestGridLayout:
    """Tests for the grid warehouse layout generator."""

    def test_generates_valid_graph(self, small_warehouse):
        issues = small_warehouse.validate()
        assert len(issues) == 0, f"Validation issues: {issues}"

    def test_has_required_node_types(self, small_warehouse):
        assert len(small_warehouse.nodes_by_type(NodeType.STORAGE)) > 0
        assert len(small_warehouse.nodes_by_type(NodeType.PICK_STATION)) == 2
        assert len(small_warehouse.nodes_by_type(NodeType.CHARGING)) == 2
        assert len(small_warehouse.nodes_by_type(NodeType.INTERSECTION)) > 0

    def test_pick_stations_reachable_from_storage(self, small_warehouse):
        storage = small_warehouse.nodes_by_type(NodeType.STORAGE)
        pick_stations = small_warehouse.nodes_by_type(NodeType.PICK_STATION)
        # At least one storage node should reach at least one pick station
        reachable = False
        for s in storage[:5]:  # test a sample
            for ps in pick_stations:
                try:
                    dist = small_warehouse.shortest_path_distance(s, ps)
                    if dist < float("inf"):
                        reachable = True
                        break
                except Exception:
                    continue
            if reachable:
                break
        assert reachable, "No storage node can reach any pick station"

    def test_charging_reachable(self, small_warehouse):
        """AGVs should be able to reach charging stations."""
        intersections = small_warehouse.nodes_by_type(NodeType.INTERSECTION)
        chargers = small_warehouse.nodes_by_type(NodeType.CHARGING)
        # At least one intersection should reach a charger
        reachable = False
        for i_node in intersections[:5]:
            for cs in chargers:
                try:
                    small_warehouse.shortest_path_distance(i_node, cs)
                    reachable = True
                    break
                except Exception:
                    continue
            if reachable:
                break
        assert reachable

    def test_storage_zones(self, small_warehouse):
        """Storage nodes should be classifiable into proximity zones."""
        # Precompute distances first
        small_warehouse.precompute_distances()
        zones = small_warehouse.storage_nodes_by_zone()
        total = len(zones["near"]) + len(zones["mid"]) + len(zones["far"])
        assert total == len(small_warehouse.nodes_by_type(NodeType.STORAGE))

    def test_node_count_scales_with_config(self):
        """More aisles / bays should produce more nodes."""
        small = WarehouseConfig(
            warehouse=WarehouseLayoutConfig(n_aisles=4, bays_per_aisle=10),
            stations=StationConfig(n_pick_stations=2, n_charging_stations=2, n_parking_spots=2),
        )
        large = WarehouseConfig(
            warehouse=WarehouseLayoutConfig(n_aisles=10, bays_per_aisle=20),
            stations=StationConfig(n_pick_stations=4, n_charging_stations=4, n_parking_spots=8),
        )
        g_small = GridLayoutGenerator(small).generate()
        g_large = GridLayoutGenerator(large).generate()
        assert g_large.n_nodes > g_small.n_nodes
        assert g_large.n_edges > g_small.n_edges
