"""Warehouse graph representation.

The warehouse is modeled as a directed graph where:
- Nodes represent physical locations (storage bays, intersections, stations)
- Edges represent traversable paths with direction, distance, and capacity
- The graph is the single source of truth for warehouse topology

All pathfinding and simulation modules operate on this graph.
"""

from __future__ import annotations

from enum import Enum, auto

import networkx as nx


class NodeType(Enum):
    """Types of locations in the warehouse."""

    INTERSECTION = auto()  # Highway junction
    STORAGE = auto()  # Pod storage location (side branch off aisle)
    AISLE_POINT = auto()  # Waypoint along an aisle (AGV travels through)
    PICK_STATION = auto()  # Human picker processes items
    CHARGING = auto()  # AGV charging dock
    PARKING = auto()  # Safe waiting / buffer location
    DEPOT = auto()  # AGV starting position


class EdgeType(Enum):
    """Types of traversable paths."""

    AISLE = auto()  # Narrow storage aisle (typically one-way, capacity 1)
    HIGHWAY = auto()  # Wide cross-aisle / main corridor (two-way, capacity 2+)
    STATION_ACCESS = auto()  # Short edge connecting highway to station
    RACK_ACCESS = auto()  # Short edge from aisle waypoint to storage rack
    UTURN = auto()  # Cross-link between paired aisles


class WarehouseGraph:
    """Directed graph representing a warehouse floor.

    Wraps a NetworkX DiGraph with typed nodes and edges, providing
    domain-specific queries (e.g., "give me all storage nodes near
    a pick station") while keeping the raw graph accessible for
    pathfinding algorithms.

    Attributes:
        graph: The underlying NetworkX DiGraph.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._distance_cache: dict[tuple[str, str], float] | None = None

    # ── Node management ──────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        x: float,
        y: float,
        **attrs,
    ) -> None:
        """Add a node with spatial coordinates and type.

        Args:
            node_id: Unique identifier (e.g., "S_3_12" for storage aisle 3, bay 12).
            node_type: What kind of location this is.
            x: X coordinate in meters (east-west).
            y: Y coordinate in meters (north-south).
            **attrs: Additional attributes (e.g., pod_id for storage nodes).
        """
        self.graph.add_node(
            node_id,
            node_type=node_type,
            x=x,
            y=y,
            **attrs,
        )
        self._distance_cache = None  # invalidate cache

    def get_node(self, node_id: str) -> dict:
        """Get all attributes of a node."""
        return self.graph.nodes[node_id]

    def nodes_by_type(self, node_type: NodeType) -> list[str]:
        """Return all node IDs of a given type."""
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == node_type]

    # ── Edge management ──────────────────────────────────────────────

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: EdgeType,
        distance: float,
        capacity: int = 1,
        **attrs,
    ) -> None:
        """Add a directed edge between two nodes.

        Args:
            from_node: Source node ID.
            to_node: Target node ID.
            edge_type: Type of path.
            distance: Physical distance in meters.
            capacity: Max concurrent AGVs on this edge.
            **attrs: Additional attributes.
        """
        self.graph.add_edge(
            from_node,
            to_node,
            edge_type=edge_type,
            distance=distance,
            capacity=capacity,
            **attrs,
        )
        self._distance_cache = None

    def add_bidirectional_edge(
        self,
        node_a: str,
        node_b: str,
        edge_type: EdgeType,
        distance: float,
        capacity: int = 2,
        **attrs,
    ) -> None:
        """Add edges in both directions (for highways and wide corridors)."""
        self.add_edge(node_a, node_b, edge_type, distance, capacity, **attrs)
        self.add_edge(node_b, node_a, edge_type, distance, capacity, **attrs)

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def neighbors(self, node_id: str) -> list[str]:
        """Return successor nodes (nodes reachable from this node)."""
        return list(self.graph.successors(node_id))

    def edge_distance(self, from_node: str, to_node: str) -> float:
        """Get distance of a specific edge. Raises KeyError if edge doesn't exist."""
        return self.graph.edges[from_node, to_node]["distance"]

    def edge_capacity(self, from_node: str, to_node: str) -> int:
        """Get capacity of a specific edge."""
        return self.graph.edges[from_node, to_node]["capacity"]

    def shortest_path_distance(self, source: str, target: str) -> float:
        """Compute shortest path distance ignoring capacity/conflicts.

        Uses cached all-pairs shortest paths if available, otherwise
        computes single-source Dijkstra.

        Returns:
            Distance in meters. Raises nx.NetworkXNoPath if unreachable.
        """
        if self._distance_cache is not None:
            dist = self._distance_cache.get((source, target))
            if dist is not None:
                return dist
            raise nx.NetworkXNoPath(f"No path from {source} to {target}")

        return nx.shortest_path_length(self.graph, source, target, weight="distance")

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Compute shortest path (list of node IDs) ignoring conflicts."""
        return nx.shortest_path(self.graph, source, target, weight="distance")

    def precompute_distances(self, sources: list[str] | None = None) -> None:
        """Precompute shortest-path distances from selected sources to all nodes.

        This is used by the task assignment module to quickly estimate
        travel distances without running full pathfinding.

        Args:
            sources: Node IDs to compute from. If None, computes all-pairs
                     (expensive for large graphs — prefer selective computation).
        """
        self._distance_cache = {}
        if sources is None:
            # All-pairs — use Floyd-Warshall for dense graphs, Johnson for sparse
            lengths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="distance"))
            for src, targets in lengths.items():
                for tgt, dist in targets.items():
                    self._distance_cache[(src, tgt)] = dist
        else:
            for src in sources:
                lengths = nx.single_source_dijkstra_path_length(self.graph, src, weight="distance")
                for tgt, dist in lengths.items():
                    self._distance_cache[(src, tgt)] = dist

    def nearest_node_of_type(self, from_node: str, target_type: NodeType) -> tuple[str, float]:
        """Find the nearest node of a given type from a source node.

        Returns:
            Tuple of (node_id, distance).

        Raises:
            ValueError: If no nodes of the target type exist.
        """
        candidates = self.nodes_by_type(target_type)
        if not candidates:
            raise ValueError(f"No nodes of type {target_type} in the graph")

        best_node = None
        best_dist = float("inf")
        for candidate in candidates:
            try:
                dist = self.shortest_path_distance(from_node, candidate)
                if dist < best_dist:
                    best_dist = dist
                    best_node = candidate
            except nx.NetworkXNoPath:
                continue

        if best_node is None:
            raise ValueError(f"No reachable nodes of type {target_type} from {from_node}")
        return best_node, best_dist

    def storage_nodes_by_zone(self) -> dict[str, list[str]]:
        """Group storage nodes by proximity zone (near/mid/far from pick stations).

        Used for demand-weighted pod placement: high-velocity SKUs
        go in the 'near' zone.

        Returns:
            Dict with keys 'near', 'mid', 'far', each mapping to a list of node IDs.
        """
        pick_stations = self.nodes_by_type(NodeType.PICK_STATION)
        storage_nodes = self.nodes_by_type(NodeType.STORAGE)

        if not pick_stations or not storage_nodes:
            return {"near": [], "mid": [], "far": []}

        # Compute min distance from each storage node to any pick station
        distances = []
        for s_node in storage_nodes:
            min_dist = float("inf")
            for ps in pick_stations:
                try:
                    d = self.shortest_path_distance(s_node, ps)
                    min_dist = min(min_dist, d)
                except nx.NetworkXNoPath:
                    continue
            distances.append((s_node, min_dist))

        distances.sort(key=lambda x: x[1])
        n = len(distances)
        cutoff_near = n // 3
        cutoff_mid = 2 * n // 3

        return {
            "near": [d[0] for d in distances[:cutoff_near]],
            "mid": [d[0] for d in distances[cutoff_near:cutoff_mid]],
            "far": [d[0] for d in distances[cutoff_mid:]],
        }

    def validate(self) -> list[str]:
        """Run basic sanity checks on the graph.

        Returns:
            List of warning/error messages (empty = all good).
        """
        issues = []

        # Check connectivity
        if not nx.is_weakly_connected(self.graph):
            components = list(nx.weakly_connected_components(self.graph))
            issues.append(
                f"Graph is not connected: {len(components)} components "
                f"(sizes: {[len(c) for c in components]})"
            )

        # Check required node types exist
        for required in [NodeType.STORAGE, NodeType.PICK_STATION, NodeType.CHARGING]:
            if not self.nodes_by_type(required):
                issues.append(f"No nodes of type {required.name}")

        # Check pick stations are reachable from at least one storage node
        pick_stations = self.nodes_by_type(NodeType.PICK_STATION)
        storage_nodes = self.nodes_by_type(NodeType.STORAGE)
        if pick_stations and storage_nodes:
            test_storage = storage_nodes[0]
            reachable = False
            for ps in pick_stations:
                if nx.has_path(self.graph, test_storage, ps):
                    reachable = True
                    break
            if not reachable:
                issues.append(f"No pick station reachable from storage node {test_storage}")

        return issues
