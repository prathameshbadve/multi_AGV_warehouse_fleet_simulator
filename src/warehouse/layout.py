"""
Warehouse layout generators — paired aisle design.

Builds a WarehouseGraph from a WarehouseConfig. Models an Amazon/Kiva-style
warehouse with paired one-way aisles, storage racks on both outer sides,
and U-turn cross-links between paired aisles.

Layout geometry (top-down, not to scale):

     [CS0]         [CS1]         [CS2]         [CS3]
       |             |             |             |
    ═══H═════════════H═════════════H═════════════H═══  ← Top highway
       |             |             |             |
    [PK]  [PK]                          [PK]  [PK]    ← Parking (off highway)
       |             |             |             |
    ┌──┐  ┌──┐   ┌──┐  ┌──┐   ┌──┐  ┌──┐   ┌──┐  ┌──┐
    │S │  │S │   │S │  │S │   │S │  │S │   │S │  │S │
    │  │←A↓ A↑ → │  │  │  │← A↓ A↑ →│  │   │  │←A↓ A↑ →  │  │
    │S │  │S │   │S │  │S │   │S │  │S │   │S │  │S │
    │  │  ↓ U ↑  │  │   │  │  ↓ U ↑  │  │   │  │  ↓ U ↑  │  │
    │S │  │S │   │S │  │S │   │S │  │S │   │S │  │S │
    └──┘  └──┘   └──┘  └──┘   └──┘  └──┘   └──┘  └──┘
       |             |             |             |
    ═══H═════════════H═════════════H═════════════H═══  ← Middle highway
       ...         ...           ...           ...
    ═══H═════════════H═════════════H═════════════H═══  ← Bottom highway
       |             |             |             |
            [PS0]         [PS1]         [PS2]         [PS3]

    S  = Storage rack (side branch, accessed from adjacent aisle only)
    A↓ = Left aisle (south-bound)
    A↑ = Right aisle (north-bound)
    U  = U-turn cross-link between paired aisles
    H  = Highway intersection
    PK = Parking spot (off-highway spur)

Coordinate system:
    x → east (pair index determines x band)
    y → north (bay index, 0 = south / pick station side)
"""

from __future__ import annotations

from src.warehouse.config import WarehouseConfig
from src.warehouse.graph import WarehouseGraph, NodeType, EdgeType


class GridLayoutGenerator:
    """Generates a paired-aisle warehouse graph.

    Each pair of aisles has:
    - Left aisle: one-way south-bound (decreasing y)
    - Right aisle: one-way north-bound (increasing y)
    - Storage racks on the outer left and outer right sides
    - U-turn cross-links at regular intervals
    - Connections to horizontal highways at top and bottom of each segment
    """

    def __init__(self, config: WarehouseConfig) -> None:
        self.wh = config.warehouse
        self.st = config.stations
        self.config = config

        self.n_pairs = self.wh.n_aisles // 2
        if self.wh.n_aisles % 2 != 0:
            raise ValueError(f"n_aisles must be even for paired layout, got {self.wh.n_aisles}")

    def generate(self) -> WarehouseGraph:
        """Build and return the complete warehouse graph."""
        g = WarehouseGraph()

        self._add_highway_intersections(g)
        self._add_aisle_pairs(g)
        self._add_highway_edges(g)
        self._add_pick_stations(g)
        self._add_charging_stations(g)
        self._add_parking_spots(g)

        issues = g.validate()
        if issues:
            for issue in issues:
                print(f"[LAYOUT WARNING] {issue}")

        return g

    # ── Geometry helpers ─────────────────────────────────────────

    def _highway_y_positions(self) -> list[float]:
        """Compute y-coordinates for horizontal highways."""
        max_y = self.wh.bays_per_aisle * self.wh.bay_depth_m
        bottom_y = 0.0
        top_y = max_y

        if self.wh.bays_per_aisle <= 15:
            return [bottom_y, top_y]

        n_segments = max(2, self.wh.bays_per_aisle // 15 + 1)
        step = max_y / n_segments
        return [round(i * step, 2) for i in range(n_segments + 1)]

    def _pair_x_positions(self, pair: int) -> dict[str, float]:
        """Compute x-coordinates for all elements of an aisle pair.

        Returns dict with keys: left_aisle, right_aisle, left_rack, right_rack
        """
        left_aisle = pair * self.wh.pair_spacing_m
        right_aisle = left_aisle + self.wh.inner_pair_gap_m
        left_rack = left_aisle - self.wh.rack_offset_m
        right_rack = right_aisle + self.wh.rack_offset_m
        return {
            "left_aisle": left_aisle,
            "right_aisle": right_aisle,
            "left_rack": left_rack,
            "right_rack": right_rack,
        }

    # ── Highway intersections ────────────────────────────────────

    def _add_highway_intersections(self, g: WarehouseGraph) -> None:
        """Add intersection nodes where each aisle meets each highway.

        Each pair has TWO intersections per highway (one per aisle),
        connected by a short bidirectional edge.
        """
        highways_y = self._highway_y_positions()

        for pair in range(self.n_pairs):
            xp = self._pair_x_positions(pair)

            for hw_idx, y in enumerate(highways_y):
                il_id = f"IL_{pair}_{hw_idx}"
                ir_id = f"IR_{pair}_{hw_idx}"

                g.add_node(il_id, NodeType.INTERSECTION, x=xp["left_aisle"], y=y)
                g.add_node(ir_id, NodeType.INTERSECTION, x=xp["right_aisle"], y=y)

                # Connect the pair's two intersections (short inner-pair link)
                g.add_bidirectional_edge(
                    il_id,
                    ir_id,
                    EdgeType.HIGHWAY,
                    self.wh.inner_pair_gap_m,
                    capacity=2,
                )

    # ── Aisle pairs with storage and U-turns ─────────────────────

    def _add_aisle_pairs(self, g: WarehouseGraph) -> None:
        """Add paired aisles, side-branch storage, and U-turn cross-links."""
        highways_y = self._highway_y_positions()

        for pair in range(self.n_pairs):
            xp = self._pair_x_positions(pair)

            for seg in range(len(highways_y) - 1):
                self._add_aisle_segment(g, pair, seg, xp, highways_y)

    def _add_aisle_segment(
        self,
        g: WarehouseGraph,
        pair: int,
        seg: int,
        xp: dict[str, float],
        highways_y: list[float],
    ) -> None:
        """Add one segment (between two highways) of one aisle pair.

        Creates: aisle waypoints, storage side-branches, U-turn links,
        and connects everything to the highway intersections.
        """
        y_bottom = highways_y[seg]
        y_top = highways_y[seg + 1]
        seg_length = y_top - y_bottom
        n_bays = max(1, int(seg_length / self.wh.bay_depth_m) - 1)
        bay_step = seg_length / (n_bays + 1)

        left_aisle_nodes: list[str] = []  # ordered south → north (increasing y)
        right_aisle_nodes: list[str] = []

        for bay in range(n_bays):
            bay_y = y_bottom + (bay + 1) * bay_step

            # ── Aisle waypoints (AGV travels through these) ──────
            al_id = f"AL_{pair}_{seg}_{bay}"
            ar_id = f"AR_{pair}_{seg}_{bay}"
            g.add_node(al_id, NodeType.AISLE_POINT, x=xp["left_aisle"], y=bay_y)
            g.add_node(ar_id, NodeType.AISLE_POINT, x=xp["right_aisle"], y=bay_y)
            left_aisle_nodes.append(al_id)
            right_aisle_nodes.append(ar_id)

            # ── Storage racks (side branches) ────────────────────
            # Left storage: accessed only from left aisle
            sl_id = f"SL_{pair}_{seg}_{bay}"
            g.add_node(sl_id, NodeType.STORAGE, x=xp["left_rack"], y=bay_y)
            g.add_bidirectional_edge(
                sl_id,
                al_id,
                EdgeType.RACK_ACCESS,
                self.wh.rack_offset_m,
                capacity=1,
            )

            # Right storage: accessed only from right aisle
            sr_id = f"SR_{pair}_{seg}_{bay}"
            g.add_node(sr_id, NodeType.STORAGE, x=xp["right_rack"], y=bay_y)
            g.add_bidirectional_edge(
                sr_id,
                ar_id,
                EdgeType.RACK_ACCESS,
                self.wh.rack_offset_m,
                capacity=1,
            )

            # ── U-turn cross-links ───────────────────────────────
            if self.wh.uturn_interval > 0 and bay % self.wh.uturn_interval == 0:
                g.add_bidirectional_edge(
                    al_id,
                    ar_id,
                    EdgeType.UTURN,
                    self.wh.inner_pair_gap_m,
                    capacity=1,
                )

        # ── Connect aisle waypoints in sequence (one-way) ────────

        # Left aisle = south-bound: top_intersection → highest_bay →
        # ... → lowest_bay → bottom_intersection
        top_il = f"IL_{pair}_{seg + 1}"
        bot_il = f"IL_{pair}_{seg}"
        prev = top_il
        for node in reversed(left_aisle_nodes):
            g.add_edge(prev, node, EdgeType.AISLE, bay_step, capacity=1)
            prev = node
        g.add_edge(prev, bot_il, EdgeType.AISLE, bay_step, capacity=1)

        # Right aisle = north-bound: bottom_intersection → lowest_bay →
        # ... → highest_bay → top_intersection
        top_ir = f"IR_{pair}_{seg + 1}"
        bot_ir = f"IR_{pair}_{seg}"
        prev = bot_ir
        for node in right_aisle_nodes:
            g.add_edge(prev, node, EdgeType.AISLE, bay_step, capacity=1)
            prev = node
        g.add_edge(prev, top_ir, EdgeType.AISLE, bay_step, capacity=1)

    # ── Highway edges ────────────────────────────────────────────

    def _add_highway_edges(self, g: WarehouseGraph) -> None:
        """Add horizontal highway edges connecting aisle pairs.

        Highway path at each level: IL_0 ↔ IR_0 ↔ IL_1 ↔ IR_1 ↔ ...
        The inner-pair links (IL↔IR) are added in _add_highway_intersections.
        This method adds the inter-pair links (IR_p ↔ IL_{p+1}).
        """
        highways_y = self._highway_y_positions()
        inter_pair_dist = self.wh.pair_spacing_m - self.wh.inner_pair_gap_m

        for hw_idx in range(len(highways_y)):
            for pair in range(self.n_pairs - 1):
                node_a = f"IR_{pair}_{hw_idx}"
                node_b = f"IL_{pair + 1}_{hw_idx}"
                g.add_bidirectional_edge(
                    node_a,
                    node_b,
                    EdgeType.HIGHWAY,
                    inter_pair_dist,
                    capacity=self.wh.highway_capacity,
                )

    # ── Pick stations ────────────────────────────────────────────

    def _add_pick_stations(self, g: WarehouseGraph) -> None:
        """Add pick stations along the south edge below the bottom highway."""
        n_ps = self.st.n_pick_stations
        if n_ps == 0:
            return

        pair_indices = self._distribute_positions(n_ps, self.n_pairs)
        offset_y = self.wh.bay_depth_m * 1.5

        for ps_idx, pair in enumerate(pair_indices):
            xp = self._pair_x_positions(pair)
            x = (xp["left_aisle"] + xp["right_aisle"]) / 2
            y = -offset_y

            ps_id = f"PS_{ps_idx}"
            g.add_node(ps_id, NodeType.PICK_STATION, x=x, y=y)

            # Connect to both bottom highway intersections of this pair
            edge_dist = (offset_y**2 + (self.wh.inner_pair_gap_m / 2) ** 2) ** 0.5
            g.add_bidirectional_edge(
                ps_id,
                f"IL_{pair}_0",
                EdgeType.STATION_ACCESS,
                edge_dist,
                capacity=2,
            )
            g.add_bidirectional_edge(
                ps_id,
                f"IR_{pair}_0",
                EdgeType.STATION_ACCESS,
                edge_dist,
                capacity=2,
            )

    # ── Charging stations ────────────────────────────────────────

    def _add_charging_stations(self, g: WarehouseGraph) -> None:
        """Add charging stations along the north edge above the top highway."""
        n_cs = self.st.n_charging_stations
        if n_cs == 0:
            return

        highways_y = self._highway_y_positions()
        top_hw_idx = len(highways_y) - 1
        max_y = highways_y[-1]
        offset_y = self.wh.bay_depth_m * 1.5

        pair_indices = self._distribute_positions(n_cs, self.n_pairs)

        for cs_idx, pair in enumerate(pair_indices):
            xp = self._pair_x_positions(pair)
            x = (xp["left_aisle"] + xp["right_aisle"]) / 2
            y = max_y + offset_y

            cs_id = f"CS_{cs_idx}"
            g.add_node(cs_id, NodeType.CHARGING, x=x, y=y)

            edge_dist = (offset_y**2 + (self.wh.inner_pair_gap_m / 2) ** 2) ** 0.5
            g.add_bidirectional_edge(
                cs_id,
                f"IL_{pair}_{top_hw_idx}",
                EdgeType.STATION_ACCESS,
                edge_dist,
                capacity=2,
            )
            g.add_bidirectional_edge(
                cs_id,
                f"IR_{pair}_{top_hw_idx}",
                EdgeType.STATION_ACCESS,
                edge_dist,
                capacity=2,
            )

    # ── Parking spots ────────────────────────────────────────────

    def _add_parking_spots(self, g: WarehouseGraph) -> None:
        """Add parking spots as clear off-highway spurs.

        Parking is placed in the inter-pair gaps along highways. The
        connectivity and offset direction depends on the highway position:

        - Bottom highway: parking ABOVE highway, connected to IR (north-bound
          aisle entry). AGVs leaving parking head into the storage area.
        - Middle highways: parking ABOVE highway, connected to BOTH IL
          (south-bound entry) and IR (north-bound entry). AGVs can go
          either direction.
        - Top highway: parking BELOW highway, connected to IL (south-bound
          aisle entry). AGVs leaving parking head down into storage.
        """
        
        n_parking = self.st.n_parking_spots
        if n_parking == 0 or self.n_pairs < 2:
            return

        highways_y = self._highway_y_positions()
        n_highways = len(highways_y)

        # Build candidate list with connectivity metadata
        # Each candidate: (x, y, list_of_intersection_ids_to_connect)
        candidates: list[tuple[float, float, list[str]]] = []

        for hw_idx, hw_y in enumerate(highways_y):
            is_bottom = hw_idx == 0
            is_top = hw_idx == n_highways - 1

            # Y offset: top highway places parking below; others above
            if is_top:
                park_y = hw_y - self.wh.bay_depth_m * 0.9
            else:
                park_y = hw_y + self.wh.bay_depth_m * 0.9

            for pair in range(self.n_pairs - 1):
                xp_this = self._pair_x_positions(pair)
                xp_next = self._pair_x_positions(pair + 1)
                gap_x = (xp_this["right_rack"] + xp_next["left_rack"]) / 2

                # Determine which intersections to connect to
                connections: list[str] = []
                if is_bottom:
                    # Connect to IR only (north-bound entry)
                    connections = [f"IR_{pair}_{hw_idx}"]
                elif is_top:
                    # Connect to IL only (south-bound entry)
                    connections = [f"IL_{pair + 1}_{hw_idx}"]
                else:
                    # Middle: connect to both directions
                    connections = [
                        f"IR_{pair}_{hw_idx}",  # north-bound
                        f"IL_{pair + 1}_{hw_idx}",  # south-bound
                    ]

                candidates.append((gap_x, park_y, connections))

        # Evenly sample
        step = max(1, len(candidates) // n_parking)
        selected = candidates[::step][:n_parking]

        for pk_idx, (px, py, connections) in enumerate(selected):
            pk_id = f"PK_{pk_idx}"
            g.add_node(pk_id, NodeType.PARKING, x=px, y=py)

            for int_id in connections:
                int_data = g.get_node(int_id)
                dist = ((px - int_data["x"]) ** 2 + (py - int_data["y"]) ** 2) ** 0.5
                g.add_bidirectional_edge(
                    pk_id,
                    int_id,
                    EdgeType.STATION_ACCESS,
                    dist,
                    capacity=1,
                )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _distribute_positions(n_items: int, n_slots: int) -> list[int]:
        """Evenly distribute n_items across n_slots, returning slot indices."""
        if n_items >= n_slots:
            return list(range(n_slots))
        step = n_slots / (n_items + 1)
        return [int(round((i + 1) * step)) for i in range(n_items)]
