"""
Warehouse graph visualization.

Renders the warehouse layout as a top-down floor plan with:
- Storage racks as colored blocks flanking aisle pairs
- Aisle paths shown as directional arrows
- U-turn cross-links as dashed connectors
- Highways as thick corridors
- Parking as clearly off-highway grey blocks
- Pick stations and chargers as labeled icons

Usage:
    from src.warehouse.config import load_config
    from src.warehouse.layout import GridLayoutGenerator
    from src.analysis.visualizations import plot_warehouse

    config = load_config("config/default_warehouse.yaml")
    wh = GridLayoutGenerator(config).generate()
    fig = plot_warehouse(wh)
    fig.savefig("warehouse_layout.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

from src.warehouse.graph import WarehouseGraph, NodeType, EdgeType

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


# ── Styling constants ────────────────────────────────────────────

STORAGE_COLOR = "#6baed6"
STORAGE_SIZE = 50
AISLE_POINT_COLOR = "#d9d9d9"
AISLE_POINT_SIZE = 8
INTERSECTION_COLOR = "#636363"
INTERSECTION_SIZE = 30
PICK_STATION_COLOR = "#e6550d"
PICK_STATION_SIZE = 140
CHARGING_COLOR = "#31a354"
CHARGING_SIZE = 120
PARKING_COLOR = "#969696"
PARKING_SIZE = 90

EDGE_STYLES: dict[EdgeType, dict] = {
    EdgeType.HIGHWAY: {
        "color": "#525252",
        "linewidth": 2.5,
        "alpha": 0.6,
        "linestyle": "-",
    },
    EdgeType.AISLE: {
        "color": "#9ecae1",
        "linewidth": 0.8,
        "alpha": 0.5,
        "linestyle": "-",
    },
    EdgeType.RACK_ACCESS: {
        "color": "#6baed6",
        "linewidth": 0.4,
        "alpha": 0.25,
        "linestyle": "-",
    },
    EdgeType.STATION_ACCESS: {
        "color": "#bdbdbd",
        "linewidth": 0.8,
        "alpha": 0.4,
        "linestyle": "--",
    },
    EdgeType.UTURN: {
        "color": "#fd8d3c",
        "linewidth": 1.0,
        "alpha": 0.6,
        "linestyle": "--",
    },
}


def plot_warehouse(
    warehouse: WarehouseGraph,
    title: str = "Warehouse Layout",
    figsize: tuple[float, float] | None = None,
    show_arrows: bool = True,
    show_node_labels: bool = False,
    highlight_nodes: list[str] | None = None,
    highlight_color: str = "#e41a1c",
) -> Figure:
    """Render the warehouse graph as a top-down floor plan.

    Args:
        warehouse: The WarehouseGraph to plot.
        title: Plot title.
        figsize: Figure size in inches. Auto-calculated if None.
        show_arrows: Draw direction arrows on one-way aisle edges.
        show_node_labels: Label all nodes with IDs (noisy for large warehouses).
        highlight_nodes: Optional list of node IDs to highlight (e.g., a path).
        highlight_color: Color for highlighted nodes.

    Returns:
        matplotlib Figure object.
    """
    g = warehouse.graph
    positions = {n: (d["x"], d["y"]) for n, d in g.nodes(data=True)}

    # ── Figure sizing ────────────────────────────────────────────
    if figsize is None:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_range = max(xs) - min(xs) + 6
        y_range = max(ys) - min(ys) + 6
        aspect = x_range / max(y_range, 1)
        fig_width = min(18, max(10, aspect * 8))
        fig_height = fig_width / max(aspect, 0.4)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("#fdfdfd")
    fig.patch.set_facecolor("white")

    # ── Draw edges (back to front) ───────────────────────────────
    _draw_edges(ax, g, positions, show_arrows)

    # ── Draw nodes by type ───────────────────────────────────────
    _draw_storage(ax, g, positions)
    _draw_aisle_points(ax, g, positions)
    _draw_intersections(ax, g, positions)
    _draw_parking(ax, g, positions)
    _draw_stations(
        ax, g, positions, NodeType.PICK_STATION, PICK_STATION_COLOR, PICK_STATION_SIZE, "^", "PS"
    )
    _draw_stations(ax, g, positions, NodeType.CHARGING, CHARGING_COLOR, CHARGING_SIZE, "D", "CS")

    # ── Highlights ───────────────────────────────────────────────
    if highlight_nodes:
        hx = [positions[n][0] for n in highlight_nodes if n in positions]
        hy = [positions[n][1] for n in highlight_nodes if n in positions]
        ax.scatter(
            hx,
            hy,
            c=highlight_color,
            s=180,
            marker="o",
            zorder=10,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.85,
        )

    # ── Node labels ──────────────────────────────────────────────
    if show_node_labels:
        for nid, (x, y) in positions.items():
            ax.annotate(
                nid,
                (x, y),
                textcoords="offset points",
                xytext=(2, 2),
                fontsize=4,
                alpha=0.5,
                zorder=11,
            )

    # ── Legend ────────────────────────────────────────────────────
    _add_legend(ax, warehouse)

    # ── Axis formatting ──────────────────────────────────────────
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x (meters)", fontsize=10)
    ax.set_ylabel("y (meters)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.1, linestyle="--")

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad = 3
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.tight_layout()
    return fig


# ── Edge drawing ─────────────────────────────────────────────────


def _draw_edges(ax: Axes, g, positions: dict, show_arrows: bool) -> None:
    """Draw all edges, styled by type."""
    drawn_pairs: set[tuple] = set()

    for u, v, data in g.edges(data=True):
        etype = data.get("edge_type", EdgeType.AISLE)
        style = EDGE_STYLES.get(etype, EDGE_STYLES[EdgeType.AISLE])

        pair = tuple(sorted([u, v]))
        is_bidir = g.has_edge(v, u)
        if pair in drawn_pairs:
            continue

        x0, y0 = positions[u]
        x1, y1 = positions[v]

        ax.plot(
            [x0, x1],
            [y0, y1],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            zorder=1,
            solid_capstyle="round",
        )

        # Direction arrow on one-way aisle edges
        if show_arrows and not is_bidir and etype == EdgeType.AISLE:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            length = max(np.sqrt(dx**2 + dy**2), 0.01)
            ax.annotate(
                "",
                xy=(mx + dx / length * 0.25, my + dy / length * 0.25),
                xytext=(mx - dx / length * 0.25, my - dy / length * 0.25),
                arrowprops=dict(
                    arrowstyle="->", color=style["color"], lw=0.8, alpha=style["alpha"] + 0.15
                ),
                zorder=1,
            )

        drawn_pairs.add(pair)


# ── Node drawing by type ─────────────────────────────────────────


def _draw_storage(ax: Axes, g, positions: dict) -> None:
    """Draw storage nodes as larger colored squares."""
    
    nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == NodeType.STORAGE]
    if not nodes:
        return
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        c=STORAGE_COLOR,
        s=STORAGE_SIZE,
        marker="s",
        zorder=2,
        edgecolors="white",
        linewidths=0.3,
        alpha=0.85,
    )


def _draw_aisle_points(ax: Axes, g, positions: dict) -> None:
    """Draw aisle waypoints as tiny dots (infrastructure, not prominent)."""
    nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == NodeType.AISLE_POINT]
    if not nodes:
        return
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        c=AISLE_POINT_COLOR,
        s=AISLE_POINT_SIZE,
        marker="o",
        zorder=3,
        edgecolors="none",
        alpha=0.6,
    )


def _draw_intersections(ax: Axes, g, positions: dict) -> None:
    """Draw highway intersections as medium grey circles."""
    nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == NodeType.INTERSECTION]
    if not nodes:
        return
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        c=INTERSECTION_COLOR,
        s=INTERSECTION_SIZE,
        marker="o",
        zorder=4,
        edgecolors="white",
        linewidths=0.5,
    )


def _draw_parking(ax: Axes, g, positions: dict) -> None:
    """Draw parking spots as large grey squares, clearly off-highway."""
    nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == NodeType.PARKING]
    if not nodes:
        return
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        c=PARKING_COLOR,
        s=PARKING_SIZE,
        marker="s",
        zorder=4,
        edgecolors="white",
        linewidths=1.0,
        alpha=0.7,
    )
    # Label
    for _, x, y in zip(nodes, xs, ys):
        ax.annotate(
            "P",
            (x, y),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="white",
            zorder=5,
        )


def _draw_stations(
    ax: Axes,
    g,
    positions: dict,
    node_type: NodeType,
    color: str,
    size: float,
    marker: str,
    prefix: str,
) -> None:
    """Draw pick stations or charging stations with labels."""
    nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == node_type]
    if not nodes:
        return
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    ax.scatter(xs, ys, c=color, s=size, marker=marker, zorder=6, edgecolors="white", linewidths=1.0)
    for nid, x, y in zip(nodes, xs, ys):
        short_label = nid.replace(f"{prefix}_", prefix)
        offset_y = -14 if node_type == NodeType.PICK_STATION else 14
        ax.annotate(
            short_label,
            (x, y),
            textcoords="offset points",
            xytext=(0, offset_y),
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            color=color,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )


# ── Legend ────────────────────────────────────────────────────────


def _add_legend(ax: Axes, warehouse: WarehouseGraph) -> None:
    """Add a clean legend with node and edge type entries."""
    handles = []

    # Node types
    node_legend = [
        (NodeType.STORAGE, STORAGE_COLOR, STORAGE_SIZE, "s", "Storage Rack"),
        (NodeType.AISLE_POINT, AISLE_POINT_COLOR, 30, "o", "Aisle Path"),
        (NodeType.INTERSECTION, INTERSECTION_COLOR, INTERSECTION_SIZE, "o", "Highway Junction"),
        (NodeType.PARKING, PARKING_COLOR, PARKING_SIZE, "s", "Parking Bay"),
        (NodeType.PICK_STATION, PICK_STATION_COLOR, PICK_STATION_SIZE, "^", "Pick Station"),
        (NodeType.CHARGING, CHARGING_COLOR, CHARGING_SIZE, "D", "Charging Station"),
    ]
    for ntype, color, size, marker, label in node_legend:
        if warehouse.nodes_by_type(ntype):
            h = ax.scatter(
                [],
                [],
                c=color,
                s=max(size, 50),
                marker=marker,
                label=label,
                edgecolors="white",
                linewidths=0.5,
            )
            handles.append(h)

    # Edge types
    edge_legend = [
        (EdgeType.HIGHWAY, "Highway"),
        (EdgeType.AISLE, "Aisle (one-way)"),
        (EdgeType.UTURN, "U-turn Link"),
        (EdgeType.RACK_ACCESS, "Rack Access"),
    ]
    for etype, label in edge_legend:
        style = EDGE_STYLES.get(etype)
        if style:
            h = mpatches.Patch(color=style["color"], alpha=style["alpha"], label=label)
            handles.append(h)

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8.5,
        framealpha=0.95,
        title="Legend",
        title_fontsize=9,
    )


# ── Stats dashboard ──────────────────────────────────────────────


def plot_warehouse_stats(warehouse: WarehouseGraph) -> Figure:
    """Render a three-panel summary: layout + node counts + distance histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Mini layout
    g = warehouse.graph
    positions = {n: (d["x"], d["y"]) for n, d in g.nodes(data=True)}
    ax_layout = axes[0]
    _draw_edges(ax_layout, g, positions, show_arrows=False)
    _draw_storage(ax_layout, g, positions)
    _draw_aisle_points(ax_layout, g, positions)
    _draw_intersections(ax_layout, g, positions)
    _draw_parking(ax_layout, g, positions)
    _draw_stations(
        ax_layout, g, positions, NodeType.PICK_STATION, PICK_STATION_COLOR, 80, "^", "PS"
    )
    _draw_stations(ax_layout, g, positions, NodeType.CHARGING, CHARGING_COLOR, 60, "D", "CS")
    ax_layout.set_title("Layout", fontsize=11, fontweight="bold")
    ax_layout.set_aspect("equal")
    ax_layout.grid(True, alpha=0.1)

    # Panel 2: Node distribution
    ax_bar = axes[1]
    type_counts = {}
    type_colors = {
        NodeType.STORAGE: STORAGE_COLOR,
        NodeType.AISLE_POINT: AISLE_POINT_COLOR,
        NodeType.INTERSECTION: INTERSECTION_COLOR,
        NodeType.PARKING: PARKING_COLOR,
        NodeType.PICK_STATION: PICK_STATION_COLOR,
        NodeType.CHARGING: CHARGING_COLOR,
    }
    for _, data in g.nodes(data=True):
        ntype = data.get("node_type", NodeType.INTERSECTION)
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    names = [nt.name for nt in type_counts]
    counts = list(type_counts.values())
    colors = [type_colors.get(nt, "#999999") for nt in type_counts]
    bars = ax_bar.barh(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_xlabel("Count")
    ax_bar.set_title("Node Distribution", fontsize=11, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 3: Distance histogram
    ax_hist = axes[2]
    storage_nodes = warehouse.nodes_by_type(NodeType.STORAGE)
    ps_nodes = warehouse.nodes_by_type(NodeType.PICK_STATION)
    distances = []
    for s in storage_nodes:
        min_d = float("inf")
        for ps in ps_nodes:
            try:
                d = warehouse.shortest_path_distance(s, ps)
                min_d = min(min_d, d)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        if min_d < float("inf"):
            distances.append(min_d)

    if distances:
        ax_hist.hist(distances, bins=20, color=STORAGE_COLOR, edgecolor="white", alpha=0.8)
        ax_hist.axvline(
            np.mean(distances),
            color=PICK_STATION_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {np.mean(distances):.1f}m",
        )
        ax_hist.axvline(
            np.median(distances),
            color=CHARGING_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {np.median(distances):.1f}m",
        )
        ax_hist.legend(fontsize=8)
    ax_hist.set_xlabel("Distance to nearest pick station (m)")
    ax_hist.set_ylabel("# Storage nodes")
    ax_hist.set_title("Storage → Station Distance", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Warehouse Summary — {warehouse.n_nodes} nodes, {warehouse.n_edges} edges",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig
