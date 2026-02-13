"""
Generate warehouse layout diagrams.

Produces publication-quality floor plan visualizations from the warehouse
configuration. Outputs PNG files to the current directory.

Usage:
    python plot_warehouse.py                          # Default config
    python plot_warehouse.py --config path/to/config.yaml
    python plot_warehouse.py --output my_layout.png
    python plot_warehouse.py --stats                  # Include stats dashboard
    python plot_warehouse.py --path SL_2_0_7 PS_0     # Highlight a path
    python plot_warehouse.py --labels                  # Show all node IDs
"""

import argparse
from pathlib import Path

from src.warehouse.config import load_config, WarehouseConfig, WarehouseLayoutConfig, StationConfig
from src.warehouse.layout import GridLayoutGenerator
from src.warehouse.graph import NodeType
from src.analysis.visualizations import plot_warehouse, plot_warehouse_stats


def default_config() -> WarehouseConfig:
    """Fallback config when no YAML is provided."""
    return WarehouseConfig(
        warehouse=WarehouseLayoutConfig(
            n_aisles=10,
            bays_per_aisle=20,
            bay_depth_m=1.5,
            inner_pair_gap_m=2.0,
            pair_spacing_m=7.0,
            rack_offset_m=1.5,
            uturn_interval=5,
            highway_capacity=2,
        ),
        stations=StationConfig(
            n_pick_stations=4,
            n_charging_stations=4,
            n_parking_spots=12,
        ),
    )


def print_summary(wh) -> None:
    """Print graph summary to console."""
    print(f"\nWarehouse graph: {wh.n_nodes} nodes, {wh.n_edges} edges")
    for nt in NodeType:
        count = len(wh.nodes_by_type(nt))
        if count > 0:
            print(f"  {nt.name:15s}: {count}")

    issues = wh.validate()
    if issues:
        print("\n⚠️  Validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Validation passed")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(
        description="Generate warehouse layout diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to warehouse YAML config (default: config/default_warehouse.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="warehouse_layout.png",
        help="Output PNG filename (default: warehouse_layout.png)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default=None,
        help="Custom plot title",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Also generate a stats dashboard (warehouse_stats.png)",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Show node ID labels on the plot (noisy for large warehouses)",
    )
    parser.add_argument(
        "--path",
        nargs=2,
        metavar=("FROM", "TO"),
        help="Highlight shortest path between two node IDs (e.g., --path SL_2_0_7 PS_0)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image resolution (default: 150)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    config_path = args.config
    if config_path is None:
        default_path = Path(__file__).parent / "config" / "default_warehouse.yaml"
        if default_path.exists():
            config_path = str(default_path)

    if config_path:
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default config (no YAML found)")
        config = default_config()

    # ── Build warehouse ──────────────────────────────────────────
    wh = GridLayoutGenerator(config).generate()
    wh.precompute_distances()
    print_summary(wh)

    # ── Title ────────────────────────────────────────────────────
    n_pairs = config.warehouse.n_aisles // 2
    title = args.title or (
        f"Warehouse Layout — {n_pairs} Aisle Pairs × {config.warehouse.bays_per_aisle} Bays"
    )

    # ── Path highlighting ────────────────────────────────────────
    highlight = None
    if args.path:
        src_node, dst_node = args.path
        try:
            path = wh.shortest_path(src_node, dst_node)
            dist = wh.shortest_path_distance(src_node, dst_node)
            highlight = path
            print(f"\nPath: {src_node} → {dst_node}")
            print(f"  Hops: {len(path)}, Distance: {dist:.1f}m")
            title += f"\n{src_node} → {dst_node} ({dist:.1f}m, {len(path)} hops)"
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"\n⚠️  Could not find path {src_node} → {dst_node}: {e}")

    # ── Generate layout plot ─────────────────────────────────────
    fig = plot_warehouse(
        wh,
        title=title,
        show_node_labels=args.labels,
        highlight_nodes=highlight,
    )
    output_path = Path(args.output)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\n📊 Layout saved: {output_path}")

    # ── Optional stats dashboard ─────────────────────────────────
    if args.stats:
        stats_path = output_path.with_name(output_path.stem + "_stats" + output_path.suffix)
        fig_stats = plot_warehouse_stats(wh)
        fig_stats.savefig(stats_path, dpi=args.dpi, bbox_inches="tight")
        print(f"📊 Stats saved:  {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
