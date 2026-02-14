import type { LayoutEdge, LayoutNode, LayoutPayload, SnapshotPayload } from "../types";

interface WarehouseMapProps {
  layout: LayoutPayload | null;
  snapshot: SnapshotPayload | null;
  selectedAgv: string;
}

const NODE_COLOR: Record<string, string> = {
  INTERSECTION: "#5f8fd3",
  AISLE_POINT: "#6f7d8e",
  STORAGE: "#8aa064",
  PICK_STATION: "#f08c3a",
  CHARGING: "#3cbfa9",
  PARKING: "#8f6ee0",
  DEPOT: "#7f7f7f"
};

const AGV_COLOR: Record<string, string> = {
  IDLE: "#66d17a",
  TRAVELING_TO_POD: "#4fa7e7",
  PICKING_UP_POD: "#d2d866",
  TRAVELING_TO_STATION: "#4fa7e7",
  WAITING_AT_STATION: "#f7b267",
  PROCESSING: "#f28f3b",
  RETURNING_POD: "#5ca6ff",
  TRAVELING_TO_CHARGER: "#5ad1bc",
  CHARGING: "#33a89a"
};

function congestionColor(occupancy: number, capacity: number): string {
  if (capacity <= 0) return "#8a9baa";
  const ratio = occupancy / capacity;
  if (ratio < 0.3) return "#4aa96c";
  if (ratio < 0.8) return "#e5b95b";
  return "#dc5e56";
}

function mapY(layout: LayoutPayload, y: number): number {
  return layout.bounds.max_y - y + layout.bounds.min_y;
}

function renderEdge(layout: LayoutPayload, edge: LayoutEdge, nodeById: Map<string, LayoutNode>, occ: number) {
  const from = nodeById.get(edge.from);
  const to = nodeById.get(edge.to);
  if (!from || !to) return null;

  const stroke = congestionColor(occ, edge.capacity);
  const width = occ > 0 ? 0.55 : 0.3;
  return (
    <line
      key={edge.id}
      x1={from.x}
      y1={mapY(layout, from.y)}
      x2={to.x}
      y2={mapY(layout, to.y)}
      stroke={stroke}
      strokeWidth={width}
      strokeLinecap="round"
      opacity={0.9}
    />
  );
}

export function WarehouseMap({ layout, snapshot, selectedAgv }: WarehouseMapProps) {
  if (!layout) {
    return <div className="empty-panel">Waiting for simulation layout...</div>;
  }

  const nodeById = new Map(layout.nodes.map((n) => [n.id, n]));
  const edgeOcc = snapshot?.congestion.edges ?? {};
  const nodeOcc = snapshot?.congestion.nodes ?? {};
  const viewBox = `${layout.bounds.min_x} ${layout.bounds.min_y} ${
    layout.bounds.max_x - layout.bounds.min_x
  } ${layout.bounds.max_y - layout.bounds.min_y}`;

  return (
    <div className="map-panel">
      <svg className="warehouse-svg" viewBox={viewBox}>
        {layout.edges.map((edge) => renderEdge(layout, edge, nodeById, edgeOcc[edge.id] ?? 0))}
        {layout.nodes.map((node) => {
          const occ = nodeOcc[node.id] ?? 0;
          return (
            <circle
              key={node.id}
              cx={node.x}
              cy={mapY(layout, node.y)}
              r={node.type === "INTERSECTION" ? 0.22 : 0.15}
              fill={NODE_COLOR[node.type] ?? "#8e9bad"}
              stroke={congestionColor(occ, node.capacity)}
              strokeWidth={0.08}
              opacity={0.85}
            />
          );
        })}
        {snapshot?.agvs.map((agv) => {
          const isMuted = selectedAgv !== "ALL" && selectedAgv !== agv.id;
          return (
            <circle
              key={agv.id}
              className="agv-dot"
              cx={agv.x}
              cy={mapY(layout, agv.y)}
              r={0.34}
              fill={AGV_COLOR[agv.state] ?? "#ffffff"}
              stroke="#0f1720"
              strokeWidth={0.12}
              opacity={isMuted ? 0.15 : 1}
            />
          );
        })}
      </svg>
    </div>
  );
}

