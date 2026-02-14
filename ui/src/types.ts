export type EventType = "started" | "layout" | "snapshot" | "completed" | "error";

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  type: string;
  capacity: number;
}

export interface LayoutEdge {
  id: string;
  from: string;
  to: string;
  type: string;
  capacity: number;
  distance_m: number;
}

export interface LayoutPayload {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
  bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  };
}

export interface AgvLiveState {
  id: string;
  state: string;
  node_id: string;
  battery_pct: number;
  task_id: string | null;
  x: number;
  y: number;
  current_edge_id: string | null;
}

export interface SnapshotPayload {
  time_s: number;
  final: boolean;
  progress_pct: number;
  agvs: AgvLiveState[];
  congestion: {
    nodes: Record<string, number>;
    edges: Record<string, number>;
  };
  metrics: {
    orders_generated: number;
    orders_completed: number;
    orders_pending: number;
    tasks_unassigned: number;
    tasks_active: number;
    agv_utilization_pct: number;
    throughput_per_hour: number;
    agv_states: Record<string, number>;
    station_queues: Record<string, number>;
  };
}

export interface FinalMetricsPayload {
  total_orders_generated: number;
  total_orders_completed: number;
  total_simulation_time_s: number;
  avg_throughput_per_hour: number;
  avg_cycle_time_s: number;
  p95_cycle_time_s: number;
  avg_agv_utilization_pct: number;
  avg_station_utilization_pct: number;
  total_distance_traveled_m: number;
}

export interface WsEvent<T = unknown> {
  type: EventType;
  payload: T;
}

