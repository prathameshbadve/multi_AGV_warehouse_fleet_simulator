import type { FinalMetricsPayload, SnapshotPayload } from "../types";

interface MetricsPanelProps {
  snapshot: SnapshotPayload | null;
  finalMetrics: FinalMetricsPayload | null;
}

function formatNumber(value: number, digits = 1): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

export function MetricsPanel({ snapshot, finalMetrics }: MetricsPanelProps) {
  const live = snapshot?.metrics;
  const stationQueueTotal = live
    ? Object.values(live.station_queues).reduce((acc, n) => acc + n, 0)
    : 0;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <span className="metric-label">Sim Time</span>
        <span className="metric-value">{formatNumber(snapshot?.time_s ?? 0, 1)} s</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Throughput</span>
        <span className="metric-value">{formatNumber(live?.throughput_per_hour ?? 0, 1)} /hr</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">AGV Utilization</span>
        <span className="metric-value">{formatNumber(live?.agv_utilization_pct ?? 0, 1)}%</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Orders Pending</span>
        <span className="metric-value">{live?.orders_pending ?? 0}</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Unassigned Tasks</span>
        <span className="metric-value">{live?.tasks_unassigned ?? 0}</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Station Queue</span>
        <span className="metric-value">{stationQueueTotal}</span>
      </div>

      {finalMetrics ? (
        <div className="final-metrics">
          <div>Final Throughput: {formatNumber(finalMetrics.avg_throughput_per_hour, 1)} /hr</div>
          <div>Avg Cycle Time: {formatNumber(finalMetrics.avg_cycle_time_s, 1)} s</div>
          <div>P95 Cycle Time: {formatNumber(finalMetrics.p95_cycle_time_s, 1)} s</div>
          <div>Distance Traveled: {formatNumber(finalMetrics.total_distance_traveled_m, 0)} m</div>
        </div>
      ) : null}
    </div>
  );
}

