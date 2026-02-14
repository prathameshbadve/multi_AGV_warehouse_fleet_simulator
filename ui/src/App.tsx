import { useMemo, useRef, useState } from "react";

import { MetricsPanel } from "./components/MetricsPanel";
import { WarehouseMap } from "./components/WarehouseMap";
import type { FinalMetricsPayload, LayoutPayload, SnapshotPayload, WsEvent } from "./types";

type RunStatus = "idle" | "running" | "completed" | "error";

interface RunConfig {
  nAgvs: number;
  hours: number;
  seed: number;
  strategy: "cpsat" | "greedy";
  hz: number;
}

const DEFAULT_CONFIG: RunConfig = {
  nAgvs: 40,
  hours: 0.25,
  seed: 42,
  strategy: "cpsat",
  hz: 10
};

function wsBaseUrl(): string {
  const envUrl = import.meta.env.VITE_WS_URL as string | undefined;
  if (envUrl) return envUrl;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://localhost:8000/api/simulation/ws`;
}

function buildWsUrl(config: RunConfig): string {
  const url = new URL(wsBaseUrl(), window.location.href);
  url.searchParams.set("n_agvs", String(config.nAgvs));
  url.searchParams.set("hours", String(config.hours));
  url.searchParams.set("seed", String(config.seed));
  url.searchParams.set("strategy", config.strategy);
  url.searchParams.set("hz", String(config.hz));
  return url.toString();
}

export default function App() {
  const [runConfig, setRunConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [layout, setLayout] = useState<LayoutPayload | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotPayload | null>(null);
  const [finalMetrics, setFinalMetrics] = useState<FinalMetricsPayload | null>(null);
  const [selectedAgv, setSelectedAgv] = useState<string>("ALL");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);

  const agvOptions = useMemo(() => {
    if (!snapshot) return [];
    return snapshot.agvs.map((agv) => agv.id);
  }, [snapshot]);

  const connect = () => {
    if (status === "running") return;
    wsRef.current?.close();

    setStatus("running");
    setLayout(null);
    setSnapshot(null);
    setFinalMetrics(null);
    setErrorMessage("");
    setSelectedAgv("ALL");

    const ws = new WebSocket(buildWsUrl(runConfig));
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as WsEvent<any>;
      if (data.type === "layout") {
        setLayout(data.payload as LayoutPayload);
      } else if (data.type === "snapshot") {
        setSnapshot(data.payload as SnapshotPayload);
      } else if (data.type === "completed") {
        setFinalMetrics((data.payload as { metrics: FinalMetricsPayload }).metrics);
        setStatus("completed");
      } else if (data.type === "error") {
        const payload = data.payload as { message?: string };
        setErrorMessage(payload.message ?? "Unknown backend error");
        setStatus("error");
      }
    };

    ws.onerror = () => {
      setErrorMessage("WebSocket connection failed");
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === "running" ? "idle" : prev));
    };
  };

  const stop = () => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("idle");
  };

  return (
    <main className="page">
      <section className="topbar">
        <div className="title-wrap">
          <h1>AGV Fleet Live Dashboard</h1>
          <p>10 Hz simulation stream with edge/node congestion heat.</p>
        </div>
        <div className="controls">
          <label>
            AGVs
            <input
              type="number"
              min={1}
              value={runConfig.nAgvs}
              onChange={(e) => setRunConfig({ ...runConfig, nAgvs: Number(e.target.value) })}
            />
          </label>
          <label>
            Hours
            <input
              type="number"
              step={0.05}
              min={0.05}
              value={runConfig.hours}
              onChange={(e) => setRunConfig({ ...runConfig, hours: Number(e.target.value) })}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              value={runConfig.seed}
              onChange={(e) => setRunConfig({ ...runConfig, seed: Number(e.target.value) })}
            />
          </label>
          <label>
            Solver
            <select
              value={runConfig.strategy}
              onChange={(e) =>
                setRunConfig({ ...runConfig, strategy: e.target.value as "cpsat" | "greedy" })
              }
            >
              <option value="cpsat">cpsat</option>
              <option value="greedy">greedy</option>
            </select>
          </label>
          <button className="btn-primary" onClick={connect} disabled={status === "running"}>
            Run
          </button>
          <button className="btn-secondary" onClick={stop} disabled={status !== "running"}>
            Stop
          </button>
        </div>
      </section>

      <section className="content">
        <div className="left-column">
          <div className="filter-row">
            <label>
              AGV Filter
              <select value={selectedAgv} onChange={(e) => setSelectedAgv(e.target.value)}>
                <option value="ALL">All AGVs</option>
                {agvOptions.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>
            <div className={`status-pill status-${status}`}>{status.toUpperCase()}</div>
          </div>
          <WarehouseMap layout={layout} snapshot={snapshot} selectedAgv={selectedAgv} />
        </div>
        <aside className="right-column">
          <MetricsPanel snapshot={snapshot} finalMetrics={finalMetrics} />
          {errorMessage ? <div className="error-box">{errorMessage}</div> : null}
          <div className="legend">
            <div>Congestion Colors</div>
            <div className="legend-row">
              <span className="legend-dot good" />
              Low
            </div>
            <div className="legend-row">
              <span className="legend-dot medium" />
              Medium
            </div>
            <div className="legend-row">
              <span className="legend-dot high" />
              High
            </div>
          </div>
        </aside>
      </section>
    </main>
  );
}
