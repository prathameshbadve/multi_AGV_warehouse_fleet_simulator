# Multi-AGV Warehouse Fleet Simulator and Traffic Control

## Problem Statement

A warehouse operator is planning a new automated fulfillment center. They need to answer the following questions:
1. How many AGVs are required to hit a target throughput (e.g. 500 orders / hour)?
2. Where are the bottlenecks - is it robots, pick stations, aisle congestion, or charging stations?
3. What's the ROI inflection point - at what fleet size does adding more robots stop helping?

## Why Simulation + Optimization

A pure optimization model would give you the "optimal" answer under assumed conditions. But warehouse operations are stochastic: order arrivals are bursty, pick times vary, AGVs break down, and corridors get jammed. Simulation lets you stress-test the optimal answer against realistic chaos.

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    MODULE 4: STREAMLIT APP                  │
│         (Visualization, Controls, KPI Dashboard)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   MODULE 1   │  │   MODULE 2   │  │     MODULE 3     │   │
│  │  Warehouse   │→ │    Task      │→ │   Path Planning  │   │
│  │  Environment │  │  Assignment  │  │   & Conflict     │   │
│  │  & Simulator │  │  (CP-SAT)    │  │   Resolution     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         ↑                                     │             │
│         └─────────── feedback loop ───────────┘             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              SHARED: Data Models & Config                   │
└─────────────────────────────────────────────────────────────┘
```

**Key design principle:** Each module has a clean interface (input/output contract). You can unit-test Module 2 with a mock warehouse. You can demo Module 1 with random-walk AGVs before pathfinding exists.

## Live UI MVP (FastAPI + React)

The project now includes a real-time UI stack that streams simulation state to a browser:

- **Backend:** `FastAPI` + WebSocket endpoint at `/api/simulation/ws`
- **Frontend:** `Vite` + `React` + `TypeScript` app in `ui/`
- **Refresh rate:** configurable, default `10 Hz`
- **Congestion metric:** node occupancy + directed edge occupancy
- **Run model:** one simulation run per WebSocket request, ephemeral (no persistence)

### Run the backend

```bash
python scripts/run_ui_api.py --host 0.0.0.0 --port 8000
```

### Run the frontend

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:5173`.

### WebSocket query params

`/api/simulation/ws?n_agvs=40&hours=0.25&seed=42&strategy=cpsat&hz=10`

- `n_agvs`: fleet size
- `hours`: simulation duration
- `seed`: random seed
- `strategy`: `cpsat` or `greedy`
- `hz`: runtime snapshot frequency

### Stream event types

- `started`: confirms run settings
- `layout`: static warehouse graph (nodes, edges, bounds)
- `snapshot`: live state (AGVs, congestion, metrics)
- `completed`: final KPI summary
- `error`: runtime error payload

## Module 1 - Warehouse Environment & Discrete Event Simulation

### 3.1 Purpose

Model the physical warehouse as a graph, generate stochastic order arrivals, simulate AGV lifecycle (idle -> assigned -> travelling -> picking -> delivering -> returning -> charging), and track time-stepped state.

### 3.2 Warehouse Graph Model

**Representation:** Directed Graph: $G = (V, E)$ using NetworkX.

**Node Types:**
- Intersection
- Storage
- Pick-station
- Charging
- Parking
- Depot

**Edge Attributes:**
- distance
- direction
- capacity
- edge_type: aisle, cross-aisle, highway

**Default Layout:**
```
 [CS] [CS]                            [CS] [CS]    ← Charging Stations
   |    |                              |    |
===H====H======== HIGHWAY =============H====H===
   |    |    |    |    |    |    |     |    |
   S    S    S    S    S    S    S     S    S      ← Storage aisles (one-way)
   S    S    S    S    S    S    S     S    S
   S    S    S    S    S    S    S     S    S
   |    |    |    |    |    |    |     |    |
===H====H======== HIGHWAY =============H====H===
   |    |    |    |    |    |    |     |    |
   S    S    S    S    S    S    S     S    S
   S    S    S    S    S    S    S     S    S
   |    |    |    |    |    |    |     |    |
===H====H======== HIGHWAY =============H====H===
                     |
              [PS][PS][PS][PS]                     ← Pick Stations
```

**MVP Simplification Decisions:**

- Grid-based layout only.
- Aisles are one-way alternating (odd aisles go north, even aisles go south) - this is realistic and simplifies pathfinding.
- Highways (cross-aisles) are two-way with capacity 2.
- All edges have uniform speed; no acceleration / deceleration modelling.

### 3.3 Warehouse Configuration Object
```
WarehouseConfig:
    num_aisles: int = 10
    bays_per_aisle: int = 40
    num_levels: int = 1             # MVP: single level only
    aisle_width: "narrow" | "wide"  # narrow = capacity 1, wide = capacity 2
    num_pick_stations: int = 4
    num_charging_stations: int = 4
    num_parking_spots: int = 15
    highway_capacity: int = 2
    robot_speed_mps: float = 1.5
    pick_station_positions: "south" | "central"  # where pick stations sit
```

### 3.4 AGV State Machine

Each robot is a SimPy process cycling through states:
```
         ┌──────────┐
         │   IDLE   │ ← waiting for task assignment
         └────┬─────┘
              │ task assigned
              ▼
     ┌────────────────┐
     │ TRAVELING_TO   │ ← navigating to pod location
     │ POD            │
     └────────┬───────┘
              │ arrived at pod
              ▼
     ┌────────────────┐
     │  PICKING_UP    │ ← lifting pod (fixed 5s)
     │  POD           │
     └────────┬───────┘
              │ pod secured
              ▼
     ┌────────────────┐
     │ TRAVELING_TO   │ ← carrying pod to pick station
     │ PICK_STATION   │
     └────────┬───────┘
              │ arrived at station
              ▼
     ┌────────────────┐
     │  WAITING_AT    │ ← queuing if station is busy
     │  STATION       │
     └────────┬───────┘
              │ station available
              ▼
     ┌────────────────┐
     │  PROCESSING    │ ← human picks items from pod (stochastic)
     └────────┬───────┘
              │ picking complete
              ▼
     ┌────────────────┐
     │ RETURNING_POD  │ ← carrying pod back to storage
     └────────┬───────┘
              │ pod stored
              ▼
     ┌────────────────┐
     │ CHECK_BATTERY  │─── battery < threshold ──→ TRAVELING_TO_CHARGER
     └────────┬───────┘                                    │
              │ battery OK                                 ▼
              ▼                                    ┌──────────────┐
         ┌──────────┐                              │   CHARGING   │
         │   IDLE   │                              └──────┬───────┘
         └──────────┘                                     │ charged
                                                          ▼
                                                     ┌──────────┐
                                                     │   IDLE   │
                                                     └──────────┘

```

**Robot Attributes:**
- id: unqiue identifier
- state: current state (enum)
- position: current node in graph
- battery_pct: [0, 100] (drain ~0.1% per meter travelled, ~0.5% per pod lift)
- battery_threshold: 20% (robot goes to charging below this)
- current_task: task_id | None
- planned_path: List[nodes, estimated_arrival_time]
- total_distance_travelled: cumulative meters (for utilization tracking)
- tasks_completed: count

**Stochastic Elements in Module 1:**

- Order inter-arrival times (Exponential distribution)
- Items per order (Discrete Uniform or Geometric)
- Pick station processing time (Gamma)
- Pod location for SKU (Weighted random by demand zone)
- AGV breakdown (Exponential)

### 3.5 Order & Task Data Model
```
Order:
    id: str
    arrival_time: float             # simulation time
    items: list[SKU_ID]             # which SKUs needed
    priority: "standard" | "express"
    status: "pending" | "assigned" | "picking" | "complete"

Task:
    id: str
    order_id: str
    pod_location: Node              # where the pod currently sits
    pick_station: Node              # destination
    return_location: Node           # where to return pod after
    assigned_agv: AGV_ID | None
    status: "unassigned" | "assigned" | "in_progress" | "complete"
    created_time: float
    completed_time: float | None
```

**MVP Simplification:**

Each order maps to exactly one pod retrieval task (i.e., assume all items in an order come from one pod). This avoids multi-pod order splitting complexity. Flag this as a known simplification in the README — it's a reasonable MVP scope cut that you can discuss intelligently in an interview ("in production, you'd need a pod selection optimizer that minimizes the number of pods per order, which is itself a set cover problem").

### 3.6 Simulation Clock & Event Loop

**Simulation Engine:** SimPy (process-based discrete event simulation)

**Key Processes:**
1. order_generator(env, config) - generates orders per time-varying Poisson rate.
2. task_dispatcher(env, warehouse, agv_fleet) - periodically (every 5 simulation seconds) checks pending tasks, calls Module 2 for assignment.
3. agv_process(env, agv, warehouse) - one process per AGV, cycles through state machine.
4. pick_station_process(env, station) - models station as a SimPy resource with capcaity 1.
5. charging_station_process(env, station) - SimPy resource, charging time proportional to deficit.
6. metrics_collector(env, interval=60) - snapshot KPIs every simulated minute.

**Simulation Duration:** Default 4 hours of simulated time (enough to see steady-state behaviour and peak effects.)

### 3.7 Module 1 Output Contract

Module 1 produces at each dispatch interval:
```
DispatchRequest:
    current_time: float
    pending_tasks: list[Task]           # unassigned tasks
    available_agvs: list[AGV]           # idle AGVs with position + battery
    occupied_edges: dict[Edge, list[AGV_ID]]  # current traffic state
    station_queues: dict[Station_ID, int]     # queue lengths
```
