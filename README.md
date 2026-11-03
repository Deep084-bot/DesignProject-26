# Surface — Distributed 3D Structural Surface Processing System

Surface is a 3D structural inspection platform designed to reconstruct physical surfaces from calibrated stereo photographs, measure metric geometric deviations, and help engineers identify potential cracks and structural surface irregularities.

---

## 1. Project Overview & Workflow

Surface operates through a natural, end-to-end inspection journey:

$$\text{Upload stereo pair} \longrightarrow \text{Validate camera parameters} \longrightarrow \text{Reconstruct 3D surface} \longrightarrow \text{Analyze deviations} \longrightarrow \text{Classify damage} \longrightarrow \text{Inspect findings}$$

1. **Upload & Calibration**: Ingests left and right stereo camera photographs alongside camera calibration (baseline distance and focal length in JSON format).
2. **Epipolar Alignment & Rectification**: Corrects optical distortions and aligns perspective scanlines ($y_L = y_R$).
3. **Disparity Estimation**: Measures sub-pixel horizontal displacement between stereo views using OpenCV Semi-Global Block Matching (StereoSGBM).
4. **Metric 3D Triangulation & Mesh**: Triangulates physical surface coordinates in millimeters ($X, Y, Z$) and reconstructs surface meshes via Open3D screened Poisson reconstruction.
5. **Geometric Deviation Analysis**: Fits an initial robust reference plane via RANSAC and iteratively refines via least-squares (SVD) on inliers. Computes signed out-of-plane deviations in millimeters ($\text{depression} > 0$, $\text{bulge} < 0$).
6. **Damage Classification & Tile Fanout**: Evaluates image crops using a fine-tuned EfficientNet-B0 binary crack classifier, grouping positive detections and spatially correlating them with measured depth anomalies.
7. **Interactive 3D Review**: Explores the metric PLY point cloud and mesh in a Three.js WebGL viewer with OrbitControls, deviation heatmap overlay, and localized crack bounding boxes.

---

## 2. Pipeline Implementation Status

| Stage ID | Pipeline Stage | Implementation Status | Technology Stack | Operational Limits |
|---|---|---|---|---|
| `prepare` | Stereo Validation & Rectification | **Implemented** | OpenCV `stereoRectify` | Requires identical left/right image dimensions |
| `depth` | Disparity & Depth Estimation | **Implemented** | OpenCV StereoSGBM | Sub-pixel block matching; sensitive to untextured regions |
| `surface` | 3D Triangulation & Mesh Generation | **Implemented** | Pinhole triangulation ($Z = \frac{f_x \cdot B}{d}$) & Open3D Poisson Reconstruction | Configurable octree depth (`SURFACE_MESH_DEPTH`, default 8) |
| `deviations`| Out-of-Plane Geometric Deviation Analysis | **Implemented** | RANSAC plane fitting + MAD-based robust thresholding ($\sigma_{\text{MAD}}$) | Identifies depressions/bulges; does not prove structural damage |
| `damage` | Deep Learning Crack Detection & Finding Merging | **Implemented** | PyTorch EfficientNet-B0 + Celery chord tile fanout | High recall on concrete textures; fallback modes for missing weights |
| `results` | Artifact Manifest & Metric Report Delivery | **Implemented** | FastAPI JSON manifest, binary PLY, visual heatmaps & finding crops | Retains full audit trail and step durations in `inspection.json` |

> [!NOTE]
> **Engineering Scope:** Surface provides geometric measurements and localized machine learning indicators. An out-of-plane depression or fissure detection does not constitute certified structural failure without on-site civil engineering validation.

---

## 3. Architecture & Execution Modes

Surface supports two execution modes configured via `SURFACE_EXECUTION`:

### 3.1 Local Execution Mode (`SURFACE_EXECUTION=local`)
- **Default mode**: No Redis, Celery, or background worker required.
- The pipeline executes within an asynchronous Python `ThreadPoolExecutor` on the master server.
- Ideal for standalone single-laptop demonstration, unit testing, and offline evaluation.

### 3.2 Distributed Celery Execution Mode (`SURFACE_EXECUTION=celery`)
- The FastAPI Master server receives uploads, persists initial artifacts, and enqueues tasks to a Redis message broker.
- **Worker Isolation**: Workers execute in separate operating system processes (or on separate laptops) and communicate with the Master exclusively over authenticated HTTP REST endpoints (`X-Worker-Token`).
- **Tile Fanout (M4b)**: Stages 0–3 execute on a coordinator worker. The coordinator partitions damage classification into tile batches and dispatches a Celery `chord` across all available workers. A designated finalizer aggregates probabilities and writes findings.
- **Fencing & Crash Recovery**:
  - Compare-And-Swap (CAS) lease generation tokens fence against network partitioning.
  - Write-ahead recovery outbox logs on disk guarantee crash consistency across master restarts.
  - Heartbeat threads (`WorkerHeartbeatThread`, `FinalizerHeartbeatThread`) issue immediate initial pulses and periodic pulses to prevent premature failover during CPU-bound mesh reconstruction.

```
┌────────────────────────────────────────────────────────┐
│                   React Web Client                     │
│               http://localhost:5173                    │
└───────────────────────────┬────────────────────────────┘
                            │ (HTTP REST)
                            ▼
┌────────────────────────────────────────────────────────┐
│                  FastAPI Master Server                 │
│         /api/inspections  |  /api/internal/*           │
│                   Recovery Supervisor                  │
└─────────────┬────────────────────────────▲─────────────┘
              │ (Task Publication)         │ (Artifact Uploads,
              ▼                            │  Steps, Heartbeats)
     ┌────────────────┐                    │
     │  Redis Broker  │                    │
     └────────┬───────┘                    │
              │ (Task Consumption)         │
              ▼                            │
┌───────────────────────────┐    ┌─────────┴─────────────────┐
│     Celery Worker 1       │    │      Celery Worker 2      │
│     (--pool=solo)         │    │      (--pool=solo)        │
│  Stages 0-3 Coordinator   │    │  Batch Tile Classification│
└───────────────────────────┘    └───────────────────────────┘
```

For complete instructions on deploying the cluster across four laptops for live demonstration, see the [Demo Runbook](docs/RUNBOOK.md).

---

## 4. Local Setup & Quickstart

### Prerequisites
- **macOS** (Apple Silicon or Intel), Linux, or Windows 11
- **Node.js** 18+ and `npm`
- **Python** 3.10, 3.11, or 3.12
- **Redis** (optional, required only for distributed Celery mode)

### 4.1 Frontend Setup
```bash
# Install frontend dependencies
npm install

# Start Vite development server
npm run dev
# Accessible at http://localhost:5173
```

### 4.2 Backend Setup (Local Mode)
```bash
# Create and activate virtual environment
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Launch FastAPI master in local thread-pool mode
export SURFACE_EXECUTION=local
./backend/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

### 4.3 Environment Configuration
Copy `.env.example` to `.env` in the project root:
```ini
# When set, the React UI connects to the real FastAPI backend
VITE_API_BASE_URL=http://localhost:8000
```

---

## 5. Verification & Tests

### 5.1 Backend Regression Suite
Run the full backend test suite:
```bash
PYTHONPATH=. ./backend/venv/bin/pytest backend/tests/ -v
```

> [!NOTE]
> **Known Test Suite Status (180 passed, 1 limitation):**
> 180 out of 181 backend tests pass. A single test (`test_heartbeat_http_failure_does_not_kill_worker_task` in `test_recovery_reliability.py`) exhibits a known timeout limitation on macOS: Apple's `mDNSResponder` takes 5.0 seconds to resolve unresolvable `.local` domains, whereas the test fixture enforces a 1.0-second join timeout. This does not affect production execution on loopback or standard IPv4 networks.

### 5.2 Frontend Build & Lint
```bash
# Run production TypeScript build
npm run build

# Run linter
npm run lint
```

### 5.3 Cluster Benchmark
Run the automated cluster benchmark suite against a running master:
```bash
python scripts/benchmark_cluster.py --api http://127.0.0.1:8000 --jobs 8
```
Results and throughput metrics are automatically recorded in `benchmarks/results.csv`.

---

## 6. Empirical Verification Boundaries

- **Multi-Process Local Verification**: The distributed coordinator recovery, tile fanout chord, and finalizer delegation mechanisms have been verified on a local multi-process deployment on Apple Silicon macOS (FastAPI master + 2 Celery workers + Redis server).
- **Physical Multi-Machine Testing**: Testing across four physically discrete laptops over LAN Wi-Fi has not yet been executed in a formal lab environment. Operational procedures, network firewall requirements, and fallback options are detailed in [docs/RUNBOOK.md](docs/RUNBOOK.md).
- **Honest Telemetry**: Finding confidence scores and geometric deviations reflect authentic algorithmic outputs. No artificial findings or benchmark timings are fabricated.
