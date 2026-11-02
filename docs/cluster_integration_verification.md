# SURFACE Multi-Process Integration Verification (M4b Phase 4.2)

This document describes how to execute the real multi-process end-to-end integration verification harness for the SURFACE distributed inspection cluster.

---

## 1. Overview and Architecture

The verification harness (`scripts/verify_cluster_e2e.py`) launches genuinely separate operating-system processes:

```
  ┌────────────────────────────────────────────────────────────────┐
  │                 verify_cluster_e2e.py                          │
  │  Orchestration, Process Group Management, Log Parsing & E2E    │
  └──────────────┬──────────────────┬──────────────────┬───────────┘
                 │                  │                  │
                 ▼ (PID 1)          ▼ (PID 2)          ▼ (PID 3)
         ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
         │ Redis Server  │  │ FastAPI Master│  │ Celery Worker │
         │  Port 6389    │  │  Port 8008    │  │  --pool=solo  │
         └───────────────┘  └───────────────┘  └───────────────┘
```

### Key Differences from Unit/Simulated Tests:
* **No `task_always_eager=True`**: Celery tasks are published to a real Redis broker queue over TCP.
* **No In-Process Simulation**: Worker processes run in separate OS processes and communicate with the Master exclusively via HTTP REST endpoints.
* **Isolated Environment**: Runs on dedicated ports (`6389` for Redis, `8008` for Master API) to prevent collisions with existing development services.
* **Verifiable HTTP Logging**: Parses actual Uvicorn access logs to prove genuine network transfers for:
  - Input downloads (`stereo_left.png`, `stereo_right.png`)
  - Intermediate uploads (`rectified_left.png`, `disparity.npy`, `mesh.ply`, `point_cloud.ply`)
  - Tile batch downloads (`rectified_left.png`, `deviations.json`)
  - Batch commit transactions (`/batches/{b}/commit`)
  - Finding crop uploads (`findings/*.png`)

---

## 2. Prerequisites

1. **Python Virtual Environment**:
   - Must contain project dependencies (`fastapi`, `uvicorn`, `celery`, `redis`, `httpx`, `torch`, `cv2`, `open3d`).
2. **Redis Server Binary**:
   - `redis-server` must be installed (e.g. via Homebrew: `brew install redis` or apt: `apt install redis-server`).
3. **Repository Sample Data & Model Weights**:
   - `public/samples/stereo_left.png`
   - `public/samples/stereo_right.png`
   - `public/samples/camera_calibration_sample.json`
   - `backend/model_store/best_model_sdnet2018.pth`

---

## 3. Running the Integration Test

### Basic Multi-Process Integration Test:
```bash
./backend/venv/bin/python scripts/verify_cluster_e2e.py
```

### Running with Recovery Verification (Two-Worker Takeover):
```bash
./backend/venv/bin/python scripts/verify_cluster_e2e.py --run-recovery
```

### Custom Ports and Output Directory:
```bash
./backend/venv/bin/python scripts/verify_cluster_e2e.py \
  --redis-port 6390 \
  --master-port 8010 \
  --output-dir /tmp/surface_e2e_run \
  --timeout 120
```

---

## 4. Output and Artifacts

Upon completion, the harness displays a structured verification table and writes detailed logs:
* `<output_dir>/e2e_report.json`: Comprehensive machine-readable report containing exact PIDs, stage durations, batch counts, HTTP transfer metrics, and assertion statuses.
* `<output_dir>/logs/redis_server.log`: Dedicated Redis broker log.
* `<output_dir>/logs/fastapi_master.log`: Master Uvicorn server log with HTTP access records.
* `<output_dir>/logs/celery_worker_1.log`: Primary worker execution and inference log.
* `<output_dir>/logs/celery_worker_2.log`: Secondary worker log (when `--run-recovery` is enabled).

---

## 5. Teardown Guarantees

All subprocesses are launched in independent process groups. The harness uses a registered `atexit` hook and signal handlers to ensure:
* Subprocesses are gracefully terminated with `SIGTERM`.
* Any stubborn process is forcibly terminated with `SIGKILL` after a 3-second grace period.
* No orphan Uvicorn or Celery processes remain running.
