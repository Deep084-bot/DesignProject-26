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

### Running with Secondary Worker Test:
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

---

## 6. Verification Results and Empirical Claims

### 6.1 Backend Test Suite Breakdown

All 147 backend tests pass cleanly (`147 passed, 2 warnings in 42.18s`):

| Test Module | File | Tests | Status |
|---|---|---|---|
| API Inspections | `backend/tests/test_api_inspections.py` | 10 | PASS |
| Calibration Validation | `backend/tests/test_calibration.py` | 6 | PASS |
| Cluster Preflight | `backend/tests/test_cluster_preflight.py` | 11 | PASS |
| Damage Classification | `backend/tests/test_damage.py` | 18 | PASS |
| Deviation Analysis | `backend/tests/test_deviations.py` | 12 | PASS |
| Distributed Coordination | `backend/tests/test_distributed.py` | **69** | PASS |
| Mesh Reconstruction | `backend/tests/test_mesh_reconstruction.py` | 9 | PASS |
| Stereo Pipeline | `backend/tests/test_stereo_pipeline.py` | 6 | PASS |
| E2E Harness Unit Tests | `backend/tests/test_verify_cluster_e2e.py` | 6 | PASS |
| **Total** | | **147** | **PASS (100%)** |

*(Note: Earlier Phase 4.2 documentation mistakenly listed `test_distributed.py` as 65 tests due to a transcription typo; the exact collected and passing count is 69 tests, totaling 147 tests across the backend suite.)*

### 6.2 Nature of Batch Execution (Sequential vs Concurrent)

During the multi-process integration run:
* The Celery worker was executed with `--pool=solo` (single-threaded concurrency 1).
* The two tile batches (Batch 0: tiles 0–7; Batch 1: tiles 8–15) were executed **sequentially** by the worker process:
  - Batch 0 was received at `09:40:58`, claimed, executed, and committed at `09:40:59`.
  - Batch 1 was received at `09:40:59`, claimed, executed, and committed at `09:41:00`.
* **Important Claim Clarification:** This run proves genuine Celery chord fanout, batch partitioning, HTTP artifact transfers, batch claiming, and batch committing against the master API. However, because the solo worker executed these batches serially, this test **does not demonstrate concurrent batch processing or parallel speedup**. True concurrent batch processing requires deploying multiple worker nodes or worker processes with `--pool=prefork` concurrency $\ge 2$.

### 6.3 Secondary Worker Execution Scope (`--run-recovery`)

The `--run-recovery` flag tests cluster behavior with two active worker processes (`worker-e2e-node1` and `worker-e2e-node2`):
* When the secondary inspection (`INSP-2026-6D9A87`) was submitted to the shared Redis broker queue, Celery initially routed `run_inspection_task` directly to `worker-e2e-node2`.
* `worker-e2e-node1` was stopped gracefully via `SIGTERM` while in an idle state.
* `worker-e2e-node2` proceeded to execute and complete Stages 0 through 5 under its initial lease assignment (`lease_token: 1`, `finalizer_generation: 0`).
* **Important Recovery Limitation:** **No mid-flight lease takeover or stale-worker fencing was demonstrated by this particular multi-process run.** Worker 1 did not crash while actively executing a lease, and Worker 2 did not perform a CAS lease reassignment from 1 to 2.
* While mid-flight failure, lease expiration, worker fencing, outbox write-ahead logging, and supervisor recovery are thoroughly verified in the in-process distributed test suite (`backend/tests/test_distributed.py`, 69 tests), unit- and application-level test coverage is **not equivalent to a genuine OS-process failure integration test**. Genuine mid-flight OS process recovery remains a target for subsequent cluster validation on live VMs.

