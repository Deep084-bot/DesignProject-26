# SURFACE MASTER BUILD SPEC (read fully before doing anything)

You are the lead engineer finishing the project **Surface: Distributed 3D Structural Surface Processing System**
(repo: DesignProject-26). A student team demos it to a professor on 4 laptops. Follow this file exactly.
Work in **milestones M0 to M5**. After each milestone you must STOP, report (format in section 9) and wait for
the user to type `continue M<next>`. Never start the next milestone on your own.

---------------------------------------------------------------------
## 0. Operating rules (non-negotiable)

1. If anything in this spec conflicts with the real code, a file is missing, or something is ambiguous: STOP and ask. Do not guess or invent.
2. Do NOT modify `StereoImage_to_pointCloud/` or `PointCloudToMesh/` (protected team code). Importing from them is fine.
3. Do not rewrite or reformat existing working code. Make the smallest change that satisfies the spec. Explain every file you touch.
4. **Never fabricate results.** No fake findings, fake progress timers, fake worker names, or hard-coded demo numbers. If nothing is detected, the findings list is `[]`.
5. Only free/open-source tools. No paid APIs, no cloud accounts. Everything runs on a local network.
6. Units are **millimetres**. Camera frame: +X right, +Y down, +Z forward. Viewer (Three.js) frame: `(x, -y, -z)` of the camera frame.
7. All existing tests (currently **27 passing** in `backend/tests/`) must keep passing. Every new module needs its own tests. Run the full suite at the end of every milestone.
8. Default behaviour must stay backward compatible: with no new env vars set, the app runs exactly as it does today (local thread pool, no Celery, no model required to start).
9. New response fields must be OPTIONAL with defaults so the existing frontend and tests do not break.
10. Work on git branch `feature/m2-deviation-damage-distributed`. One commit per milestone with a clear message. Never push to `main`.
11. Code style: Python 3.10-3.12, type hints, short docstrings that explain WHY, small functions, no dead code. Plain-language comments (the team is learning).
12. Do not add features that are not in this spec.

---------------------------------------------------------------------
## 1. What the system does (end to end)

User uploads a LEFT photo, a RIGHT photo and a calibration JSON in the React website.
Backend pipeline (6 stages, ids in `backend/app/models/inspection.py` DEFAULT_STAGES):

| idx | id | what it does | status today |
|---|---|---|---|
| 0 | prepare | decode, validate, rectify images | done |
| 1 | depth | OpenCV StereoSGBM disparity | done |
| 2 | surface | triangulate point cloud PLY + Poisson mesh PLY | done |
| 3 | deviations | fit reference plane, measure deviation in mm, find regions | **BUILD (M1)** |
| 4 | damage | classify image tiles with EfficientNet-B0 crack model, combine with deviations, produce findings | **BUILD (M2)** |
| 5 | results | finish, timings | done |

Website shows progress, a 3D viewer (point cloud/mesh), the left image with finding boxes, and a Nodes page showing the
worker laptops. Distribution: master laptop runs Redis + FastAPI + storage; worker laptops run Celery workers that
execute inspections; a 4th laptop shows the website for the demo.

Existing key files (verify they exist before editing): `backend/app/pipeline/runner.py` (`run_pipeline`,
`PipelineExecutionResult`), `backend/app/services/job_manager.py` (`JobManager`, thread pool, `job_manager`),
`backend/app/routes/inspections.py`, `backend/app/models/inspection.py` (`Finding`, `InspectionResponse`, `PipelineStep`),
`backend/app/pipeline/depth.py`, `stereo.py`, `rectification.py`, `mesh_reconstruction.py`,
`src/types/inspection.ts`, `src/services/api.ts`, `src/components/viewer/ThreeSurfaceViewer.tsx`, `src/pages/*`.

---------------------------------------------------------------------
## 2. Shared contracts

### 2.1 Job folder (per inspection id), created by the master or a worker temp dir
`input_left.<ext>`, `input_right.<ext>`, `calibration.json`, `rectified_left.png`, `rectified_right.png`,
`disparity.npy`, `disparity_vis.png`, `pointcloud.ply`, `mesh.ply`,
`deviation_map.npy`, `deviation_heatmap.png`, `deviations.json`, `finding_01.png` (one crop per finding), `inspection.json`.

### 2.2 Region (output of deviation analysis, in `deviations.json` and in memory)
```
{ "id": 1,
  "bbox_px": {"x": int, "y": int, "width": int, "height": int},   # PIXELS of rectified_left.png
  "area_px": int, "max_abs_deviation_mm": float, "mean_deviation_mm": float,
  "sign": "depression" | "bulge",
  "centroid_px": [float, float],
  "centroid_3d_viewer": [float, float, float] }                   # viewer frame, mm
```

### 2.3 Finding (existing schema, use it as is)
`id` like `"Finding 01"`; `type` in {potential_crack, surface_irregularity, minor_anomaly};
`confidence` = classifier probability (None if no classifier evidence); `geometric_deviation_mm` (None if no deviation);
`bounding_box_2d` = `{x,y,width,height}` in **pixels of rectified_left.png** (NOT normalized);
`region_center_3d` viewer frame mm or None; `image_crop_url` = `/api/inspections/{id}/artifacts/finding_XX.png`.

### 2.4 New OPTIONAL fields (add with defaults; mirror in `src/types/inspection.ts` and the normaliser in `src/services/api.ts`)
- `InspectionResponse`: `deviation_summary: Optional[Dict[str, Any]]`, `deviation_heatmap_url: Optional[str]`,
  `classifier_status: Optional[str]` ("completed" | "unavailable" | "failed"), `classifier_error: Optional[str]`.
- `PipelineStep`: `worker: Optional[str]`, `duration_sec: Optional[float]`.
- `deviation_summary` keys: `rmse_mm, max_depression_mm, max_bulge_mm, sigma_mm, threshold_mm, inlier_fraction, region_count, plane_normal, plane_offset_mm, valid_pixel_fraction`.

### 2.5 Central settings
Create `backend/app/settings.py` reading env vars with these defaults (single place, imported everywhere):

| env var | default | meaning |
|---|---|---|
| `SURFACE_EXECUTION` | `local` | `local` or `celery` |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `SURFACE_MASTER_URL` | `http://localhost:8000` | where workers reach the master API |
| `SURFACE_WORKER_TOKEN` | (empty) | shared secret for `/api/internal/*` |
| `SURFACE_WORKER_NAME` | hostname | shown in UI |
| `SURFACE_CORS_ORIGINS` | (empty) | extra comma-separated allowed origins, appended to existing list |
| `SURFACE_MODEL_PATH` | `backend/models_store/best_model_sdnet2018.pth` | crack classifier checkpoint |
| `SURFACE_CRACK_THRESHOLD` | `0.5` | P(cracked) needed to count a tile |
| `SURFACE_TILE_SIZE` | `256` | tile side in pixels |
| `SURFACE_DEV_K` | `3.0` | noise multiplier for deviation threshold |
| `SURFACE_DEV_MIN_MM` | `5.0` | minimum deviation threshold in mm |
| `SURFACE_DEV_MIN_AREA_PX` | `200` | smallest region kept |
| `SURFACE_MESH_DEPTH` | `8` | Poisson depth passed to the mesh stage |

---------------------------------------------------------------------
## M0. Baseline check (no feature code)

1. Create the branch. Install backend deps in `backend/venv`. Run `pytest backend/tests -v` (expect 27 pass).
2. Write `scripts/run_sample_pipeline.py`: runs `run_pipeline` on `public/samples/stereo_left.png`, `stereo_right.png`,
   `camera_calibration_sample.json` into a temp dir; prints per-stage seconds, point count, mesh status, peak memory (use `resource`/`psutil` only if already installed; otherwise skip memory).
3. Make `mesh_reconstruction` read the Poisson depth from `SURFACE_MESH_DEPTH` (default 8, behaviour unchanged). Reason: the full run was killed once by memory in a limited environment; the team must be able to lower it.
4. Run the frontend: `npm install`, `npx oxlint`, `npm run build`. Report the results exactly.
GATE: report test counts, timings, any failure. Do not fix unrelated problems; list them.

---------------------------------------------------------------------
## M1. Geometric deviation analysis (stage 3)

Create `backend/app/pipeline/deviations.py` with
`analyze_deviations(disparity, baseline_mm, fx, fy, cx, cy, left_bgr=None, min_disparity=0.5, k=3.0, min_mm=5.0, min_area_px=200) -> DeviationResult`.

Algorithm (work on the DENSE GRID so pixel positions are kept, not on the unordered cloud):
1. `valid = isfinite(d) & d > min_disparity`, `Z = fx*B/d`, keep `50 <= Z <= 50000`. If valid pixels < 10% of the image: raise a clear `ValueError` (stage fails with message).
2. Back-project to camera frame: `X=(u-cx)Z/fx`, `Y=(v-cy)Z/fy`.
3. Robust plane fit: random subsample up to 50,000 valid points; RANSAC (200 iterations, fixed seed for reproducibility) to initialise; then up to 5 rounds of least squares (SVD) on inliers where `|r| < 2.5 * sigma_MAD`. Orient the normal so `n_z > 0`. Signed distance `s = (P - p0) . n` in mm: **positive = farther than the plane = depression; negative = bulge**.
4. `sigma = 1.4826 * median(|s - median(s)|)` over valid pixels. `threshold = max(k*sigma, min_mm)`.
5. `mask = |s| > threshold` on the grid. Clean: morphological open 3x3 then close 5x5. Connected components (8-connectivity). Drop regions with area < `min_area_px`. Keep at most 50 regions, ranked by max deviation.
6. For each region compute the Region fields in 2.2 (centroid_3d from the mean 3D point of the region, converted to the viewer frame).
7. Save: `deviation_map.npy` (float32 HxW, NaN invalid), `deviation_heatmap.png` (diverging colormap, blue = bulge, red = depression, symmetric clip at 3*threshold, invalid pixels transparent/black, region outlines optional), `deviations.json` (summary + regions).
8. Wire into `runner.py` stage index 3 using the disparity and rectified parameters already available. Extend `PipelineExecutionResult` with `deviation_summary`, `deviation_regions`, `deviation_heatmap_path`. If analysis raises, report stage 3 `failed` with the message and return `success=False` like other stages.
9. `JobManager`: set `job.deviation_summary` and `job.deviation_heatmap_url`. Do NOT create findings here. `job.findings` stays `[]` until M2.
10. Do NOT modify `ply_writer.py` or the PLY content.

Tests (`backend/tests/test_deviations.py`): (a) synthetic tilted plane + known 20 mm square depression (build disparity analytically) is detected, bbox within 5 px, max deviation within 15% of 20 mm, sign = depression; (b) a bulge has sign = bulge; (c) perfectly flat plane gives 0 regions; (d) plane + Gaussian noise (sigma 0.3 px disparity) gives 0 regions at defaults; (e) mostly-invalid disparity raises ValueError; (f) running the full pipeline on the repo sample pair completes and writes the three artifacts.
GATE: also run the sample pair and report `deviation_summary`, region count, and save `deviation_heatmap.png`. State honestly if the sample produces regions that look like noise.

---------------------------------------------------------------------
## M2. Crack classifier and findings (stage 4)

Create `backend/app/pipeline/damage.py`. Add `torch` and `torchvision` to `backend/requirements.txt` (CPU build is fine).

Classifier:
- Checkpoint at `SURFACE_MODEL_PATH` is a dict with keys `model_name`, `state_dict`, `classes` (`["Uncracked","Cracked"]`). Build `torchvision.models.efficientnet_b0(weights=None)`, replace `classifier[1]` with `Linear(in_features, 2)`, load the state dict, `eval()`.
- Preprocess exactly: BGR->RGB, resize to 224x224, to tensor in [0,1], normalise mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`. Softmax; "Cracked" = index 1.
- Device: CUDA if available, else MPS, else CPU. Load once (lazy singleton). `torch.no_grad()`. Batch size 16.
- Do NOT use OpenCV thresholding, edge or ridge detection as a crack detector. Do NOT invent confidence values.

Detection `detect_damage(rectified_left_bgr, deviation_regions, disparity=None, intrinsics=None, tile=SURFACE_TILE_SIZE, threshold=SURFACE_CRACK_THRESHOLD)`:
1. Tiles: a regular grid over the whole image (tile x tile, stride = tile; pad edge tiles by reflecting; images smaller than a tile are resized up).
2. Extra tiles: for each deviation region, a crop centred on its box, padded to at least tile x tile (larger regions get several tiles).
3. Classify all tiles. A tile is positive if `P(cracked) >= threshold`.
4. Merge touching/overlapping positive tiles into groups; group box = union; group confidence = max P in the group.
5. Findings (rules are configurable defaults and must be documented in the code):
   - positive group overlapping a deviation region -> `potential_crack`, description says confirmed by classifier AND depth deviation; confidence = P; geometric_deviation_mm = that region's max_abs; region_center_3d = region centroid_3d_viewer.
   - positive group, no deviation, P >= 0.8 -> `potential_crack`, description says "image evidence only, no measurable depth change"; region_center_3d from median depth inside the box converted to viewer frame (None if no valid depth); geometric_deviation_mm = None.
   - positive group, no deviation, 0.5 <= P < 0.8 -> `minor_anomaly`.
   - deviation region with no positive group -> `surface_irregularity`, confidence None, with its deviation.
6. Save a crop `finding_XX.png` per finding (with margin) and set `image_crop_url`. IDs `Finding 01`, `Finding 02`, ... sorted by severity (deviation, then confidence).
7. If the model file is missing or fails to load: do NOT crash the job. Produce only `surface_irregularity` findings from deviations, set `classifier_status="unavailable"` and `classifier_error=<message>`. If it loads and runs, `classifier_status="completed"`.
8. Wire into runner stage 4 and `JobManager` (`job.findings`, `classifier_status`, `classifier_error`). Remove the hard-coded `job.findings = []` assignment, replacing it with the real result (still `[]` when nothing is found).

Tests (`backend/tests/test_damage.py`): stub classifier via monkeypatch with deterministic probabilities to cover every finding rule, merging, box coordinates, crop files, ID ordering, and the missing-model fallback; one test loading the real checkpoint is skipped (`pytest.skip`) if the file does not exist; if `backend/tests/assets/crack_sample.png` and `no_crack_sample.png` exist, assert the real model labels them correctly.
GATE: run the sample pair; report findings (it is expected that a synthetic sample produces few or no crack findings; report what it produces, do not tune thresholds to make findings appear).

---------------------------------------------------------------------
## M3. Frontend

1. Update `src/types/inspection.ts` and the normaliser in `src/services/api.ts` for the optional fields (2.4), mapping snake_case to camelCase.
2. Inspection detail page: show the left image (use the `rectified_left.png` artifact) with finding boxes drawn in image pixel coordinates (scale by the image's natural size), colour by type, label with Finding id. Clicking a box or a finding row selects it and focuses the 3D camera (verify the existing `regionCenter3D` focus code in `ThreeSurfaceViewer.tsx` still works; fix only if broken).
3. Toggle to overlay `deviation_heatmap.png` on the image with an opacity slider. Show the `deviation_summary` as a small table (units mm).
4. Findings list: type, confidence (show "n/a" when null), deviation mm, description, crop thumbnail.
5. If `classifierStatus` is "unavailable" or "failed", show a visible notice with `classifierError`.
6. Progress page: show each stage's `worker` and `durationSec` when present.
7. In the 3D viewer add a small marker at each finding's `regionCenter3D`. Do NOT attempt per-point deviation colouring (out of scope).
8. Keep the Real/Demo switch. Demo-mode mock data must stay clearly labelled as demo.
Checks: `npx oxlint` and `npm run build` pass. Report screenshots or a written description of what each page shows with the sample pair.

---------------------------------------------------------------------
## M4. Distributed execution with Celery + Redis

### 4.1 Design (follow exactly)
- `SURFACE_EXECUTION=local` (default): current thread-pool behaviour, unchanged.
- `SURFACE_EXECUTION=celery`: `POST /api/inspections` still validates and stores inputs on the master, then enqueues a Celery task instead of using the thread pool.
- Workers do NOT share the master's disk. The worker task downloads inputs from the master, runs the same `run_pipeline`, uploads every output file back, and reports progress and the final result to the master over HTTP.
- Refactor `JobManager` so that local mode and remote updates call the SAME functions to update job state (`apply_step_update`, `apply_result`, `apply_failure`). Keep the existing public methods working.

### 4.2 Files
- `backend/app/celery_app.py`: Celery app from `REDIS_URL` (broker and result backend). Config: `task_acks_late=True`, `task_reject_on_worker_lost=True`, `worker_prefetch_multiplier=1`, `broker_transport_options={"visibility_timeout": 900}`, `task_time_limit=600`, `task_soft_time_limit=540`, JSON serializer.
- `backend/app/tasks.py`: `run_inspection_task(inspection_id)` (retries up to 2 on network errors, not on pipeline errors).
- `backend/app/routes/internal.py` (router prefix `/api/internal`): all routes require header `X-Worker-Token` equal to `SURFACE_WORKER_TOKEN` (use `secrets.compare_digest`). If the token env var is empty, these routes return 403 always. Routes: `GET /jobs/{id}/manifest` (file names + calibration filename), `POST /jobs/{id}/artifacts/{name}` (multipart upload, safe filename, stays inside the job dir), `POST /jobs/{id}/step` (`index, step_id, status, worker, duration_sec, message`), `POST /jobs/{id}/complete` (result summary: point_count, mesh_status, mesh_error, triangle_count, deviation_summary, findings, classifier_status, classifier_error, duration_sec, worker), `POST /jobs/{id}/fail` (`message, failed_stage_index, worker`). Ignore updates for jobs already completed or failed.
- `backend/app/routes/cluster.py`: `GET /api/cluster/workers` returns `{mode, workers:[{name, online, active_jobs:[...], tasks_done}]}` using `celery_app.control.inspect(timeout=1.5)`; in local mode or if Redis is unreachable return `200` with `mode:"local"` and an empty list. Also expose `GET /api/cluster/summary` (jobs queued/processing/completed/failed counts and which worker each active job runs on, from job state).
- CORS: add origins from `SURFACE_CORS_ORIGINS` to the existing list.
- Worker task flow: read manifest -> download inputs via the existing artifact URLs -> parse calibration with the existing `parse_calibration` using the decoded image size -> `run_pipeline` in a temp dir with an `on_step_change` that posts `/step` (with worker name and per-stage duration) -> upload all outputs -> post `/complete`. On exception post `/fail`. Always delete the temp dir.
- Each machine must have the full repo (the mesh stage imports `PointCloudToMesh`). Start workers with `PYTHONPATH=backend:.` from the repo root so `import backend` and `import app...` both work.

### 4.3 Fan-out (M4b, only after M4a works and its gate is passed)
Split stage 4 tile classification across workers: stage 0-3 run in one task which uploads `rectified_left.png` and `deviations.json`, then a Celery `chord(group(classify_batch.s(job, batch_indices) ...), finalize.s(job))` using `self.replace`. Each `classify_batch` downloads the image, classifies its tile batch, returns probabilities; `finalize` merges and creates findings with the same code as M2 (shared functions, no duplicated logic). Batches of about 8 tiles. If this cannot be made reliable, STOP and report; job-level distribution alone is acceptable for the demo.

### 4.4 Tests
Use `task_always_eager=True` or mocks. Cover: token required (403 without), upload path traversal rejected, step/complete/fail update job state, duplicate completion ignored, local mode untouched, cluster endpoint returns local mode safely when Redis is down.
GATE M4a: with Redis on this machine, run one master + one worker process + the API, submit the sample pair, and show the job completes with a worker name in every step. Then kill the worker mid-job and report what happens (expected: task is redelivered to another worker after the visibility timeout, or the job fails with a clear message). Report honestly.

---------------------------------------------------------------------
## M5. Benchmark, run guide and demo runbook

1. `scripts/benchmark_cluster.py --api http://MASTER_IP:8000 --jobs 8`: submits N copies of the sample pair to the API, polls until all finish or fail, and writes `benchmarks/results.csv` (columns: workers_online, jobs, wall_seconds, avg_job_seconds, jobs_per_minute, failures) and prints a table. It reads the number of online workers from `/api/cluster/workers`. Do not fabricate: measured values only.
2. `docs/RUNBOOK.md`: the exact commands in section 7, a pre-demo checklist, troubleshooting (Redis connection refused, firewall, CORS, torch import, Poisson memory), and a fallback plan (run in local mode on one laptop; keep one finished inspection in History).
3. Update the root `README.md` status table: deviations and damage implemented, execution modes, how to run distributed.
GATE: final summary of all changes, test counts, build results, known limitations.

---------------------------------------------------------------------
## 7. How it runs on the 4 laptops (macOS commands, write these into RUNBOOK.md)

Replace `MASTER_IP` with the master laptop's LAN IP (System Settings > Network). All laptops on the same Wi-Fi or one phone hotspot. Pick one password `REDIS_PASS` and one `WORKER_TOKEN`, identical on all machines.

**Every laptop (once):** clone the repo, check out the branch, `cd backend && python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..`. Copy the model file to `backend/models_store/best_model_sdnet2018.pth`.

**Master laptop:**
```
brew install redis
redis-server --bind 0.0.0.0 --protected-mode yes --requirepass "$REDIS_PASS" --save ""
# new terminals, from repo root:
export SURFACE_EXECUTION=celery
export REDIS_URL="redis://:$REDIS_PASS@MASTER_IP:6379/0"
export SURFACE_MASTER_URL="http://MASTER_IP:8000"
export SURFACE_WORKER_TOKEN="$WORKER_TOKEN"
export SURFACE_CORS_ORIGINS="http://DEMO_IP:5173"
./backend/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
# monitoring dashboard (Flower):
PYTHONPATH=backend:. ./backend/venv/bin/python -m celery -A app.celery_app flower --port=5555
```
**Each worker laptop:**
```
export REDIS_URL="redis://:$REDIS_PASS@MASTER_IP:6379/0"
export SURFACE_MASTER_URL="http://MASTER_IP:8000"
export SURFACE_WORKER_TOKEN="$WORKER_TOKEN"
export SURFACE_WORKER_NAME="worker-1"      # worker-2 on the other laptop
PYTHONPATH=backend:. ./backend/venv/bin/python -m celery -A app.celery_app worker --pool=solo --loglevel=info -n $SURFACE_WORKER_NAME@%h
```
(`--pool=solo` avoids fork crashes with OpenCV/Open3D on macOS and works on Windows.)

**Demo laptop:** `npm install`; `.env` with `VITE_API_BASE_URL=http://MASTER_IP:8000`; `npm run dev`; open `http://localhost:5173`.

**Pre-demo checks (all must pass):** `redis-cli -h MASTER_IP -a $REDIS_PASS ping` returns PONG from each worker; `curl http://MASTER_IP:8000/api/health` works from the demo laptop; Flower at `http://MASTER_IP:5555` lists both workers; the Nodes page shows both online; one sample inspection completes with different worker names visible.

---------------------------------------------------------------------
## 8. Do-not list
- No cloud services, no paid packages, no Docker requirement.
- No changing thresholds just to make findings appear.
- No per-point colouring in the 3D viewer, no RAFT-Stereo integration, no new pages other than Nodes.
- No committing virtual environments or job storage. Model file may be committed only if under 50 MB (it is about 16 MB).

---------------------------------------------------------------------
## 9. Report format after every milestone (use exactly this)
1. Files created/changed (one line each, why).
2. Commands you ran and their real output summary (tests passed/failed counts, build result).
3. How the user can verify by hand in 2 minutes.
4. Known limitations or deviations from this spec (be explicit).
5. Questions, if any. Then wait for `continue M<next>`.