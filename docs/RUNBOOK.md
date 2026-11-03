# SURFACE — Multi-Laptop Demo Runbook

**Project:** Distributed 3D Structural Surface Processing System (SURFACE)  
**Audience:** Demonstration operators, lab evaluators, and development team.  
**Topology:** 4 Physical Laptops (1 Master Node, 2 Worker Nodes, 1 Demo/UI Client).  

---

## 1. Network Topology & Laptop Assignments

All four machines must be connected to the **same local Wi-Fi network or mobile hotspot**.

```
┌────────────────────────────────────────────────────────┐
│                   Laptop 4: Demo UI                    │
│   Web Browser (React + Three.js): http://localhost:5173│
│   VITE_API_BASE_URL = http://<MASTER_IP>:8000          │
└──────────────┬──────────────────────────▲──────────────┘
               │ (HTTP REST / Uploads)    │
               ▼                          │
┌───────────────────────────────┐         │ (Status & Artifacts)
│       Laptop 1: Master        │         │
│       LAN IP: MASTER_IP       │         │
│  ┌─────────────────────────┐  │         │
│  │ FastAPI Master (:8000)  │──┴─────────┘
│  │ Flower Monitor (:5555)  │
│  │ Redis Broker (:6379)    │◀─────────────────────┐
│  │ Local Job Storage       │                      │
│  └─────────────────────────┘                      │
└───────────────────────────────────────────────┐   │
                                                │   │
┌───────────────────────────────────────────┐   │   │ (Celery & HTTP Artifacts)
│             Laptop 2: Worker 1            │   │   │
│  Celery Worker Process (--pool=solo)      │───┼───┤
│  Worker Identity: worker-1                │   │   │
└───────────────────────────────────────────┘   │   │
                                                │   │
┌───────────────────────────────────────────┐   │   │
│             Laptop 3: Worker 2            │   │   │
│  Celery Worker Process (--pool=solo)      │───┴───┘
│  Worker Identity: worker-2                │
└───────────────────────────────────────────┘
```

| Laptop Role | Machine Hardware | Services Running | Primary Port(s) |
| :--- | :--- | :--- | :--- |
| **Laptop 1: Master** | macOS | Redis Server, FastAPI Master, Storage, Flower | `6379`, `8000`, `5555` |
| **Laptop 2: Worker 1** | macOS / Ubuntu / Windows | Celery Worker (`worker-1`), PyTorch, Open3D | Outbound to `6379`, `8000` |
| **Laptop 3: Worker 2** | macOS / Ubuntu / Windows | Celery Worker (`worker-2`), PyTorch, Open3D | Outbound to `6379`, `8000` |
| **Laptop 4: Demo UI** | macOS / Windows | Vite React Dev Server, Chrome/Safari Browser | `5173` |

---

## 2. Common Shared Credentials & IPs

Before starting, determine:
1. `MASTER_IP`: Find the Master Laptop's IP address (macOS: `ipconfig getifaddr en0` or **System Settings > Wi-Fi / Network > Details**).
2. `DEMO_IP`: Find the Demo Laptop's IP address for CORS configuration.
3. `REDIS_PASS`: Choose a strong password shared among the cluster (e.g., `SurfaceCluster2026Secret`).
4. `WORKER_TOKEN`: Choose a shared secret string for authenticated internal routes (e.g., `surf_tok_9b4a1f8e2c3d5e7a9b0c2d4e6f8a1b3c5e7`).

> [!IMPORTANT]
> Keep `REDIS_PASS` and `WORKER_TOKEN` identical on all machines. Never commit these secrets or `.env.*.local` files to Git.

---

## 3. Initial Setup on Every Laptop (Once)

Execute on **all four laptops**:

```bash
# 1. Clone repository and switch to feature branch
git clone <REPO_URL> design-project
cd design-project
git checkout feature/m2-deviation-damage-distributed

# 2. Set up Python virtual environment
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# 3. Verify ML model weights exist
# Expected path: backend/model_store/best_model_sdnet2018.pth (15.6 MB)
ls -la backend/model_store/best_model_sdnet2018.pth
```

---

## 4. Master Laptop Setup (Laptop 1)

### 4.1 Start Redis Broker
Install Redis if not already installed (`brew install redis`):
```bash
# Start Redis bound to all interfaces with password protection
redis-server \
  --bind 0.0.0.0 \
  --protected-mode yes \
  --requirepass "$REDIS_PASS" \
  --save "" \
  --port 6379
```

### 4.2 Start FastAPI Master Server
In a **new terminal tab** from the repository root:
```bash
export SURFACE_EXECUTION=celery
export REDIS_URL="redis://:$REDIS_PASS@MASTER_IP:6379/0"
export SURFACE_MASTER_URL="http://MASTER_IP:8000"
export SURFACE_WORKER_TOKEN="$WORKER_TOKEN"
export SURFACE_CORS_ORIGINS="http://localhost:5173,http://DEMO_IP:5173"

./backend/venv/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --app-dir backend
```

### 4.3 Start Flower Monitoring Dashboard (Optional / Recommended)
In a **new terminal tab** from the repository root:
```bash
export REDIS_URL="redis://:$REDIS_PASS@MASTER_IP:6379/0"

PYTHONPATH=backend:. ./backend/venv/bin/python -m celery \
  -A app.celery_app flower \
  --port=5555
```
Access Flower in browser at `http://localhost:5555` or `http://MASTER_IP:5555`.

---

## 5. Worker Laptops Setup (Laptops 2 & 3)

### 5.1 Worker 1 (Laptop 2)
In terminal from the repository root:
```bash
export REDIS_URL="redis://:$REDIS_PASS@MASTER_IP:6379/0"
export SURFACE_MASTER_URL="http://MASTER_IP:8000"
export SURFACE_WORKER_TOKEN="$WORKER_TOKEN"
export SURFACE_WORKER_NAME="worker-1"

PYTHONPATH=backend:. ./backend/venv/bin/python -m celery \
  -A app.celery_app worker \
  --pool=solo \
  --loglevel=info \
  -n $SURFACE_WORKER_NAME@%h
```

### 5.2 Worker 2 (Laptop 3)
In terminal from the repository root:
```bash
export REDIS_URL="redis://:$REDIS_PASS@MASTER_IP:6379/0"
export SURFACE_MASTER_URL="http://MASTER_IP:8000"
export SURFACE_WORKER_TOKEN="$WORKER_TOKEN"
export SURFACE_WORKER_NAME="worker-2"

PYTHONPATH=backend:. ./backend/venv/bin/python -m celery \
  -A app.celery_app worker \
  --pool=solo \
  --loglevel=info \
  -n $SURFACE_WORKER_NAME@%h
```

> [!NOTE]
> `--pool=solo` ensures single-process serial task execution per worker, preventing OpenCV/Open3D Cocoa GUI threading conflicts on macOS and ensuring stable PyTorch inference.

---

## 6. Demo Laptop Setup (Laptop 4)

In terminal from the repository root:
```bash
# 1. Install frontend dependencies
npm install

# 2. Configure master API connection
# Create or edit .env in repository root:
echo "VITE_API_BASE_URL=http://MASTER_IP:8000" > .env

# 3. Launch Vite development server
npm run dev -- --host 0.0.0.0
```
Open `http://localhost:5173` in Google Chrome or Safari.

---

## 7. Pre-Demo Verification Checklist (Gate: All Must Pass)

Run these checks in order before presenting:

- [ ] **1. Redis Broker Reachability:**
  From Laptop 2 and Laptop 3:
  ```bash
  redis-cli -h MASTER_IP -p 6379 -a "$REDIS_PASS" ping
  # Expected output: PONG
  ```
- [ ] **2. Master API Health Check:**
  From Laptop 4 (Demo Laptop):
  ```bash
  curl -s http://MASTER_IP:8000/api/health
  # Expected output: {"status":"healthy"}
  ```
- [ ] **3. Cluster Worker Nodes Online:**
  From Laptop 4:
  ```bash
  curl -s http://MASTER_IP:8000/api/cluster/workers
  # Expected output: mode="celery", both "worker-1" and "worker-2" report online: true
  ```
- [ ] **4. Nodes Web Page:**
  Open `http://localhost:5173/nodes` in the browser. Both workers must appear green with status **Online**.
- [ ] **5. End-to-End Test Inspection:**
  - In the web UI, click **New Inspection**.
  - Upload `public/samples/stereo_left.png`, `public/samples/stereo_right.png`, and `public/samples/camera_calibration_sample.json`.
  - Submit the inspection.
  - Verify progress through Stages 0 to 5.
  - Verify that stage steps show real worker names (`worker-1` or `worker-2`).
  - Verify 3D point cloud and mesh load into the Three.js viewport.
  - Verify crack finding detection box and heatmap toggle.

---

## 8. Troubleshooting Guide

### 8.1 Redis Connection Refused or Timed Out
- **Symptom:** Worker logs display `redis.exceptions.ConnectionError: Error 61 connecting to MASTER_IP:6379. Connection refused.`
- **Check 1:** Ensure Redis on Laptop 1 was started with `--bind 0.0.0.0`. If bound to `127.0.0.1`, remote laptops cannot connect.
- **Check 2:** On Laptop 1, check macOS Firewall (**System Settings > Network > Firewall**). Either disable the firewall for the demo or allow incoming connections to `redis-server`.
- **Check 3:** Verify both laptops are on the exact same Wi-Fi subnet (e.g. `192.168.1.x`). Check `ping MASTER_IP`.

### 8.2 CORS Errors in Browser Console
- **Symptom:** Inspection submission fails with `Cross-Origin Request Blocked` or `CORS preflight channel did not succeed`.
- **Resolution:** Verify `SURFACE_CORS_ORIGINS` on Laptop 1 includes the demo laptop's origin:
  ```bash
  export SURFACE_CORS_ORIGINS="http://localhost:5173,http://DEMO_IP:5173"
  ```
  Restart FastAPI master server on Laptop 1.

### 8.3 PyTorch Import or Model Loading Error
- **Symptom:** Worker fails during Stage 4 with `FileNotFoundError: Checkpoint not found at backend/model_store/best_model_sdnet2018.pth`.
- **Resolution:** Verify file location:
  `backend/model_store/best_model_sdnet2018.pth` (note: `model_store`, singular).
  Ensure file size is approximately 15.6 MB.

### 8.4 Open3D Poisson Reconstruction High Memory Pressure
- **Symptom:** Worker process crashes with `SIGKILL` or `Killed: 9` during Stage 2 Poisson surface reconstruction.
- **Resolution:** Lower the Poisson octree depth using the environment variable:
  ```bash
  export SURFACE_MESH_DEPTH=7
  ```
  Depth 7 uses ~4x less memory than depth 8 and completes in <10 seconds.

### 8.5 Worker Disconnects or Supervisor Recovery Triggered
- **Symptom:** A worker laptop goes to sleep or drops Wi-Fi; master logs `Recovery supervisor started grace period`.
- **Resolution:**
  - Set energy saver on all laptops to prevent sleep while plugged into AC power.
  - If a worker drops offline, the recovery supervisor automatically recovers the job and reassigns it to the remaining online worker under a fenced lease increment.

---

## 9. Fallback Plan (Single-Laptop Demonstration)

If Wi-Fi drops, network isolation is enforced, or multiple laptops fail:

1. **Activate Single-Laptop Local Execution:**
   On Laptop 1:
   ```bash
   unset REDIS_URL
   export SURFACE_EXECUTION=local
   ./backend/venv/bin/python -m uvicorn app.main:app --port 8000 --app-dir backend
   ```
   In local mode, the pipeline executes within an asynchronous thread pool without requiring Redis, Celery, or external network connectivity.

2. **Pre-Cooked Inspection in History:**
   Keep one previously completed inspection stored in `backend/storage/jobs/`.
   Navigate to `http://localhost:5173/history` and click the completed inspection. The full 3D point cloud, reconstructed mesh, deviation heatmap, and crack finding crops will load directly from local storage.

---

## 10. Security & Hygiene Checklist

- [ ] Never commit `.env` or `.env.*.local` files containing secrets.
- [ ] Never commit Redis passwords or worker tokens.
- [ ] Do not expose port 6379 (Redis) or port 5555 (Flower) to public untrusted networks.
- [ ] Bind Redis with `--protected-mode yes` and `--requirepass`.

