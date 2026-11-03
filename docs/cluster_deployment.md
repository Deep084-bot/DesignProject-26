# SURFACE Distributed Cluster Deployment Guide (M4b Phase 4.1)

This guide documents how to configure, preflight, and start the distributed SURFACE inspection cluster across a Windows 11 host and two Ubuntu virtual machines.

---

## 1. Target Topology and Network Roles

```
┌────────────────────────────────────────────────────────┐
│             Host Machine: Windows 11                   │
│     Browser / React Frontend: http://localhost:5173    │
│     VITE_API_BASE_URL = http://<MASTER_VM_IP>:8000     │
└──────────────┬──────────────────────────▲──────────────┘
               │ (HTTP API Requests)      │
               ▼                          │
┌───────────────────────────────┐         │
│     Master VM 1 (Ubuntu)      │         │
│     e.g., 192.168.56.10       │         │
│  ┌─────────────────────────┐  │         │
│  │ FastAPI Master (:8000)  │──┼─────────┘
│  │ Recovery Supervisor     │  │
│  └───────────┬─────────────┘  │
│  ┌───────────▼─────────────┐  │ (HTTP Artifacts / Heartbeats)
│  │ Redis Broker (:6379)    │◀─┼──────────────────┐
│  └─────────────────────────┘  │                  │
└───────────────────────────────┘                  │
                                                   │
┌──────────────────────────────────────────────────┴─────┐
│                  Worker VM 2 (Ubuntu)                  │
│                  e.g., 192.168.56.11                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Celery Worker Process (--pool=solo)              │  │
│  │   SURFACE_MASTER_URL = http://192.168.56.10:8000 │  │
│  │   REDIS_URL          = redis://192.168.56.10:6379│  │
│  │   SURFACE_WORKER_TOKEN = <shared-secret>         │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

| Machine | Role | Services Running | Primary Script |
|---|---|---|---|
| **Windows 11 Host** | UI / Client | Vite React Frontend (`port 5173`) | `npm run dev` |
| **Ubuntu VM 1** | Cluster Master | Redis (`port 6379`), FastAPI (`port 8000`), Recovery Supervisor | `./scripts/start_master.sh` |
| **Ubuntu VM 2** | Cluster Worker | Celery Worker (`--pool=solo`, OpenCV, Open3D, PyTorch) | `./scripts/start_worker.sh` |

---

## 2. Finding VM IP Addresses and Verifying Ports

On each Ubuntu VM, find its IP address using:
```bash
ip -br addr show
# Or:
hostname -I
```
Identify the IP on your Host-Only or Bridged private network (e.g. `192.168.56.10` for VM 1, `192.168.56.11` for VM 2).

### Required Network Ports:
* **Port 8000 (TCP)**: FastAPI Master API on VM 1. Must be reachable by Windows Host and VM 2.
* **Port 6379 (TCP)**: Redis Broker on VM 1. Must be reachable by VM 1 and VM 2.
* **Port 5173 (TCP)**: Vite Frontend dev server on Windows Host (accessible in host browser).

---

## 3. Configuring Redis Safely on Master VM 1

By default, Ubuntu Redis binds only to `127.0.0.1`. To allow Worker VM 2 to connect securely:

1. Edit `/etc/redis/redis.conf` on Master VM 1:
   ```conf
   # Bind to loopback AND the VM private IP (replace 192.168.56.10 with VM 1 IP):
   bind 127.0.0.1 192.168.56.10

   # Require strong password authentication:
   requirepass YourStrongRedisPasswordHere

   # Keep protected-mode active:
   protected-mode yes
   ```
   > [!IMPORTANT]
   > Do **NOT** set `protected-mode no` with an unauthenticated Redis instance exposed to the network.

2. Restart Redis service:
   ```bash
   sudo systemctl restart redis-server
   ```

3. Configure Ubuntu firewall (`ufw`) on Master VM 1:
   ```bash
   # Allow Redis only from the private subnet:
   sudo ufw allow from 192.168.56.0/24 to any port 6379 proto tcp
   # Allow Master API from host and worker:
   sudo ufw allow 8000/tcp
   ```

---

## 4. Setting Up Environment Files

### On Master VM 1 (`backend/.env.master`):
```bash
cp backend/.env.master.example backend/.env.master
chmod 600 backend/.env.master
```
Edit `backend/.env.master` and configure:
```env
SURFACE_EXECUTION=celery
REDIS_URL=redis://:YourStrongRedisPasswordHere@127.0.0.1:6379/0
SURFACE_MASTER_URL=http://192.168.56.10:8000
SURFACE_WORKER_TOKEN=my-secure-cluster-shared-secret-2026
SURFACE_WORKER_NAME=master-coordinator
SURFACE_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://192.168.56.1:5173
SURFACE_MODEL_PATH=backend/model_store/best_model_sdnet2018.pth
```

### On Worker VM 2 (`backend/.env.worker`):
```bash
cp backend/.env.worker.example backend/.env.worker
chmod 600 backend/.env.worker
```
Edit `backend/.env.worker` and configure:
```env
SURFACE_EXECUTION=celery
REDIS_URL=redis://:YourStrongRedisPasswordHere@192.168.56.10:6379/0
SURFACE_MASTER_URL=http://192.168.56.10:8000
SURFACE_WORKER_TOKEN=my-secure-cluster-shared-secret-2026
SURFACE_WORKER_NAME=worker-ubuntu-node2
SURFACE_MODEL_PATH=backend/model_store/best_model_sdnet2018.pth
```

### Critical Rules:
* **MUST MATCH**: `SURFACE_WORKER_TOKEN`, `REDIS_URL` password/db, and `SURFACE_EXECUTION=celery` must be identical on both machines.
* **MACHINE SPECIFIC**:
  * `SURFACE_WORKER_NAME`: Must be unique per worker (e.g. `worker-node-1`, `worker-node-2`).
  * `SURFACE_MASTER_URL`: Points to Master VM 1's IP (`http://192.168.56.10:8000`).
  * `SURFACE_CORS_ORIGINS`: On Master VM 1, include the Windows Host IP address running the frontend.

---

## 5. Running the Preflight Diagnostic Verification

Before launching services, run `check_cluster_preflight.py` to confirm network, Redis, token, and filesystem prerequisites.

### On Master VM 1:
```bash
./backend/venv/bin/python scripts/check_cluster_preflight.py --env-file backend/.env.master --role master
```

### On Worker VM 2:
```bash
./backend/venv/bin/python scripts/check_cluster_preflight.py --env-file backend/.env.worker --role worker
```

The script prints clear `[PASS]`, `[WARN]`, and `[FAIL]` status indicators for each check:
* **Dependencies**: Checks FastAPI, Celery, Redis, PyTorch, OpenCV, Open3D.
* **Filesystem**: Checks `backend/app/`, `backend/storage/`, and model weights at `backend/model_store/best_model_sdnet2018.pth`.
* **Configuration & Security**: Checks token length, masked Redis URL, execution mode (secrets are never displayed).
* **Redis Network Connect**: Tests TCP connection and protocol `PING` with configured password.
* **Master Health & Auth Probe**: Performs read-only health checks and validates token acceptance.

---

## 6. Starting and Stopping Cluster Processes

### Step 1: Start Master API (Master VM 1)
```bash
./scripts/start_master.sh backend/.env.master
```
* Binds Uvicorn to `0.0.0.0:8000`.
* Acquires master storage lock.
* Reconciles crash outbox on startup.
* Starts background `RecoverySupervisor`.

### Step 2: Start Celery Worker (Worker VM 2)
```bash
./scripts/start_worker.sh backend/.env.worker
```
* Starts Celery worker in `--pool=solo` mode named `<WORKER_NAME>@%h`.
* Connects to Redis broker on Master VM 1.
* Ready to process coordinator tasks, fanout tile batches, and aggregation finalizers.

### Step 3: Verify Cluster Status on Web Frontend (Windows Host)
1. In the repository on your Windows host, set `VITE_API_BASE_URL=http://192.168.56.10:8000` in `.env`.
2. Start the Vite dev server: `npm run dev`.
3. Open `http://localhost:5173/nodes` in your browser.
4. Verify the top badge displays **"Celery + Redis"** in emerald, and Worker VM 2 appears as an active node with a pulsing green **Online** badge.

### Stopping Processes:
* To stop Master or Worker: Press `Ctrl+C` in the running terminal. Both scripts handle `SIGINT` gracefully and cleanly tear down child processes.

---

## 7. Common Configuration Errors and Troubleshooting

| Symptom / Error | Root Cause | Solution |
|---|---|---|
| `HTTP 403 Forbidden` on internal worker routes | `SURFACE_WORKER_TOKEN` is unset or differs between Master and Worker. | Verify both `backend/.env.master` and `backend/.env.worker` specify the exact same non-empty `SURFACE_WORKER_TOKEN`. |
| `Redis TCP connect: Connection refused` | Redis is not running or is bound only to `127.0.0.1`. | Edit `/etc/redis/redis.conf` on VM 1 to `bind 127.0.0.1 <VM1_IP>` and restart `redis-server`. |
| `Redis authentication failed` | Password in `REDIS_URL` does not match `requirepass` in `redis.conf`. | Update password in `REDIS_URL` in `.env.master` and `.env.worker`. |
| Browser shows CORS error when accessing API | Windows Host origin is not in Master's `SURFACE_CORS_ORIGINS`. | Add your Windows host IP (e.g. `http://192.168.56.1:5173`) to `SURFACE_CORS_ORIGINS` in `backend/.env.master` and restart master. |
| Nodes page shows "No Celery Workers Detected" | Celery worker is not running or cannot reach Redis. | Run `scripts/check_cluster_preflight.py` on Worker VM 2 to test Redis connectivity. |
| Model checkpoint warning | `best_model_sdnet2018.pth` is missing on Worker VM 2. | Copy `backend/model_store/best_model_sdnet2018.pth` (16 MB) to Worker VM 2. |
