"""
Real Integration Gate Verification for SURFACE M4a: Worker-Loss Recovery.

Demonstrates actual worker recovery after forceful termination (SIGKILL) of active worker:
1. Starts Redis broker and FastAPI master with configured recovery intervals:
   - Heartbeat interval: 2.0s
   - Lease timeout: 8.0s
   - Supervisor interval: 1.0s
   - Offline grace period: 2.0s
2. Starts two independent Celery workers with --pool=solo:
   - Worker A: worker-recovery-a
   - Worker B: worker-recovery-b
3. Submits real stereo pair inspection.
4. Detects which worker claims generation 1 lease.
5. While pipeline is actively running, kills the active worker via SIGKILL.
6. Observes master supervisor detect worker loss and trigger generation 2 reassignment.
7. Observes surviving worker (Worker B) claim generation 2 and execute the full pipeline to completion.
8. Verifies output artifacts, findings, and inspection completion contracts.
9. Cleans up all processes.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
import httpx
import redis

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = REPO_ROOT / "backend" / "venv" / "bin" / "python"
REDIS_BIN = Path("/opt/homebrew/opt/redis/bin/redis-server")


def main():
    print("=" * 80)
    print("SURFACE M4a: REAL TWO-WORKER RECOVERY INTEGRATION GATE")
    print("=" * 80)

    # 1. Start or Connect to Redis Server
    print("\n[Step 1] Connecting to or starting Redis Server on port 6379...")
    redis_proc = None
    r_client = redis.Redis(host="localhost", port=6379, db=0)
    try:
        r_client.ping()
        print("Connected to existing Redis server on port 6379.")
    except Exception:
        print("Starting new Redis Server on port 6379...")
        redis_proc = subprocess.Popen(
            [str(REDIS_BIN), "--port", "6379", "--save", "", "--appendonly", "no"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.0)
        if redis_proc.poll() is not None:
            stdout, stderr = redis_proc.communicate()
            print(f"ERROR: Failed to start Redis server. Out: {stdout}, Err: {stderr}")
            sys.exit(1)
        print(f"Redis server started (PID: {redis_proc.pid})")

    r_client.flushall()

    # Environment variables
    env_base = os.environ.copy()
    env_base["SURFACE_EXECUTION"] = "celery"
    env_base["REDIS_URL"] = "redis://localhost:6379/0"
    env_base["SURFACE_MASTER_URL"] = "http://localhost:8000"
    env_base["SURFACE_WORKER_TOKEN"] = "gate-recovery-token-xyz"
    env_base["SURFACE_HEARTBEAT_INTERVAL"] = "2.0"
    env_base["SURFACE_LEASE_TIMEOUT"] = "8.0"
    env_base["SURFACE_SUPERVISOR_INTERVAL"] = "1.0"
    env_base["SURFACE_OFFLINE_GRACE_PERIOD"] = "2.0"
    env_base["SURFACE_MESH_DEPTH"] = "6"
    env_base["PYTHONPATH"] = f"{REPO_ROOT / 'backend'}:{REPO_ROOT}"

    # 2. Start Master FastAPI API
    print("\n[Step 2] Starting Master FastAPI API on http://localhost:8000...")
    api_proc = subprocess.Popen(
        [
            str(PYTHON_BIN),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--app-dir",
            str(REPO_ROOT / "backend"),
        ],
        env=env_base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    client = httpx.Client(base_url="http://localhost:8000", timeout=15.0)
    api_online = False
    for _ in range(30):
        try:
            res = client.get("/api/health")
            if res.status_code == 200:
                api_online = True
                break
        except Exception:
            time.sleep(0.4)

    if not api_online:
        print("ERROR: FastAPI master API failed to start.")
        api_proc.kill()
        redis_proc.kill()
        sys.exit(1)
    print("Master FastAPI API is online.")

    # 3. Start Worker A and Worker B
    print("\n[Step 3] Starting Worker A (worker-recovery-a) and Worker B (worker-recovery-b)...")
    env_a = env_base.copy()
    env_a["SURFACE_WORKER_NAME"] = "worker-recovery-a"
    worker_a_proc = subprocess.Popen(
        [
            str(PYTHON_BIN),
            "-m",
            "celery",
            "-A",
            "app.celery_app",
            "worker",
            "--pool=solo",
            "--loglevel=info",
            "-n",
            "worker-recovery-a@localhost",
        ],
        env=env_a,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
    )

    env_b = env_base.copy()
    env_b["SURFACE_WORKER_NAME"] = "worker-recovery-b"
    worker_b_proc = subprocess.Popen(
        [
            str(PYTHON_BIN),
            "-m",
            "celery",
            "-A",
            "app.celery_app",
            "worker",
            "--pool=solo",
            "--loglevel=info",
            "-n",
            "worker-recovery-b@localhost",
        ],
        env=env_b,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
    )

    # Wait for both workers to register
    print("Waiting for both workers to register in cluster...")
    workers_online = []
    for _ in range(30):
        try:
            res = client.get("/api/cluster/workers")
            if res.status_code == 200:
                w_list = res.json().get("workers", [])
                online_names = [w["name"] for w in w_list if w.get("online")]
                if len(online_names) >= 2:
                    workers_online = online_names
                    break
        except Exception:
            pass
        time.sleep(0.5)

    print(f"Online workers: {workers_online}")
    if len(workers_online) < 2:
        print("ERROR: Both workers failed to register within timeout.")
        worker_a_proc.kill()
        worker_b_proc.kill()
        api_proc.kill()
        redis_proc.kill()
        sys.exit(1)

    try:
        # 4. Submit sample inspection
        print("\n[Step 4] Submitting stereo pair inspection to POST /api/inspections...")
        sample_left = REPO_ROOT / "public" / "samples" / "stereo_left.png"
        sample_right = REPO_ROOT / "public" / "samples" / "stereo_right.png"
        sample_calib = REPO_ROOT / "public" / "samples" / "camera_calibration_sample.json"

        files = {
            "left_image": ("stereo_left.png", open(sample_left, "rb"), "image/png"),
            "right_image": ("stereo_right.png", open(sample_right, "rb"), "image/png"),
            "calibration_file": ("camera_calibration_sample.json", open(sample_calib, "rb"), "application/json"),
        }
        data = {
            "name": "M4a Worker-Loss Recovery Gate",
            "surface_type": "Concrete Surface",
        }

        submit_resp = client.post("/api/inspections", data=data, files=files)
        assert submit_resp.status_code == 201, f"Submission failed: {submit_resp.text}"
        inspection_id = submit_resp.json()["id"]
        print(f"Inspection submitted: {inspection_id} (lease: {submit_resp.json()['lease_token']})")

        # 5. Monitor task assignment and identify active worker
        print("\n[Step 5] Waiting for task assignment to active worker...")
        assigned_worker_name = None
        assigned_worker_proc = None
        surviving_worker_name = None
        surviving_worker_proc = None

        for _ in range(40):
            insp = client.get(f"/api/inspections/{inspection_id}").json()
            worker = insp.get("assigned_worker")
            step_idx = insp.get("current_step_index", 0)
            status = insp.get("status")

            if worker:
                assigned_worker_name = worker
                if "worker-recovery-a" in worker:
                    assigned_worker_proc = worker_a_proc
                    surviving_worker_name = "worker-recovery-b"
                    surviving_worker_proc = worker_b_proc
                else:
                    assigned_worker_proc = worker_b_proc
                    surviving_worker_name = "worker-recovery-a"
                    surviving_worker_proc = worker_a_proc
                print(f"Task claimed by: {assigned_worker_name} (step: {step_idx}, status: {status})")
                break
            time.sleep(0.5)

        if not assigned_worker_name:
            raise RuntimeError("Task was not claimed by any worker within timeout.")

        # Wait until Stage 1 or 2 is running
        print("Waiting for active pipeline execution...")
        for _ in range(30):
            insp = client.get(f"/api/inspections/{inspection_id}").json()
            step_idx = insp.get("current_step_index", 0)
            if step_idx >= 1:
                print(f"Pipeline running at step index {step_idx} ({insp['steps'][step_idx]['id']})")
                break
            time.sleep(0.5)

        # 6. Forcefully kill the active worker with SIGKILL
        print(f"\n[Step 6] Forcefully terminating active worker '{assigned_worker_name}' via SIGKILL (PID: {assigned_worker_proc.pid})...")
        os.kill(assigned_worker_proc.pid, signal.SIGKILL)
        assigned_worker_proc.wait(timeout=5.0)
        print(f"Worker process {assigned_worker_name} confirmed dead (exit code: {assigned_worker_proc.returncode})")

        # Confirm surviving worker is alive
        assert surviving_worker_proc.poll() is None, f"Surviving worker {surviving_worker_name} died unexpectedly!"
        print(f"Surviving worker '{surviving_worker_name}' is confirmed alive (PID: {surviving_worker_proc.pid})")

        # 7. Observe detection, reassignment, and recovery execution
        print("\n[Step 7] Monitoring recovery supervisor detection and lease reassignment...")
        recovery_detected = False
        reassigned_lease = None

        t0 = time.time()
        while time.time() - t0 < 45.0:
            insp = client.get(f"/api/inspections/{inspection_id}").json()
            current_lease = insp.get("lease_token", 1)
            current_worker = insp.get("assigned_worker")
            current_status = insp.get("status")
            step_idx = insp.get("current_step_index", 0)

            summary = client.get("/api/cluster/summary").json()
            active_assignments = summary.get("active_assignments", {})

            print(
                f"[{round(time.time() - t0, 1)}s] status={current_status}, lease={current_lease}, "
                f"worker={current_worker}, step={step_idx}, active_assignments={active_assignments}"
            )

            if current_lease >= 2:
                recovery_detected = True
                reassigned_lease = current_lease

            if current_status == "completed":
                print(f"\nSUCCESS: Inspection {inspection_id} reached 'completed' state!")
                break

            if current_status == "failed":
                raise RuntimeError(f"Inspection failed during recovery: {insp.get('failure_message')}")

            time.sleep(1.5)

        assert recovery_detected, "Recovery supervisor failed to reassign lease within timeout!"
        print(f"Lease successfully incremented to generation {reassigned_lease}!")

        # 8. Verify final completed state
        final_insp = client.get(f"/api/inspections/{inspection_id}").json()
        assert final_insp["status"] == "completed", f"Expected completed status, got: {final_insp['status']}"
        assert final_insp["lease_token"] == 2, f"Expected final lease_token=2, got: {final_insp['lease_token']}"
        assert final_insp["point_count"] and final_insp["point_count"] > 0, "Missing or zero point_count"
        assert final_insp["point_cloud_url"] is not None, "Missing point_cloud_url"
        assert final_insp["mesh_url"] is not None, "Missing mesh_url"
        assert final_insp["triangle_count"] and final_insp["triangle_count"] > 0, "Missing triangle_count"
        assert final_insp["deviation_summary"] is not None, "Missing deviation_summary"

        # Verify surviving worker executed and completed the task
        completed_step_workers = [s.get("worker") for s in final_insp["steps"] if s.get("status") == "completed"]
        print(f"Completed step workers: {set(completed_step_workers)}")
        assert any(surviving_worker_name in str(w) for w in completed_step_workers), (
            f"Expected surviving worker '{surviving_worker_name}' to have executed the pipeline!"
        )

        # 9. Verify artifact retrieval endpoints
        print("\n[Step 8] Verifying artifact retrieval endpoints...")
        pc_resp = client.get(f"/api/inspections/{inspection_id}/artifacts/pointcloud.ply")
        assert pc_resp.status_code == 200, "pointcloud.ply retrieval failed"
        assert len(pc_resp.content) > 1000

        mesh_resp = client.get(f"/api/inspections/{inspection_id}/artifacts/mesh.ply")
        assert mesh_resp.status_code == 200, "mesh.ply retrieval failed"
        assert len(mesh_resp.content) > 1000

        heatmap_resp = client.get(f"/api/inspections/{inspection_id}/artifacts/deviation_heatmap.png")
        assert heatmap_resp.status_code == 200, "deviation_heatmap.png retrieval failed"
        assert len(heatmap_resp.content) > 500

        print("\n" + "=" * 80)
        print("M4a WORKER-LOSS RECOVERY INTEGRATION GATE: PASSED (100% VERIFIED)")
        print("=" * 80)

    finally:
        print("\n[Cleanup] Terminating background processes...")
        procs_to_kill = [worker_a_proc, worker_b_proc, api_proc]
        if redis_proc:
            procs_to_kill.append(redis_proc)
        for p in procs_to_kill:
            try:
                if p and p.poll() is None:
                    p.terminate()
                    p.wait(timeout=2.0)
            except Exception:
                try:
                    if p:
                        p.kill()
                except Exception:
                    pass
        print("All processes terminated.")


if __name__ == "__main__":
    main()
