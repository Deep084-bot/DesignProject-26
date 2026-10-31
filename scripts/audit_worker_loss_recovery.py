"""
Controlled Real Recovery Test and Stale Assignment Audit for SURFACE M4a.

Strict Scope:
- Do not alter visibility timeout (remains 900s).
- Run Redis, Master FastAPI, Worker A (worker-recovery-a), and Worker B (worker-recovery-b).
- Submit sample pair.
- Confirm active worker assignment before terminating active worker.
- Terminate active worker forcefully (SIGKILL) while task is executing.
- Monitor surviving worker, master inspection progress, and Redis unacked queue.
- Audit stale assignment state on master.
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
    print("=" * 75)
    print("SURFACE M4a: WORKER-LOSS RECOVERY AUDIT & CONTROLLED TEST")
    print("=" * 75)

    # 1. Start Redis Server
    print("\n[Step 1] Starting Redis Server on port 6379...")
    redis_proc = subprocess.Popen(
        [str(REDIS_BIN), "--port", "6379", "--save", "", "--appendonly", "no"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    if redis_proc.poll() is not None:
        print("ERROR: Failed to start Redis server.")
        sys.exit(1)
    print("Redis server started (PID: %d)" % redis_proc.pid)

    r_client = redis.Redis(host="localhost", port=6379, db=0)
    r_client.flushall()

    # Environment variables
    env_base = os.environ.copy()
    env_base["SURFACE_EXECUTION"] = "celery"
    env_base["REDIS_URL"] = "redis://localhost:6379/0"
    env_base["SURFACE_MASTER_URL"] = "http://localhost:8000"
    env_base["SURFACE_WORKER_TOKEN"] = "audit-recovery-token-xyz"
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

    client = httpx.Client(base_url="http://localhost:8000", timeout=10.0)
    api_online = False
    for _ in range(25):
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

    # 4. Wait for both workers to register
    print("Waiting for both workers to be reported online by /api/cluster/workers...")
    workers_online = []
    for attempt in range(30):
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

    print(f"Online workers detected: {workers_online}")
    if len(workers_online) < 2:
        print("ERROR: Both workers failed to register within timeout.")
        worker_a_proc.kill()
        worker_b_proc.kill()
        api_proc.kill()
        redis_proc.kill()
        sys.exit(1)

    try:
        # 5. Submit sample inspection
        print("\n[Step 4] Submitting real sample stereo pair to POST /api/inspections...")
        sample_left = REPO_ROOT / "public" / "samples" / "stereo_left.png"
        sample_right = REPO_ROOT / "public" / "samples" / "stereo_right.png"
        sample_calib = REPO_ROOT / "public" / "samples" / "camera_calibration_sample.json"

        files = {
            "left_image": ("stereo_left.png", open(sample_left, "rb"), "image/png"),
            "right_image": ("stereo_right.png", open(sample_right, "rb"), "image/png"),
            "calibration_file": ("camera_calibration_sample.json", open(sample_calib, "rb"), "application/json"),
        }
        data = {
            "name": "M4a Worker-Loss Recovery Audit",
            "surface_type": "Concrete Retaining Wall",
        }

        submit_resp = client.post("/api/inspections", data=data, files=files)
        assert submit_resp.status_code == 201, f"Submission failed: {submit_resp.text}"
        inspection_id = submit_resp.json()["id"]
        print(f"Inspection submitted: {inspection_id} (initial status: {submit_resp.json()['status']})")

        # 6. Wait for task to be picked up and identify the active worker
        print("\n[Step 5] Monitoring task assignment and initial progress...")
        assigned_worker_name = None
        assigned_worker_proc = None
        surviving_worker_name = None
        surviving_worker_proc = None

        for _ in range(40):
            insp_data = client.get(f"/api/inspections/{inspection_id}").json()
            status = insp_data.get("status")
            step_idx = insp_data.get("current_step_index", 0)
            steps = insp_data.get("steps", [])
            last_worker = None
            for s in steps:
                if s.get("worker"):
                    last_worker = s["worker"]

            if status == "processing" and last_worker:
                assigned_worker_name = last_worker
                print(f"Task actively picked up by: '{assigned_worker_name}' at Step {step_idx}: {steps[step_idx].get('label')}")
                if "worker-recovery-a" in assigned_worker_name:
                    assigned_worker_proc = worker_a_proc
                    surviving_worker_name = "worker-recovery-b"
                    surviving_worker_proc = worker_b_proc
                else:
                    assigned_worker_proc = worker_b_proc
                    surviving_worker_name = "worker-recovery-a"
                    surviving_worker_proc = worker_a_proc
                break
            time.sleep(0.3)

        if not assigned_worker_name or not assigned_worker_proc:
            print("ERROR: Task was not picked up by any worker.")
            sys.exit(1)

        print(f"Active worker to be terminated: {assigned_worker_name} (PID: {assigned_worker_proc.pid})")
        print(f"Surviving worker remaining online: {surviving_worker_name} (PID: {surviving_worker_proc.pid})")

        # Give it a moment to enter Step 2 (mesh reconstruction)
        time.sleep(2.0)
        current_state_before_kill = client.get(f"/api/inspections/{inspection_id}").json()
        print(f"Current step before termination: Step {current_state_before_kill['current_step_index']}: {current_state_before_kill['steps'][current_state_before_kill['current_step_index']]['label']}")

        # 7. Terminate active worker forcefully (SIGKILL)
        print("\n[Step 6] Forcefully terminating active worker process via SIGKILL...")
        kill_timestamp = time.time()
        assigned_worker_proc.send_signal(signal.SIGKILL)
        assigned_worker_proc.wait()
        print(f"Active worker killed at timestamp: {kill_timestamp:.3f}")

        # 8. PART 3 AUDIT: Check stale assignment state immediately after kill
        print("\n[Step 7 / Part 3 Audit] Inspecting cluster status immediately after active worker termination...")
        time.sleep(1.0)
        cluster_workers_after_kill = client.get("/api/cluster/workers").json()
        print(f"  /api/cluster/workers response: {cluster_workers_after_kill}")

        cluster_summary_after_kill = client.get("/api/cluster/summary").json()
        print(f"  /api/cluster/summary response: {cluster_summary_after_kill}")

        active_assignments = cluster_summary_after_kill.get("active_assignments", {})
        print(f"  Active assignments map: {active_assignments}")

        dead_worker_in_summary = assigned_worker_name in active_assignments.values()
        print(f"  --> Is the DEAD worker '{assigned_worker_name}' still reported as assigned in cluster summary? {dead_worker_in_summary}")

        # Inspect Redis directly
        print("\n[Redis State Audit]")
        redis_keys = [k.decode("utf-8") for k in r_client.keys("*")]
        print(f"  Redis keys in DB 0: {redis_keys}")
        for k in redis_keys:
            k_type = r_client.type(k).decode("utf-8")
            if k_type == "zset":
                zitems = r_client.zrange(k, 0, -1, withscores=True)
                print(f"    Key '{k}' (zset, length {len(zitems)}):")
                for member, score in zitems:
                    remaining_sec = score - time.time()
                    print(f"      member={member[:30]}... score={score:.1f} (remaining: {remaining_sec:.1f}s until visibility timeout)")
            elif k_type == "hash":
                hlen = r_client.hlen(k)
                print(f"    Key '{k}' (hash, entries: {hlen})")
            elif k_type == "list":
                llen = r_client.llen(k)
                print(f"    Key '{k}' (list, length: {llen})")

        # 9. Observe whether surviving worker picks up the task
        print(f"\n[Step 8] Observing surviving worker ({surviving_worker_name}) for task redelivery...")
        print("Monitoring for up to 60 seconds...")

        task_received_by_survivor = False
        survivor_start_time = None
        time_to_redelivery = None

        poll_start = time.time()
        while time.time() - poll_start < 60:
            insp = client.get(f"/api/inspections/{inspection_id}").json()
            curr_status = insp.get("status")
            curr_steps = insp.get("steps", [])

            # Check if any step shows the surviving worker
            survivor_steps = [s for s in curr_steps if s.get("worker") and surviving_worker_name in s.get("worker")]
            if survivor_steps:
                task_received_by_survivor = True
                survivor_start_time = time.time()
                time_to_redelivery = survivor_start_time - kill_timestamp
                print(f"\nSUCCESS: Task received by surviving worker '{surviving_worker_name}'!")
                print(f"Redelivery elapsed time: {time_to_redelivery:.2f} seconds")
                break

            # Also check if completed or failed
            if curr_status == "completed":
                print(f"\nInspection completed: {insp.get('point_count')} points, status: {curr_status}")
                break
            elif curr_status == "failed":
                print(f"\nInspection failed: {insp.get('failure_message')}")
                break

            print(f"  [{time.time()-poll_start:4.1f}s elapsed] Status: {curr_status} | Step {insp.get('current_step_index')}")
            time.sleep(2.0)

        # 10. Summary and Conclusion of Test
        print("\n" + "=" * 75)
        print("RECOVERY AUDIT TEST RESULTS SUMMARY")
        print("=" * 75)
        print(f"Inspection ID:                {inspection_id}")
        print(f"Initial Active Worker:        {assigned_worker_name}")
        print(f"Killed At:                    t = {kill_timestamp:.3f}")
        print(f"Surviving Worker:             {surviving_worker_name}")
        print(f"Task Received by Survivor:    {task_received_by_survivor}")
        if task_received_by_survivor:
            print(f"Time to Redelivery:           {time_to_redelivery:.2f} s")
        else:
            print("Time to Redelivery:           Not redelivered within 60s observation window.")

        final_insp = client.get(f"/api/inspections/{inspection_id}").json()
        print(f"Final Inspection Status:      {final_insp.get('status')}")
        print(f"Final Current Step:           {final_insp.get('current_step_index')}")
        print(f"Step Workers:                 {[s.get('worker') for s in final_insp.get('steps', [])]}")

        final_summary = client.get("/api/cluster/summary").json()
        print(f"Final Cluster Summary:        {final_summary}")

    finally:
        print("\nCleaning up processes...")
        for p in [worker_a_proc, worker_b_proc, api_proc, redis_proc]:
            try:
                p.kill()
            except Exception:
                pass
        print("All processes terminated.")


if __name__ == "__main__":
    main()

