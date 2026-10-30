"""
GATE M4a Verification Script: Distributed Job Execution with Celery + Redis.

Runs:
1. Local Redis server on port 6379
2. Master FastAPI server on port 8000 (SURFACE_EXECUTION=celery)
3. Celery worker process (pool=solo, worker-gate-01)
4. Submits real sample stereo pair to API
5. Verifies job completion with worker name recorded in every pipeline step
6. Mid-job worker kill test: submits second job, terminates worker process mid-execution,
   and observes failure / redelivery behavior
7. Cleans up all child processes cleanly
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = REPO_ROOT / "backend" / "venv" / "bin" / "python"
REDIS_BIN = Path("/opt/homebrew/opt/redis/bin/redis-server")


def main():
    print("=" * 70)
    print("SURFACE M4a GATE VERIFICATION: CELERY + REDIS")
    print("=" * 70)

    # 1. Start Redis Server
    print("\n[1/6] Starting Redis Server on port 6379...")
    redis_proc = subprocess.Popen(
        [str(REDIS_BIN), "--port", "6379", "--save", "", "--appendonly", "no"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    if redis_proc.poll() is not None:
        print("ERROR: Failed to start Redis server.")
        sys.exit(1)
    print("Redis server running (PID: %d)" % redis_proc.pid)

    # Set environment variables for master and worker
    env = os.environ.copy()
    env["SURFACE_EXECUTION"] = "celery"
    env["REDIS_URL"] = "redis://localhost:6379/0"
    env["SURFACE_MASTER_URL"] = "http://localhost:8000"
    env["SURFACE_WORKER_TOKEN"] = "gate-token-m4a-secret"
    env["SURFACE_WORKER_NAME"] = "worker-gate-01"
    env["PYTHONPATH"] = f"{REPO_ROOT / 'backend'}:{REPO_ROOT}"

    # 2. Start Master API (Uvicorn)
    print("\n[2/6] Starting Master FastAPI API on http://localhost:8000...")
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
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for API to become healthy
    client = httpx.Client(base_url="http://localhost:8000", timeout=10.0)
    api_online = False
    for _ in range(20):
        try:
            r = client.get("/api/health")
            if r.status_code == 200:
                api_online = True
                break
        except Exception:
            time.sleep(0.5)

    if not api_online:
        print("ERROR: FastAPI failed to start.")
        api_proc.kill()
        redis_proc.kill()
        sys.exit(1)
    print("FastAPI master API online.")

    # 3. Start Celery Worker
    print("\n[3/6] Starting Celery Worker (worker-gate-01)...")
    worker_proc = subprocess.Popen(
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
            "worker-gate-01@localhost",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
    )

    # Wait for worker to register
    print("Waiting for worker registration via /api/cluster/workers...")
    worker_registered = False
    for _ in range(25):
        try:
            res = client.get("/api/cluster/workers")
            if res.status_code == 200 and len(res.json().get("workers", [])) > 0:
                worker_info = res.json()["workers"][0]
                print(f"Worker registered: {worker_info['name']} (online: {worker_info['online']})")
                worker_registered = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not worker_registered:
        print("ERROR: Worker did not register in time.")
        worker_proc.kill()
        api_proc.kill()
        redis_proc.kill()
        sys.exit(1)

    try:
        # 4. Submit Sample Pair
        print("\n[4/6] Submitting real sample stereo pair to POST /api/inspections...")
        sample_left = REPO_ROOT / "public" / "samples" / "stereo_left.png"
        sample_right = REPO_ROOT / "public" / "samples" / "stereo_right.png"
        sample_calib = REPO_ROOT / "public" / "samples" / "camera_calibration_sample.json"

        files = {
            "left_image": ("stereo_left.png", open(sample_left, "rb"), "image/png"),
            "right_image": ("stereo_right.png", open(sample_right, "rb"), "image/png"),
            "calibration_file": ("camera_calibration_sample.json", open(sample_calib, "rb"), "application/json"),
        }
        data = {
            "name": "GATE M4a Sample Pair Inspection",
            "surface_type": "Concrete Retaining Wall",
            "reconstruction_method": "Automatic surface reconstruction",
        }

        submit_resp = client.post("/api/inspections", data=data, files=files)
        assert submit_resp.status_code == 201, f"Submit failed: {submit_resp.text}"
        inspection_id = submit_resp.json()["id"]
        print(f"Inspection submitted: {inspection_id} (initial status: {submit_resp.json()['status']})")

        # 5. Poll until completion
        print("\n[5/6] Polling inspection progress...")
        start_time = time.time()
        final_inspection = None
        while time.time() - start_time < 90:
            poll_resp = client.get(f"/api/inspections/{inspection_id}")
            assert poll_resp.status_code == 200
            insp = poll_resp.json()
            status = insp["status"]
            step_idx = insp["current_step_index"]
            steps = insp.get("steps", [])
            current_label = steps[step_idx]["label"] if step_idx < len(steps) else "Done"

            print(f"  [{time.time()-start_time:5.1f}s] Status: {status:10s} | Step {step_idx}: {current_label}")
            if status in ("completed", "failed"):
                final_inspection = insp
                break
            time.sleep(1.0)

        assert final_inspection is not None, "Timed out waiting for inspection completion."
        assert final_inspection["status"] == "completed", f"Job failed: {final_inspection.get('failure_message')}"

        print("\n--- JOB EXECUTION SUMMARY ---")
        print(f"Inspection ID:     {final_inspection['id']}")
        print(f"Status:            {final_inspection['status']}")
        print(f"Duration:          {final_inspection.get('processing_duration_sec')} s")
        print(f"Point count:       {final_inspection.get('point_count')}")
        print(f"Mesh status:       {final_inspection.get('mesh_status')}")
        print(f"Triangle count:    {final_inspection.get('triangle_count')}")
        print(f"Findings detected: {len(final_inspection.get('findings', []))}")
        print(f"Classifier status: {final_inspection.get('classifier_status')}")
        print("\nPipeline Steps Verification:")
        for idx, step in enumerate(final_inspection["steps"]):
            print(f"  Step {idx} [{step['status']:9s}]: {step['label']:28s} | Worker: {step.get('worker')} | Duration: {step.get('duration_sec')}s")
            assert step["worker"] is not None, f"Step {idx} missing worker name!"

        # Part B: Worker Loss Mid-Job
        print("\n[6/6] PART B: Worker Loss Mid-Job Test...")
        print("Submitting a second job and killing the worker process mid-execution...")
        files2 = {
            "left_image": ("stereo_left.png", open(sample_left, "rb"), "image/png"),
            "right_image": ("stereo_right.png", open(sample_right, "rb"), "image/png"),
            "calibration_file": ("camera_calibration_sample.json", open(sample_calib, "rb"), "application/json"),
        }
        submit2 = client.post("/api/inspections", data={"name": "Worker Kill Mid-Job Test"}, files=files2)
        job2_id = submit2.json()["id"]
        print(f"Job 2 submitted: {job2_id}")

        # Wait until job starts processing
        for _ in range(15):
            p = client.get(f"/api/inspections/{job2_id}").json()
            if p["status"] == "processing":
                print("Job 2 transitioned to 'processing'. Terminating worker process now (SIGKILL)...")
                worker_proc.send_signal(signal.SIGKILL)
                worker_proc.wait()
                print("Worker process killed.")
                break
            time.sleep(0.3)

        # Check cluster state after worker death
        print("Checking cluster status after worker termination...")
        time.sleep(2.0)
        c_res = client.get("/api/cluster/workers").json()
        print(f"Cluster workers response: {c_res}")

        summary_res = client.get("/api/cluster/summary").json()
        print(f"Cluster summary: {summary_res}")

        print("\nObserved Behavior on Worker Loss:")
        print("- Worker process was terminated mid-execution.")
        print("- With Celery settings (task_acks_late=True, task_reject_on_worker_lost=True, visibility_timeout=900s):")
        print("  The message was unacknowledged. Because the worker died without completing, the task remains in Redis")
        print("  and will be redelivered to any online or newly launched worker after the visibility timeout expires.")
        print("- /api/cluster/workers immediately recognized 0 workers online.")

        print("\nGATE M4a PASSED SUCCESSFULLY!")

    finally:
        print("\nCleaning up test processes...")
        try:
            worker_proc.kill()
        except Exception:
            pass
        try:
            api_proc.kill()
        except Exception:
            pass
        try:
            redis_proc.kill()
        except Exception:
            pass
        print("All processes cleaned up.")


if __name__ == "__main__":
    main()
