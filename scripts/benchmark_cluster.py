#!/usr/bin/env python3
"""
SURFACE — Cluster Benchmark Runner.

Submits N inspection jobs to the running master API, monitors execution until
terminal state, records monotonic wall-clock timings, and appends empirical results
to benchmarks/results.csv.

Required invocation:
    python scripts/benchmark_cluster.py --api http://MASTER_IP:8000 --jobs 8
"""

import argparse
import csv
import datetime
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_LEFT_CANDIDATES = [
    Path("dist/samples/stereo_left.png"),
    Path("public/samples/stereo_left.png"),
]
DEFAULT_RIGHT_CANDIDATES = [
    Path("dist/samples/stereo_right.png"),
    Path("public/samples/stereo_right.png"),
]
DEFAULT_CALIB_CANDIDATES = [
    Path("dist/samples/camera_calibration_sample.json"),
    Path("public/samples/camera_calibration_sample.json"),
]


def resolve_sample_file(candidates: List[Path], custom_path: Optional[str]) -> Path:
    """Finds a valid sample file path from CLI argument or fallback candidates."""
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Specified sample file does not exist: {custom_path}")
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"No valid sample file found among candidates: {candidates}")


def encode_multipart_formdata(fields: Dict[str, str], files: Dict[str, Path]) -> Tuple[bytes, str]:
    """Encodes form fields and file attachments into standard multipart/form-data."""
    import uuid
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))
    for name, filepath in files.items():
        filename = filepath.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def check_cluster_preconditions(api_url: str) -> Tuple[bool, str, int, str]:
    """
    Validates cluster health and fetches online worker count.
    Returns: (is_healthy, mode, online_worker_count, error_message)
    """
    # 1. Health check
    health_url = f"{api_url.rstrip('/')}/api/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status != 200:
                return False, "", 0, f"Master health check returned status {resp.status}"
            data = json.loads(resp.read().decode())
            if data.get("status") != "healthy":
                return False, "", 0, f"Master reported non-healthy status: {data.get('status')}"
    except Exception as exc:
        return False, "", 0, f"Cannot connect to master API at {health_url}: {exc}"

    # 2. Worker count check
    workers_url = f"{api_url.rstrip('/')}/api/cluster/workers"
    try:
        req = urllib.request.Request(workers_url, method="GET")
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            w_data = json.loads(resp.read().decode())
            mode = w_data.get("mode", "unknown")
            workers = w_data.get("workers", [])
            online_count = sum(1 for w in workers if w.get("online"))
            if mode == "celery" and online_count == 0:
                return False, mode, 0, "Cluster is in Celery mode but zero workers are online."
            if mode == "local":
                online_count = 1  # In-process thread pool
            return True, mode, online_count, ""
    except Exception as exc:
        return False, "", 0, f"Failed to retrieve cluster worker status: {exc}"


def submit_inspection(
    api_url: str,
    left_path: Path,
    right_path: Path,
    calib_path: Path,
    job_index: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Submits a single inspection job to /api/inspections.
    Returns: (inspection_id, error_message)
    """
    url = f"{api_url.rstrip('/')}/api/inspections"
    fields = {
        "name": f"Benchmark Run Job {job_index:02d}",
        "surface_type": "Concrete",
        "reconstruction_method": "Automatic",
    }
    files = {
        "left_image": left_path,
        "right_image": right_path,
        "calibration_file": calib_path,
    }
    try:
        body, content_type = encode_multipart_formdata(fields, files)
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode())
            insp_id = data.get("id")
            if not insp_id:
                return None, f"Response missing job ID: {data}"
            return insp_id, None
    except Exception as exc:
        return None, str(exc)


def poll_inspections(
    api_url: str,
    inspection_ids: List[str],
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 1.0,
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """
    Polls submitted inspections until all reach a terminal state ('completed' or 'failed')
    or timeout expires.
    Returns: (job_results_dict, did_time_out)
    """
    active_ids = set(inspection_ids)
    results: Dict[str, Dict[str, Any]] = {
        iid: {"status": "submitted", "duration_sec": None, "error": None}
        for iid in inspection_ids
    }
    start_time = time.monotonic()

    while active_ids and (time.monotonic() - start_time) < timeout_sec:
        for iid in list(active_ids):
            url = f"{api_url.rstrip('/')}/api/inspections/{iid}"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    data = json.loads(resp.read().decode())
                    status = data.get("status")
                    if status in ("completed", "failed"):
                        results[iid]["status"] = status
                        results[iid]["duration_sec"] = data.get("processing_duration_sec")
                        results[iid]["assigned_worker"] = data.get("assigned_worker")
                        results[iid]["failure_message"] = data.get("failure_message")
                        active_ids.remove(iid)
            except Exception as exc:
                # Network or transient error polling status; keep retrying until timeout
                results[iid]["error"] = str(exc)
        if active_ids:
            time.sleep(poll_interval_sec)

    timed_out = len(active_ids) > 0
    for iid in active_ids:
        results[iid]["status"] = "timed_out"

    return results, timed_out


def record_benchmark_csv(
    csv_path: Path,
    workers_online: int,
    jobs_count: int,
    wall_seconds: float,
    avg_job_seconds: float,
    jobs_per_minute: float,
    failures: int,
) -> None:
    """
    Appends the measured benchmark result to results.csv, creating headers if new.
    Preserves all historical rows.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    fieldnames = [
        "timestamp",
        "workers_online",
        "jobs",
        "wall_seconds",
        "avg_job_seconds",
        "jobs_per_minute",
        "failures",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": now_iso,
            "workers_online": workers_online,
            "jobs": jobs_count,
            "wall_seconds": round(wall_seconds, 3),
            "avg_job_seconds": round(avg_job_seconds, 3),
            "jobs_per_minute": round(jobs_per_minute, 2),
            "failures": failures,
        })


def format_benchmark_table(
    workers_online: int,
    jobs_count: int,
    wall_seconds: float,
    avg_job_seconds: float,
    jobs_per_minute: float,
    failures: int,
    mode: str,
) -> str:
    """Formats benchmark metrics into a clean human-readable ASCII table."""
    lines = [
        "=" * 64,
        "             SURFACE CLUSTER BENCHMARK REPORT",
        "=" * 64,
        f"  Execution Mode       : {mode}",
        f"  Online Workers       : {workers_online}",
        f"  Jobs Submitted       : {jobs_count}",
        f"  Successful Jobs      : {jobs_count - failures}",
        f"  Failures / Timeouts  : {failures}",
        "-" * 64,
        f"  Total Wall Clock     : {wall_seconds:8.2f} s",
        f"  Average Job Duration : {avg_job_seconds:8.2f} s",
        f"  Throughput           : {jobs_per_minute:8.2f} jobs/min",
        "=" * 64,
    ]
    return "\n".join(lines)


def run_benchmark(
    api_url: str,
    jobs_count: int,
    left_path: Path,
    right_path: Path,
    calib_path: Path,
    csv_path: Path,
    timeout_sec: float = 900.0,
) -> Dict[str, Any]:
    """Main orchestration function for running the cluster benchmark."""
    print(f"[1/4] Checking cluster health at {api_url}...")
    healthy, mode, online_workers, err_msg = check_cluster_preconditions(api_url)
    if not healthy:
        raise RuntimeError(f"Cluster precondition failure: {err_msg}")

    print(f"      Mode: {mode} | Online workers: {online_workers}")
    print(f"[2/4] Submitting {jobs_count} inspection jobs...")

    t0 = time.monotonic()
    submitted_ids: List[str] = []
    submission_errors = 0

    for idx in range(1, jobs_count + 1):
        iid, err = submit_inspection(api_url, left_path, right_path, calib_path, idx)
        if err or not iid:
            print(f"      Job {idx:02d}: Submission FAILED ({err})")
            submission_errors += 1
        else:
            print(f"      Job {idx:02d}: Submitted -> ID: {iid}")
            submitted_ids.append(iid)

    if not submitted_ids:
        wall_sec = time.monotonic() - t0
        raise RuntimeError("All job submissions failed. Aborting benchmark.")

    print(f"[3/4] Monitoring {len(submitted_ids)} jobs until terminal state...")
    poll_results, timed_out = poll_inspections(
        api_url, submitted_ids, timeout_sec=timeout_sec, poll_interval_sec=1.5
    )
    wall_seconds = time.monotonic() - t0

    # Calculate statistics
    completed_durations: List[float] = []
    failures = submission_errors

    for iid, r in poll_results.items():
        if r.get("status") == "completed":
            dur = r.get("duration_sec")
            if dur is not None and dur > 0:
                completed_durations.append(dur)
        else:
            failures += 1

    if completed_durations:
        avg_job_seconds = sum(completed_durations) / len(completed_durations)
    else:
        avg_job_seconds = wall_seconds / max(1, jobs_count)

    jobs_per_minute = (jobs_count / wall_seconds * 60.0) if wall_seconds > 0 else 0.0

    print(f"[4/4] Writing results to {csv_path}...")
    record_benchmark_csv(
        csv_path=csv_path,
        workers_online=online_workers,
        jobs_count=jobs_count,
        wall_seconds=wall_seconds,
        avg_job_seconds=avg_job_seconds,
        jobs_per_minute=jobs_per_minute,
        failures=failures,
    )

    table = format_benchmark_table(
        workers_online=online_workers,
        jobs_count=jobs_count,
        wall_seconds=wall_seconds,
        avg_job_seconds=avg_job_seconds,
        jobs_per_minute=jobs_per_minute,
        failures=failures,
        mode=mode,
    )
    print("\n" + table + "\n")

    return {
        "mode": mode,
        "workers_online": online_workers,
        "jobs": jobs_count,
        "wall_seconds": wall_seconds,
        "avg_job_seconds": avg_job_seconds,
        "jobs_per_minute": jobs_per_minute,
        "failures": failures,
        "submitted_ids": submitted_ids,
        "poll_results": poll_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SURFACE Cluster Benchmark Suite: Submit N jobs, measure throughput, record metrics."
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL for FastAPI master server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of inspection jobs to submit concurrently (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Maximum total wait time in seconds (default: 900s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results.csv",
        help="Target CSV file for empirical results (default: benchmarks/results.csv)",
    )
    parser.add_argument("--left", type=str, default=None, help="Custom path to left image")
    parser.add_argument("--right", type=str, default=None, help="Custom path to right image")
    parser.add_argument("--calib", type=str, default=None, help="Custom path to calibration JSON")

    args = parser.parse_args()

    try:
        left_path = resolve_sample_file(DEFAULT_LEFT_CANDIDATES, args.left)
        right_path = resolve_sample_file(DEFAULT_RIGHT_CANDIDATES, args.right)
        calib_path = resolve_sample_file(DEFAULT_CALIB_CANDIDATES, args.calib)
    except FileNotFoundError as err:
        print(f"[ERROR] Sample file missing: {err}", file=sys.stderr)
        sys.exit(2)

    csv_path = Path(args.output)

    try:
        run_benchmark(
            api_url=args.api,
            jobs_count=args.jobs,
            left_path=left_path,
            right_path=right_path,
            calib_path=calib_path,
            csv_path=csv_path,
            timeout_sec=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n[WARNING] Benchmark interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] Benchmark execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

