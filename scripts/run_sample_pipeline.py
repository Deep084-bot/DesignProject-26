#!/usr/bin/env python3
"""
Benchmark script: Runs the existing Surface stereo pipeline on the bundled sample pair.

Executes run_pipeline on:
- public/samples/stereo_left.png
- public/samples/stereo_right.png
- public/samples/camera_calibration_sample.json

Outputs are written to a temporary directory without modifying sample or storage files.
Reports per-stage timing, reconstructed point count, mesh status, and peak memory.
Exits with code 0 on success, or 1 on failure.
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict

# Ensure backend package is in python search path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.pipeline.calibration_parser import parse_calibration
from backend.app.pipeline.runner import run_pipeline


def get_peak_memory_mb() -> Optional[float]:
    """Returns peak resident set size in megabytes, or None if unavailable."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS (Darwin) ru_maxrss is in bytes; on Linux it is in kilobytes.
        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024.0 * 1024.0)
        return ru.ru_maxrss / 1024.0
    except Exception:
        return None


def main() -> int:
    samples_dir = REPO_ROOT / "public" / "samples"
    left_path = samples_dir / "stereo_left.png"
    right_path = samples_dir / "stereo_right.png"
    calib_path = samples_dir / "camera_calibration_sample.json"

    for path in (left_path, right_path, calib_path):
        if not path.is_file():
            print(f"Error: Sample file not found: {path}", file=sys.stderr)
            return 1

    print("============================================================")
    print("SURFACE — Sample Pipeline Benchmark (M0)")
    print("============================================================")
    print(f"Left image:  {left_path.name}")
    print(f"Right image: {right_path.name}")
    print(f"Calibration: {calib_path.name}")
    print("------------------------------------------------------------")

    # Parse calibration using the existing calibration parser
    try:
        calib_content = calib_path.read_text(encoding="utf-8")
        calibration, _ = parse_calibration(calib_content, file_name=calib_path.name)
    except Exception as exc:
        print(f"Error: Failed to parse calibration: {exc}", file=sys.stderr)
        return 1

    stage_durations: Dict[str, float] = {}
    stage_start_times: Dict[str, float] = {}

    def on_step_change(idx: int, step_id: str, status: str, msg: Optional[str] = None) -> None:
        now = time.perf_counter()
        if status == "active":
            stage_start_times[step_id] = now
        elif status in ("completed", "failed"):
            start = stage_start_times.get(step_id, now)
            duration = now - start
            stage_durations[step_id] = duration
            status_text = "completed" if status == "completed" else f"FAILED: {msg}"
            print(f"  [{idx}] {step_id:<12} {duration:>6.2f}s  ({status_text})")

    print("Executing pipeline stages:")
    with tempfile.TemporaryDirectory(prefix="surface_benchmark_") as temp_dir:
        job_dir = Path(temp_dir)
        total_start = time.perf_counter()

        result = run_pipeline(
            job_dir=job_dir,
            left_image_path=left_path,
            right_image_path=right_path,
            calibration=calibration,
            on_step_change=on_step_change,
        )

        total_wall_sec = time.perf_counter() - total_start

        # Check artifact existence while inside the temporary directory
        mesh_exists = False
        mesh_size_bytes = 0
        if result.mesh_path and result.mesh_path.exists():
            mesh_exists = True
            mesh_size_bytes = result.mesh_path.stat().st_size

        point_cloud_exists = False
        if result.point_cloud_path and result.point_cloud_path.exists():
            point_cloud_exists = True

    print("------------------------------------------------------------")
    print("Benchmark Results:")
    print(f"  Overall Status:        {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Total Duration:        {result.duration_sec:.2f}s (wall: {total_wall_sec:.2f}s)")
    print(f"  Reconstructed Points:  {result.point_count:,}")
    print(f"  Point Cloud Artifact:  {'EXISTS' if point_cloud_exists else 'MISSING'}")
    print(f"  Mesh Status:           {result.mesh_status}")
    print(f"  Mesh Triangles:        {result.triangle_count if result.triangle_count is not None else 0:,}")
    mesh_size_str = f"{mesh_size_bytes / 1024:.1f} KB" if mesh_exists else "N/A"
    print(f"  Mesh Artifact:         {'EXISTS' if mesh_exists else 'MISSING'} ({mesh_size_str})")

    peak_mb = get_peak_memory_mb()
    if peak_mb is not None:
        print(f"  Peak Memory (RSS):     {peak_mb:.1f} MB")
    else:
        print("  Peak Memory (RSS):     skipped (unavailable on this platform)")

    print("------------------------------------------------------------")
    print("Stage Durations:")
    for stage_id in ("prepare", "depth", "surface", "deviations", "damage", "results"):
        dur = stage_durations.get(stage_id)
        dur_str = f"{dur:.2f}s" if dur is not None else "not executed"
        print(f"  - {stage_id:<12}: {dur_str}")
    print("============================================================")

    if not result.success:
        print(f"Pipeline failed: {result.failure_message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
