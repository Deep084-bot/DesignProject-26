"""
Unit tests for the Real Multi-Process Integration Verification Harness (M4b Phase 4.2).

Verifies:
- Log parsing for verifiable HTTP artifact downloads and uploads.
- Port discovery and socket readiness helpers.
- Process manager lifecycle and cleanup guarantees.
- Error handling on unreachable services.
"""

from pathlib import Path
import subprocess
import sys
import time
import pytest

from scripts.verify_cluster_e2e import (
    parse_http_transfers_from_log,
    find_free_port,
    wait_for_redis,
    wait_for_api_health,
    ClusterProcessManager,
    find_redis_binary,
)


def test_parse_http_transfers_from_log():
    sample_log = """
127.0.0.1:52101 - "GET /api/health HTTP/1.1" 200 OK
127.0.0.1:52102 - "POST /api/inspections HTTP/1.1" 201 Created
127.0.0.1:52103 - "GET /api/inspections/INSP-TEST-001/artifacts/left_image.png HTTP/1.1" 200 OK
127.0.0.1:52104 - "GET /api/inspections/INSP-TEST-001/artifacts/right_image.png HTTP/1.1" 200 OK
127.0.0.1:52105 - "POST /api/internal/jobs/INSP-TEST-001/artifacts/rectified_left.png HTTP/1.1" 200 OK
127.0.0.1:52106 - "POST /api/internal/jobs/INSP-TEST-001/artifacts/mesh.ply HTTP/1.1" 200 OK
127.0.0.1:52107 - "GET /api/inspections/INSP-TEST-001/artifacts/rectified_left.png HTTP/1.1" 200 OK
127.0.0.1:52108 - "GET /api/inspections/INSP-OTHER-999/artifacts/rectified_left.png HTTP/1.1" 200 OK
127.0.0.1:52109 - "GET /api/inspections/INSP-TEST-001/artifacts/missing.png HTTP/1.1" 404 Not Found
    """
    transfers = parse_http_transfers_from_log(sample_log, "INSP-TEST-001")

    assert transfers["downloads_count"] == 3
    assert transfers["uploads_count"] == 2
    assert "left_image.png" in transfers["download_artifacts"]
    assert "right_image.png" in transfers["download_artifacts"]
    assert "rectified_left.png" in transfers["download_artifacts"]
    assert "mesh.ply" in transfers["upload_artifacts"]
    assert "rectified_left.png" in transfers["upload_artifacts"]

    # Verify other inspection was ignored
    for d in transfers["downloads"]:
        assert "INSP-OTHER-999" not in d["path"]
    # Verify 404 was ignored
    for d in transfers["downloads"]:
        assert "missing.png" not in d["path"]


def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_wait_for_redis_timeout_on_unused_port():
    unused_port = find_free_port()
    # Should safely return False on timeout without unhandled error
    assert wait_for_redis("127.0.0.1", unused_port, timeout=0.2) is False


def test_wait_for_api_health_timeout_on_unused_port():
    unused_port = find_free_port()
    # Should safely return False on timeout without unhandled error
    assert wait_for_api_health(f"http://127.0.0.1:{unused_port}", timeout=0.2) is False


def test_cluster_process_manager_starts_and_stops_cleanly(tmp_path):
    log_dir = tmp_path / "logs"
    proc_mgr = ClusterProcessManager(log_dir)

    # Launch a simple sleep subprocess
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = proc_mgr.start_process("test_sleeper", cmd, {}, tmp_path)

    assert proc.pid in [p.pid for p in proc_mgr.processes.values()]
    assert proc.poll() is None  # Process is running

    # Stop the process
    proc_mgr.stop_process("test_sleeper", timeout=2.0)
    assert proc.poll() is not None  # Process is stopped

    proc_mgr.stop_all()


def test_find_redis_binary_returns_valid_path_or_none():
    redis_bin = find_redis_binary()
    if redis_bin is not None:
        assert redis_bin.exists()
        assert redis_bin.is_file()
