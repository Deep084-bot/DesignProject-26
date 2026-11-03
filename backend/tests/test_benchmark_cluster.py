"""
Unit and Integration Tests for Cluster Benchmark Suite (scripts/benchmark_cluster.py).
Validates:
- Precondition validation & worker count retrieval.
- Job submission & multipart payload encoding.
- Polling lifecycle, completion detection, failure states, timeouts.
- CSV append logic, column headers, historical record preservation.
- Error handling and table formatting.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import benchmark_cluster as bc


@pytest.fixture
def dummy_samples(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    calib = tmp_path / "calib.json"
    left.write_bytes(b"\x89PNG\r\n\x1a\nfakeleft")
    right.write_bytes(b"\x89PNG\r\n\x1a\nfakeright")
    calib.write_text('{"baseline_distance_mm": 120.0}', encoding="utf-8")
    return left, right, calib


def test_resolve_sample_file_finds_valid_and_raises_on_missing(tmp_path):
    f = tmp_path / "sample.png"
    f.write_text("dummy")

    assert bc.resolve_sample_file([f], None) == f
    assert bc.resolve_sample_file([], str(f)) == f

    with pytest.raises(FileNotFoundError):
        bc.resolve_sample_file([tmp_path / "nonexistent.png"], None)

    with pytest.raises(FileNotFoundError):
        bc.resolve_sample_file([], str(tmp_path / "nonexistent.png"))


def test_check_cluster_preconditions_healthy_celery():
    def mock_urlopen(req, timeout=5.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        mock_resp = MagicMock()
        mock_resp.status = 200
        if "api/health" in url:
            mock_resp.read.return_value = json.dumps({"status": "healthy"}).encode()
        elif "api/cluster/workers" in url:
            mock_resp.read.return_value = json.dumps({
                "mode": "celery",
                "workers": [
                    {"name": "worker-1", "online": True},
                    {"name": "worker-2", "online": True},
                    {"name": "worker-3", "online": False},
                ],
            }).encode()
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        healthy, mode, online_workers, err = bc.check_cluster_preconditions("http://mock-master:8000")
        assert healthy is True
        assert mode == "celery"
        assert online_workers == 2
        assert err == ""


def test_check_cluster_preconditions_celery_zero_workers_fails():
    def mock_urlopen(req, timeout=5.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        mock_resp = MagicMock()
        mock_resp.status = 200
        if "api/health" in url:
            mock_resp.read.return_value = json.dumps({"status": "healthy"}).encode()
        elif "api/cluster/workers" in url:
            mock_resp.read.return_value = json.dumps({
                "mode": "celery",
                "workers": [{"name": "worker-1", "online": False}],
            }).encode()
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        healthy, mode, online_workers, err = bc.check_cluster_preconditions("http://mock-master:8000")
        assert healthy is False
        assert mode == "celery"
        assert online_workers == 0
        assert "zero workers are online" in err


def test_check_cluster_preconditions_local_mode():
    def mock_urlopen(req, timeout=5.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        mock_resp = MagicMock()
        mock_resp.status = 200
        if "api/health" in url:
            mock_resp.read.return_value = json.dumps({"status": "healthy"}).encode()
        elif "api/cluster/workers" in url:
            mock_resp.read.return_value = json.dumps({"mode": "local", "workers": []}).encode()
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        healthy, mode, online_workers, err = bc.check_cluster_preconditions("http://mock-master:8000")
        assert healthy is True
        assert mode == "local"
        assert online_workers == 1


def test_check_cluster_preconditions_unreachable_api():
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        healthy, mode, online_workers, err = bc.check_cluster_preconditions("http://mock-master:8000")
        assert healthy is False
        assert "Cannot connect to master API" in err


def test_submit_inspection_success_and_failure(dummy_samples):
    left, right, calib = dummy_samples

    # Success case
    mock_resp = MagicMock()
    mock_resp.status = 201
    mock_resp.read.return_value = json.dumps({"id": "INSP-TEST-01"}).encode()
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        iid, err = bc.submit_inspection("http://mock-master", left, right, calib, 1)
        assert iid == "INSP-TEST-01"
        assert err is None

    # Failure case: HTTP error
    with patch("urllib.request.urlopen", side_effect=Exception("HTTP 500 Internal Error")):
        iid, err = bc.submit_inspection("http://mock-master", left, right, calib, 1)
        assert iid is None
        assert "500" in err


def test_poll_inspections_success_and_failure():
    call_counts = {}

    def mock_urlopen(req, timeout=5.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        mock_resp = MagicMock()
        mock_resp.status = 200
        if "INSP-1" in url:
            call_counts["INSP-1"] = call_counts.get("INSP-1", 0) + 1
            if call_counts["INSP-1"] == 1:
                mock_resp.read.return_value = json.dumps({"status": "processing"}).encode()
            else:
                mock_resp.read.return_value = json.dumps({
                    "status": "completed",
                    "processing_duration_sec": 14.5,
                    "assigned_worker": "worker-1",
                }).encode()
        elif "INSP-2" in url:
            mock_resp.read.return_value = json.dumps({
                "status": "failed",
                "failure_message": "OpenCV stereosgbm failure",
            }).encode()
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        results, timed_out = bc.poll_inspections(
            "http://mock-master", ["INSP-1", "INSP-2"], timeout_sec=5.0, poll_interval_sec=0.01
        )
        assert timed_out is False
        assert results["INSP-1"]["status"] == "completed"
        assert results["INSP-1"]["duration_sec"] == 14.5
        assert results["INSP-2"]["status"] == "failed"
        assert results["INSP-2"]["failure_message"] == "OpenCV stereosgbm failure"


def test_poll_inspections_timeout():
    def mock_urlopen(req, timeout=5.0):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"status": "processing"}).encode()
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        results, timed_out = bc.poll_inspections(
            "http://mock-master", ["INSP-STUCK"], timeout_sec=0.05, poll_interval_sec=0.01
        )
        assert timed_out is True
        assert results["INSP-STUCK"]["status"] == "timed_out"


def test_record_benchmark_csv_creates_and_appends_without_overwriting(tmp_path):
    csv_file = tmp_path / "benchmarks" / "results.csv"

    # 1. First record creates headers and row
    bc.record_benchmark_csv(
        csv_path=csv_file,
        workers_online=2,
        jobs_count=4,
        wall_seconds=50.2,
        avg_job_seconds=12.5,
        jobs_per_minute=4.78,
        failures=0,
    )

    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["workers_online"] == "2"
    assert rows[0]["jobs"] == "4"
    assert rows[0]["wall_seconds"] == "50.2"

    # 2. Second record appends without overwriting
    bc.record_benchmark_csv(
        csv_path=csv_file,
        workers_online=3,
        jobs_count=8,
        wall_seconds=75.4,
        avg_job_seconds=9.4,
        jobs_per_minute=6.36,
        failures=1,
    )

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["jobs"] == "4"
    assert rows[1]["jobs"] == "8"
    assert rows[1]["workers_online"] == "3"
    assert rows[1]["failures"] == "1"


def test_format_benchmark_table_output():
    table = bc.format_benchmark_table(
        workers_online=2,
        jobs_count=8,
        wall_seconds=64.25,
        avg_job_seconds=15.2,
        jobs_per_minute=7.47,
        failures=0,
        mode="celery",
    )
    assert "SURFACE CLUSTER BENCHMARK REPORT" in table
    assert "Online Workers       : 2" in table
    assert "Jobs Submitted       : 8" in table
    assert "Throughput           :     7.47 jobs/min" in table


def test_run_benchmark_end_to_end_mocked(tmp_path, dummy_samples):
    left, right, calib = dummy_samples
    csv_file = tmp_path / "results.csv"

    def mock_urlopen(req, timeout=5.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        mock_resp = MagicMock()
        mock_resp.status = 200
        if "api/health" in url:
            mock_resp.read.return_value = json.dumps({"status": "healthy"}).encode()
        elif "api/cluster/workers" in url:
            mock_resp.read.return_value = json.dumps({
                "mode": "celery",
                "workers": [{"name": "worker-1", "online": True}, {"name": "worker-2", "online": True}],
            }).encode()
        elif "api/inspections" in url and req.get_method() == "POST":
            mock_resp.status = 201
            mock_resp.read.return_value = json.dumps({"id": f"INSP-MOCK-{time.time()}"}).encode()
        elif "api/inspections/" in url:
            mock_resp.read.return_value = json.dumps({
                "status": "completed",
                "processing_duration_sec": 12.0,
            }).encode()
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        summary = bc.run_benchmark(
            api_url="http://mock-master:8000",
            jobs_count=3,
            left_path=left,
            right_path=right,
            calib_path=calib,
            csv_path=csv_file,
            timeout_sec=5.0,
        )

        assert summary["jobs"] == 3
        assert summary["workers_online"] == 2
        assert summary["failures"] == 0
        assert summary["avg_job_seconds"] == 12.0
        assert csv_file.exists()

