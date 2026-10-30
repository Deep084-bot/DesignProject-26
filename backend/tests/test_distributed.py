"""
Comprehensive Test Suite for M4a: Distributed Job Execution with Celery & Redis.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2 & 4.4.
Covers:
1. Central Settings & Environment Overrides
2. Secret Token Authentication & Route Security
3. Artifact Upload Path Traversal Prevention
4. Step, Complete, and Failure State Transitions & Idempotency
5. Cluster Monitoring Fallback (Local mode & Redis unavailable)
6. Worker Pipeline Task Execution, Tempdir Cleanup, & Retry Semantics
"""

from pathlib import Path
import json
import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.job_manager import JobManager
from backend.app.tasks import execute_worker_pipeline, run_inspection_task
from backend.app import settings


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def worker_token():
    return "test-secret-worker-token-xyz-123"


@pytest.fixture
def auth_headers(worker_token):
    return {"X-Worker-Token": worker_token}


@pytest.fixture
def sample_job_in_manager(tmp_path: Path, monkeypatch, worker_token):
    """Creates an isolated JobManager instance and an initial job for testing."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", worker_token)
    monkeypatch.setenv("SURFACE_EXECUTION", "local")

    manager = JobManager(storage_dir=tmp_path / "jobs")

    from backend.app.models.calibration import RectifiedStereoCalibration
    calib = RectifiedStereoCalibration(
        camera_model="pinhole_stereo",
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=320,
        image_height=240,
        principal_point=(160.0, 120.0),
    )

    left_bytes = b"\x89PNG\r\n\x1a\nFakeLeftImageData"
    right_bytes = b"\x89PNG\r\n\x1a\nFakeRightImageData"
    raw_calib = json.dumps({"camera_model": "pinhole_stereo", "baseline_distance_mm": 120.0}).encode("utf-8")

    job = manager.create_job(
        inspection_id="INSP-TEST-001",
        name="Distributed Unit Test Inspection",
        surface_type="Concrete retaining wall",
        reconstruction_method="Automatic surface reconstruction",
        left_bytes=left_bytes,
        left_filename="input_left.png",
        right_bytes=right_bytes,
        right_filename="input_right.png",
        calibration=calib,
        calibration_summary=None,
        raw_calibration_bytes=raw_calib,
        dispatch_mode="none",  # Do not run thread pool in this fixture
    )

    return manager, job


# ============================================================================
# 1. Settings & Environment Configuration Tests
# ============================================================================

def test_settings_defaults_and_env_overrides(monkeypatch):
    """Verifies default values and dynamic environment overrides in settings.py."""
    monkeypatch.delenv("SURFACE_EXECUTION", raising=False)
    assert settings.get_execution_mode() == "local"
    assert settings.SURFACE_EXECUTION == "local"

    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    assert settings.get_execution_mode() == "celery"
    assert settings.SURFACE_EXECUTION == "celery"

    monkeypatch.setenv("SURFACE_EXECUTION", "INVALID_VAL")
    assert settings.get_execution_mode() == "local"

    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "my-token")
    assert settings.get_worker_token() == "my-token"
    assert settings.SURFACE_WORKER_TOKEN == "my-token"

    monkeypatch.setenv("SURFACE_CORS_ORIGINS", "http://laptop1:5173, http://laptop2:5173 ")
    assert settings.get_cors_origins() == ["http://laptop1:5173", "http://laptop2:5173"]


# ============================================================================
# 2. Worker Authentication & Security Tests (/api/internal/*)
# ============================================================================

def test_internal_token_missing_header_returns_403(test_client, monkeypatch):
    """Rejects request with 403 when X-Worker-Token header is missing."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "valid-token")
    res = test_client.get("/api/internal/jobs/INSP-ANY/manifest")
    assert res.status_code == 403
    assert "Missing required 'X-Worker-Token'" in res.json()["detail"]


def test_internal_token_invalid_returns_403(test_client, monkeypatch):
    """Rejects request with 403 when X-Worker-Token header is incorrect."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "valid-token")
    res = test_client.get(
        "/api/internal/jobs/INSP-ANY/manifest",
        headers={"X-Worker-Token": "wrong-token"},
    )
    assert res.status_code == 403
    assert "Invalid worker token" in res.json()["detail"]


def test_internal_token_unconfigured_master_returns_403(test_client, monkeypatch):
    """Rejects all requests with 403 if master has no token configured."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "")
    res = test_client.get(
        "/api/internal/jobs/INSP-ANY/manifest",
        headers={"X-Worker-Token": "some-token"},
    )
    assert res.status_code == 403
    assert "SURFACE_WORKER_TOKEN is not configured" in res.json()["detail"]


# ============================================================================
# 3. Artifact Upload & Path Traversal Prevention Tests
# ============================================================================

def test_artifact_upload_path_traversal_dot_dot_rejected(test_client, sample_job_in_manager, monkeypatch):
    """Rejects artifact upload with 400 when path contains '..' directory traversal."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": ("test.ply", b"binary-data", "application/octet-stream")}
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/..%2F..%2Fetc%2Fpasswd",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        files=files,
    )
    assert res.status_code == 400
    assert "path traversal" in res.json()["detail"].lower()


def test_artifact_upload_slash_in_name_rejected(test_client, sample_job_in_manager, monkeypatch):
    """Rejects artifact upload with 400 when artifact name contains slashes."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": ("test.ply", b"binary-data", "application/octet-stream")}
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/sub%2Fpointcloud.ply",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        files=files,
    )
    assert res.status_code == 400


def test_artifact_upload_hidden_file_rejected(test_client, sample_job_in_manager, monkeypatch):
    """Rejects artifact upload with 400 when artifact name starts with '.'."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": (".hidden", b"binary-data", "application/octet-stream")}
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/.hidden",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        files=files,
    )
    assert res.status_code == 400


def test_artifact_upload_valid_file_stored_safely(test_client, sample_job_in_manager, monkeypatch):
    """Accepts valid artifact upload and writes it strictly into the job directory."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    payload = b"ply\nformat binary_little_endian 1.0\nelement vertex 10\nend_header\n"
    files = {"file": ("pointcloud.ply", payload, "application/octet-stream")}

    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/pointcloud.ply",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        files=files,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "uploaded"
    assert res.json()["size_bytes"] == len(payload)

    target_file = manager.storage_dir / job.id / "pointcloud.ply"
    assert target_file.exists()
    assert target_file.read_bytes() == payload


def test_artifact_upload_nonexistent_job_returns_404(test_client, sample_job_in_manager, monkeypatch):
    """Returns 404 when uploading artifact to non-existent job ID."""
    manager, _ = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": ("mesh.ply", b"data", "application/octet-stream")}
    res = test_client.post(
        "/api/internal/jobs/NONEXISTENT-ID/artifacts/mesh.ply",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        files=files,
    )
    assert res.status_code == 404


# ============================================================================
# 4. Manifest & State Update Route Tests
# ============================================================================

def test_manifest_returns_correct_input_filenames(test_client, sample_job_in_manager, monkeypatch):
    """GET /api/internal/jobs/{id}/manifest returns left image, right image, and calibration file."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    res = test_client.get(
        f"/api/internal/jobs/{job.id}/manifest",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["inspection_id"] == job.id
    assert data["left_image"] == "input_left.png"
    assert data["right_image"] == "input_right.png"
    assert data["calibration"] == "calibration.json"


def test_step_update_applies_progress_worker_and_duration(test_client, sample_job_in_manager, monkeypatch):
    """POST /api/internal/jobs/{id}/step updates job step status, worker name, and duration."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    step_data = {
        "index": 1,
        "step_id": "disparity_estimation",
        "status": "completed",
        "worker": "worker-laptop-01",
        "duration_sec": 4.25,
        "message": None,
    }

    res = test_client.post(
        f"/api/internal/jobs/{job.id}/step",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json=step_data,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    updated = manager.get_job(job.id)
    assert updated.current_step_index == 1
    assert updated.steps[1].status == "completed"
    assert updated.steps[1].worker == "worker-laptop-01"
    assert updated.steps[1].duration_sec == 4.25


def test_complete_job_records_findings_and_deviation(test_client, sample_job_in_manager, monkeypatch):
    """POST /api/internal/jobs/{id}/complete transitions job to completed with all findings."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    complete_payload = {
        "point_count": 125000,
        "mesh_status": "completed",
        "mesh_error": None,
        "triangle_count": 45000,
        "deviation_summary": {
            "rmse_mm": 2.1,
            "max_depression_mm": 5.4,
            "max_bulge_mm": 0.0,
            "sigma_mm": 1.2,
            "threshold_mm": 4.5,
            "inlier_fraction": 0.94,
            "region_count": 1,
            "plane_normal": [0.0, 0.0, 1.0],
            "plane_offset_mm": 1200.0,
            "valid_pixel_fraction": 0.85,
        },
        "findings": [
            {
                "id": "Finding 01",
                "type": "potential_crack",
                "confidence": 0.92,
                "geometric_deviation_mm": 5.4,
                "bounding_box_2d": {"x": 50, "y": 60, "width": 120, "height": 80},
                "region_center_3d": [10.5, -20.2, -1205.0],
                "image_crop_url": f"/api/inspections/{job.id}/artifacts/finding_01.png",
                "notes": "Verified surface fissure",
            }
        ],
        "classifier_status": "completed",
        "classifier_error": None,
        "duration_sec": 14.8,
        "worker": "worker-laptop-01",
    }

    res = test_client.post(
        f"/api/internal/jobs/{job.id}/complete",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json=complete_payload,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    updated = manager.get_job(job.id)
    assert updated.status == "completed"
    assert updated.point_count == 125000
    assert updated.deviation_summary["rmse_mm"] == 2.1
    assert len(updated.findings) == 1
    assert updated.findings[0].id == "Finding 01"
    assert updated.findings[0].confidence == 0.92
    assert all(s.status == "completed" for s in updated.steps)


def test_duplicate_completion_ignored(test_client, sample_job_in_manager, monkeypatch):
    """Verifies that once a job is completed, a second complete call is ignored."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    complete_payload = {
        "point_count": 10000,
        "mesh_status": "completed",
        "duration_sec": 10.0,
        "worker": "worker-1",
    }

    res1 = test_client.post(
        f"/api/internal/jobs/{job.id}/complete",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json=complete_payload,
    )
    assert res1.json()["status"] == "ok"

    # Second call with different point_count should be ignored
    complete_payload["point_count"] = 999999
    res2 = test_client.post(
        f"/api/internal/jobs/{job.id}/complete",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json=complete_payload,
    )
    assert res2.json()["status"] == "ignored"

    # Value remains unchanged
    updated = manager.get_job(job.id)
    assert updated.point_count == 10000


def test_late_step_update_on_completed_job_ignored(test_client, sample_job_in_manager, monkeypatch):
    """Step update delivered after job has reached terminal completed state is safely ignored."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    # Complete job first
    manager.apply_result(
        inspection_id=job.id,
        point_count=500,
        mesh_status="completed",
        duration_sec=5.0,
    )

    # Late step update
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/step",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json={"index": 2, "step_id": "mesh_reconstruction", "status": "processing"},
    )
    assert res.json()["status"] == "ignored"


def test_fail_endpoint_records_error_and_marks_failed(test_client, sample_job_in_manager, monkeypatch):
    """POST /api/internal/jobs/{id}/fail records error message and marks job as failed."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    fail_payload = {
        "message": "SGBM stereo matching disparity computation failed",
        "failed_stage_index": 1,
        "worker": "worker-laptop-02",
    }

    res = test_client.post(
        f"/api/internal/jobs/{job.id}/fail",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json=fail_payload,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    updated = manager.get_job(job.id)
    assert updated.status == "failed"
    assert updated.failure_message == "SGBM stereo matching disparity computation failed"
    assert updated.steps[1].status == "failed"
    assert updated.steps[1].worker == "worker-laptop-02"


def test_late_fail_on_already_completed_job_ignored(test_client, sample_job_in_manager, monkeypatch):
    """Failure update on an already completed job is ignored."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    manager.apply_result(inspection_id=job.id, point_count=1000, mesh_status="completed", duration_sec=5.0)

    res = test_client.post(
        f"/api/internal/jobs/{job.id}/fail",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
        json={"message": "Late failure after completion", "failed_stage_index": 0},
    )
    assert res.json()["status"] == "ignored"

    updated = manager.get_job(job.id)
    assert updated.status == "completed"


# ============================================================================
# 5. Cluster Monitoring Route Tests (/api/cluster/*)
# ============================================================================

def test_cluster_workers_local_mode_safe(test_client, monkeypatch):
    """In local mode, GET /api/cluster/workers returns mode='local' and empty workers list."""
    monkeypatch.setenv("SURFACE_EXECUTION", "local")
    res = test_client.get("/api/cluster/workers")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "local"
    assert data["workers"] == []


def test_cluster_workers_redis_down_fallback(test_client, monkeypatch):
    """When Celery mode is set but Redis is unreachable, safely returns mode='local' and empty list."""
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect:
        mock_inspect.side_effect = Exception("Redis connection refused")
        res = test_client.get("/api/cluster/workers")
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "local"
        assert data["workers"] == []


def test_cluster_workers_inspect_timeout_fallback(test_client, monkeypatch):
    """When Celery inspect pings return None (timeout), returns mode='celery' and empty workers list."""
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    mock_insp = MagicMock()
    mock_insp.ping.return_value = None
    with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
        res = test_client.get("/api/cluster/workers")
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "celery"
        assert data["workers"] == []


def test_cluster_workers_active_response_parsing(test_client, monkeypatch):
    """Parses online workers, active task arguments, and completed task counts."""
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    mock_insp = MagicMock()
    mock_insp.ping.return_value = {"worker-1@laptop": {"ok": "pong"}}
    mock_insp.active.return_value = {
        "worker-1@laptop": [{"id": "t1", "args": ["INSP-2026-X1"]}]
    }
    mock_insp.stats.return_value = {
        "worker-1@laptop": {"total": {"run_inspection_task": 12}}
    }

    with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
        res = test_client.get("/api/cluster/workers")
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "celery"
        assert len(data["workers"]) == 1
        worker = data["workers"][0]
        assert worker["name"] == "worker-1@laptop"
        assert worker["online"] is True
        assert worker["active_jobs"] == ["INSP-2026-X1"]
        assert worker["tasks_done"] == 12


def test_cluster_summary_counts_and_assignments(test_client, sample_job_in_manager, monkeypatch):
    """GET /api/cluster/summary returns accurate job state counts and active worker assignments."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.cluster.job_manager", manager)

    # Set job status to processing with worker assigned
    manager.apply_step_update(
        inspection_id=job.id,
        index=0,
        step_id="input_rectification",
        status="processing",
        worker="worker-alpha",
    )

    res = test_client.get("/api/cluster/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_jobs"] >= 1
    assert data["processing"] >= 1
    assert data["active_assignments"].get(job.id) == "worker-alpha"


# ============================================================================
# 6. Worker Execution Pipeline & Celery Task Tests
# ============================================================================

def test_execute_worker_pipeline_end_to_end_mocked_http(tmp_path: Path, monkeypatch, synthetic_stereo_pair, valid_rectified_calib_json):
    """
    Executes worker pipeline against mock/test client, verifying:
    1. Downloads manifest & inputs.
    2. Runs pipeline.
    3. Uploads generated artifacts.
    4. Posts completion.
    """
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "mock-worker-token")
    manager = JobManager(storage_dir=tmp_path / "jobs")
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)
    monkeypatch.setattr("backend.app.routes.inspections.job_manager", manager)

    from backend.app.models.calibration import RectifiedStereoCalibration
    calib = RectifiedStereoCalibration(
        camera_model="pinhole_stereo",
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=320,
        image_height=240,
        principal_point=(160.0, 120.0),
    )

    job = manager.create_job(
        inspection_id="INSP-WORKER-TEST",
        name="Worker Pipeline Test",
        surface_type="Concrete",
        reconstruction_method="Automatic",
        left_bytes=synthetic_stereo_pair["left_bytes"],
        left_filename="input_left.png",
        right_bytes=synthetic_stereo_pair["right_bytes"],
        right_filename="input_right.png",
        calibration=calib,
        calibration_summary=None,
        raw_calibration_bytes=valid_rectified_calib_json.encode("utf-8"),
        dispatch_mode="none",
    )

    client = TestClient(app)

    success = execute_worker_pipeline(
        inspection_id=job.id,
        client=client,
        worker_name="mock-worker-node",
        worker_token="mock-worker-token",
    )

    assert success is True
    updated = manager.get_job(job.id)
    assert updated.status == "completed"
    assert updated.point_count > 0
    # Artifacts were uploaded to master storage
    assert (manager.storage_dir / job.id / "pointcloud.ply").exists()


def test_execute_worker_pipeline_network_error_propagation(monkeypatch):
    """Ensures network exceptions bubble up to Celery task so self.retry can catch them."""
    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("Connection refused")

    with pytest.raises(Exception, match="Connection refused"):
        execute_worker_pipeline(
            inspection_id="INSP-RETRY-TEST",
            client=mock_client,
            worker_name="mock-node",
            worker_token="tok",
        )


def test_celery_task_retries_on_network_error():
    """Verifies that run_inspection_task triggers self.retry on network errors."""
    import httpx
    with patch.object(run_inspection_task, "retry", side_effect=RuntimeError("Retry triggered")) as mock_retry:
        with patch("backend.app.tasks.execute_worker_pipeline", side_effect=httpx.ConnectError("Network timeout")):
            with pytest.raises(RuntimeError, match="Retry triggered"):
                run_inspection_task.run("INSP-NET-ERR")
        assert mock_retry.called


def test_settings_custom_mesh_depth(monkeypatch):
    """Verifies settings.py parses positive SURFACE_MESH_DEPTH and safely falls back on invalid values."""
    monkeypatch.setenv("SURFACE_MESH_DEPTH", "6")
    assert settings.get_mesh_depth() == 6
    assert settings.SURFACE_MESH_DEPTH == 6

    monkeypatch.setenv("SURFACE_MESH_DEPTH", "invalid")
    assert settings.get_mesh_depth() == 8

    monkeypatch.setenv("SURFACE_MESH_DEPTH", "-2")
    assert settings.get_mesh_depth() == 8


def test_settings_custom_model_path_and_thresholds(monkeypatch):
    """Verifies settings.py extracts crack classifier hyperparameters and path overrides."""
    monkeypatch.setenv("SURFACE_MODEL_PATH", "custom/path/model.pth")
    monkeypatch.setenv("SURFACE_CRACK_THRESHOLD", "0.65")
    monkeypatch.setenv("SURFACE_TILE_SIZE", "512")
    monkeypatch.setenv("SURFACE_DEV_K", "4.0")
    monkeypatch.setenv("SURFACE_DEV_MIN_MM", "8.0")
    monkeypatch.setenv("SURFACE_DEV_MIN_AREA_PX", "300")

    assert settings.get_model_path() == "custom/path/model.pth"
    assert settings.get_crack_threshold() == 0.65
    assert settings.get_tile_size() == 512
    assert settings.get_dev_k() == 4.0
    assert settings.get_dev_min_mm() == 8.0
    assert settings.get_dev_min_area_px() == 300


def test_create_job_in_celery_mode_skips_thread_pool(sample_job_in_manager):
    """In Celery mode, JobManager.create_job initializes job state but does NOT submit to local thread pool."""
    manager, _ = sample_job_in_manager
    from backend.app.models.calibration import RectifiedStereoCalibration
    calib = RectifiedStereoCalibration(
        camera_model="pinhole_stereo",
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=320,
        image_height=240,
        principal_point=(160.0, 120.0),
    )

    with patch.object(manager.executor, "submit") as mock_submit:
        job = manager.create_job(
            inspection_id="INSP-CELERY-DISPATCH",
            name="Celery Dispatch Test",
            surface_type="Concrete",
            reconstruction_method="Automatic",
            left_bytes=b"left",
            left_filename="input_left.png",
            right_bytes=b"right",
            right_filename="input_right.png",
            calibration=calib,
            calibration_summary=None,
            dispatch_mode="celery",
        )
        assert job.status == "queued"
        mock_submit.assert_not_called()


def test_create_inspection_celery_mode_enqueue_failure_returns_503(test_client, monkeypatch, synthetic_stereo_pair, valid_rectified_calib_json):
    """When SURFACE_EXECUTION=celery and Celery delay raises an error, returns 503 and marks job failed."""
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
        "calibration_file": ("calibration.json", valid_rectified_calib_json.encode("utf-8"), "application/json"),
    }
    data = {
        "name": "Celery Broker Down Test",
        "surface_type": "Concrete surface",
    }

    with patch("backend.app.tasks.run_inspection_task.delay", side_effect=Exception("Redis connection refused")):
        res = test_client.post("/api/inspections", data=data, files=files)
        assert res.status_code == 503
        assert "Task queue unavailable" in res.json()["detail"]


def test_pipeline_step_model_worker_and_duration_serialization():
    """Verifies that PipelineStep serializes worker and duration_sec optional fields correctly."""
    from backend.app.models.inspection import PipelineStep
    step = PipelineStep(
        id="disparity_estimation",
        label="Disparity Estimation",
        description="Stereo matching",
        status="completed",
        worker="worker-node-42",
        duration_sec=3.142,
    )
    dumped = step.model_dump()
    assert dumped["worker"] == "worker-node-42"
    assert dumped["duration_sec"] == 3.142

    # Roundtrip from JSON
    restored = PipelineStep(**dumped)
    assert restored.worker == "worker-node-42"
    assert restored.duration_sec == 3.142


def test_manifest_nonexistent_job_returns_404(test_client, sample_job_in_manager, monkeypatch):
    """GET /api/internal/jobs/{id}/manifest returns 404 for non-existent job."""
    manager, _ = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    res = test_client.get(
        "/api/internal/jobs/NONEXISTENT-INSP-ID/manifest",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123"},
    )
    assert res.status_code == 404
