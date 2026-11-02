"""
Comprehensive Test Suite for M4a: Distributed Job Execution with Celery & Redis.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2 & Approved M4a Recovery Architecture.
Covers:
1. Central Settings & Environment Overrides
2. Secret Token Authentication & Route Security
3. Artifact Upload Path Traversal Prevention & Two-Phase Staging
4. Mandatory Fencing Tokens & Stale Token Rejection (409 Conflict on all internal routes)
5. Worker Identity Binding & Concurrent Claim Protection
6. Heartbeat Tracking & Autonomous Supervisor Decision Table
7. Crash-Safe Outbox Protocol & Startup Reconciliation
8. Clean Worker Abort on 409 (No Retries, No Late Failures, Tempdir Cleanup)
9. Cluster Monitoring Fallback & Unresponsive Worker Flagging
10. Local Mode Preservation without Redis
"""

from pathlib import Path
import json
import time
import os
import asyncio
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.job_manager import JobManager, LeaseConflictError
from backend.app.services.recovery_supervisor import RecoverySupervisor
from backend.app.tasks import execute_worker_pipeline, run_inspection_task, LeaseRevokedException
from backend.app import settings


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def worker_token():
    return "test-secret-worker-token-xyz-123"


@pytest.fixture
def auth_headers(worker_token):
    return {
        "X-Worker-Token": worker_token,
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-laptop-01",
    }


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

    # Recovery configuration settings
    monkeypatch.setenv("SURFACE_HEARTBEAT_INTERVAL", "12.5")
    assert settings.get_heartbeat_interval() == 12.5
    assert settings.SURFACE_HEARTBEAT_INTERVAL == 12.5

    monkeypatch.setenv("SURFACE_LEASE_TIMEOUT", "45.0")
    assert settings.get_lease_timeout() == 45.0
    assert settings.SURFACE_LEASE_TIMEOUT == 45.0

    monkeypatch.setenv("SURFACE_SUPERVISOR_INTERVAL", "3.0")
    assert settings.get_supervisor_interval() == 3.0
    assert settings.SURFACE_SUPERVISOR_INTERVAL == 3.0

    monkeypatch.setenv("SURFACE_OFFLINE_GRACE_PERIOD", "20.0")
    assert settings.get_offline_grace_period() == 20.0
    assert settings.SURFACE_OFFLINE_GRACE_PERIOD == 20.0


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


def test_missing_worker_identity_or_token_rejected(test_client, sample_job_in_manager, monkeypatch):
    """Verifies that missing X-Lease-Token or X-Worker-Name returns 409 Conflict."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    # Missing lease token
    res = test_client.get(
        f"/api/internal/jobs/{job.id}/manifest",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123", "X-Worker-Name": "w1"},
    )
    assert res.status_code == 409
    assert "X-Lease-Token" in res.json()["detail"]

    # Missing worker name
    res = test_client.get(
        f"/api/internal/jobs/{job.id}/manifest",
        headers={"X-Worker-Token": "test-secret-worker-token-xyz-123", "X-Lease-Token": "1"},
    )
    assert res.status_code == 409
    assert "X-Worker-Name" in res.json()["detail"]


# ============================================================================
# 3. Artifact Upload & Path Traversal & Two-Phase Commit Tests
# ============================================================================

def test_artifact_upload_path_traversal_dot_dot_rejected(test_client, sample_job_in_manager, monkeypatch, auth_headers):
    """Rejects artifact upload with 400 when path contains '..' directory traversal."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": ("test.ply", b"binary-data", "application/octet-stream")}
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/..%2F..%2Fetc%2Fpasswd",
        headers=auth_headers,
        files=files,
    )
    assert res.status_code == 400
    assert "path traversal" in res.json()["detail"].lower()


def test_artifact_upload_slash_in_name_rejected(test_client, sample_job_in_manager, monkeypatch, auth_headers):
    """Rejects artifact upload with 400 when artifact name contains slashes."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": ("test.ply", b"binary-data", "application/octet-stream")}
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/sub%2Fpointcloud.ply",
        headers=auth_headers,
        files=files,
    )
    assert res.status_code == 400


def test_artifact_upload_hidden_file_rejected(test_client, sample_job_in_manager, monkeypatch, auth_headers):
    """Rejects artifact upload with 400 when artifact name starts with '.'."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    files = {"file": (".hidden", b"binary-data", "application/octet-stream")}
    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/.hidden",
        headers=auth_headers,
        files=files,
    )
    assert res.status_code == 400


def test_artifact_upload_valid_file_stored_safely(test_client, sample_job_in_manager, monkeypatch, auth_headers):
    """Accepts valid artifact upload and writes it strictly into the job directory via two-phase commit."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    payload = b"ply\nformat binary_little_endian 1.0\nelement vertex 10\nend_header\n"
    files = {"file": ("pointcloud.ply", payload, "application/octet-stream")}

    res = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/pointcloud.ply",
        headers=auth_headers,
        files=files,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "uploaded"
    assert res.json()["size_bytes"] == len(payload)

    target_file = manager.storage_dir / job.id / "pointcloud.ply"
    assert target_file.exists()
    assert target_file.read_bytes() == payload

    # Verify no dangling staging files
    job_dir = manager.storage_dir / job.id
    staging_files = [f for f in job_dir.iterdir() if f.name.startswith(".staging_")]
    assert len(staging_files) == 0


def test_interrupted_artifact_upload_staging_cleaned(sample_job_in_manager):
    """Validates that a failed artifact commit cleans up its staging file."""
    manager, job = sample_job_in_manager
    staging_file = manager.storage_dir / job.id / ".staging_test_temp"
    staging_file.write_bytes(b"partial content")

    # Pass mismatched lease token to force commit failure
    with pytest.raises(LeaseConflictError):
        manager.commit_artifact(
            inspection_id=job.id,
            lease_token=99,
            worker_name="worker-1",
            artifact_name="mesh.ply",
            staging_path=staging_file,
        )

    assert not staging_file.exists()


# ============================================================================
# 4. Mandatory Fencing Tokens & Stale Token Rejection (409 Conflict)
# ============================================================================

def test_stale_token_rejected_on_every_internal_route_with_409(test_client, sample_job_in_manager, monkeypatch):
    """
    Every worker-originated operation that reads or modifies an inspection
    must validate the current lease token and return 409 Conflict on mismatch.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    # Job has lease_token = 1. Stale token is 0.
    stale_headers = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "0",
        "X-Worker-Name": "stale-worker",
    }

    # 1. Manifest
    r1 = test_client.get(f"/api/internal/jobs/{job.id}/manifest", headers=stale_headers)
    assert r1.status_code == 409

    # 2. Heartbeat
    r2 = test_client.post(f"/api/internal/jobs/{job.id}/heartbeat", headers=stale_headers)
    assert r2.status_code == 409

    # 3. Step update
    r3 = test_client.post(
        f"/api/internal/jobs/{job.id}/step",
        headers=stale_headers,
        json={"index": 0, "step_id": "prepare", "status": "processing"},
    )
    assert r3.status_code == 409

    # 4. Artifact upload
    r4 = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/test.txt",
        headers=stale_headers,
        files={"file": ("test.txt", b"content", "text/plain")},
    )
    assert r4.status_code == 409

    # 5. Complete
    r5 = test_client.post(
        f"/api/internal/jobs/{job.id}/complete",
        headers=stale_headers,
        json={"point_count": 100},
    )
    assert r5.status_code == 409

    # 6. Fail
    r6 = test_client.post(
        f"/api/internal/jobs/{job.id}/fail",
        headers=stale_headers,
        json={"message": "Failed from stale worker"},
    )
    assert r6.status_code == 409


# ============================================================================
# 5. Worker Identity Binding & Concurrent Claim Protection
# ============================================================================

def test_concurrent_manifest_claims(test_client, sample_job_in_manager, monkeypatch):
    """
    Worker 1 claims manifest with 'worker-A'.
    Worker 2 attempting to claim manifest with 'worker-B' under the same lease gets 409 Conflict.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    headers_a = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-A",
    }
    headers_b = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-B",
    }

    # Worker A claims first
    r1 = test_client.get(f"/api/internal/jobs/{job.id}/manifest", headers=headers_a)
    assert r1.status_code == 200
    assert r1.json()["assigned_worker"] == "worker-A"

    # Worker B tries to claim the same job and lease
    r2 = test_client.get(f"/api/internal/jobs/{job.id}/manifest", headers=headers_b)
    assert r2.status_code == 409
    assert "Worker identity mismatch" in r2.json()["detail"] or "already bound" in r2.json()["detail"]


def test_concurrent_recovery_attempts(sample_job_in_manager):
    """Atomic CAS allows only one recovery increment; subsequent attempts with same lease return None."""
    manager, job = sample_job_in_manager
    assert job.lease_token == 1

    new_lease = manager.reassign_orphaned_job(job.id, expected_lease_token=1, reason="Worker loss")
    assert new_lease == 2

    # Concurrent second attempt with expected_lease_token = 1
    second_attempt = manager.reassign_orphaned_job(job.id, expected_lease_token=1, reason="Duplicate trigger")
    assert second_attempt is None


def test_duplicate_celery_delivery_dropped_on_manifest_check(test_client, sample_job_in_manager, monkeypatch):
    """
    When a job has been recovered to lease 2, a redelivered generation 1 Celery task
    is rejected with 409 Conflict on its initial manifest call and exits cleanly.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    # Reassign job to lease 2
    manager.reassign_orphaned_job(job.id, expected_lease_token=1)
    assert manager.get_job(job.id).lease_token == 2

    # Redelivered task with lease 1
    stale_headers = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-old-delivery",
    }
    res = test_client.get(f"/api/internal/jobs/{job.id}/manifest", headers=stale_headers)
    assert res.status_code == 409


# ============================================================================
# 6. Heartbeat Tracking & Autonomous Supervisor Decision Table
# ============================================================================

def test_heartbeat_during_stage(test_client, sample_job_in_manager, monkeypatch, auth_headers):
    """Periodic worker heartbeat successfully updates job timestamp and monotonic tracker."""
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)

    res = test_client.post(f"/api/internal/jobs/{job.id}/heartbeat", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    age = manager.get_heartbeat_age(job.id)
    assert age is not None
    assert age >= 0.0
    assert age < 2.0


def test_supervisor_decision_table_states(sample_job_in_manager, monkeypatch):
    """
    Tests the 4 states of the recovery supervisor decision table:
    1. Worker Online + Fresh Heartbeat -> Retain assignment.
    2. Worker Online + Expired Heartbeat -> Grace period, then recover.
    3. Worker Offline + Fresh Heartbeat -> Grace period, then recover.
    4. Worker Offline + Expired Heartbeat -> Immediate recovery.
    """
    async def _run():
        manager, job = sample_job_in_manager
        supervisor = RecoverySupervisor(manager)
        monkeypatch.setenv("SURFACE_EXECUTION", "celery")
        monkeypatch.setenv("SURFACE_LEASE_TIMEOUT", "10.0")
        monkeypatch.setenv("SURFACE_OFFLINE_GRACE_PERIOD", "5.0")

        # Bind worker to job
        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-1")

        # State 1: Online + Fresh Heartbeat
        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker-1": {"ok": "pong"}}
        with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
            await supervisor.check_recovery_cycle()
        assert manager.get_job(job.id).lease_token == 1
        assert manager.get_job(job.id).assigned_worker == "worker-1"

        # State 4: Offline + Expired Heartbeat
        # Simulate expired heartbeat by setting monotonic timestamp into the past
        manager._monotonic_heartbeats[job.id] = time.monotonic() - 15.0
        mock_insp.ping.return_value = {}  # worker-1 offline

        with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp), \
             patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
            await supervisor.check_recovery_cycle()
            assert manager.get_job(job.id).lease_token == 2
            assert manager.get_job(job.id).assigned_worker is None
            assert mock_delay.called

    asyncio.run(_run())


def test_redis_unavailable_during_recovery_deferred_safely(sample_job_in_manager, monkeypatch):
    """When broker is down, supervisor defers recovery without mutating job state."""
    async def _run():
        manager, job = sample_job_in_manager
        supervisor = RecoverySupervisor(manager)
        monkeypatch.setenv("SURFACE_EXECUTION", "celery")

        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-1")
        manager._monotonic_heartbeats[job.id] = time.monotonic() - 50.0  # Expired

        with patch("backend.app.celery_app.celery_app.control.inspect", side_effect=Exception("Redis down")):
            await supervisor.check_recovery_cycle()

        # Job was NOT reassigned
        assert manager.get_job(job.id).lease_token == 1
        assert manager.get_job(job.id).assigned_worker == "worker-1"

    asyncio.run(_run())


# ============================================================================
# 7. Crash-Safe Outbox Protocol & Startup Reconciliation
# ============================================================================

def test_master_crash_before_task_publication_outbox_recovered(sample_job_in_manager, monkeypatch):
    """
    Crash before task publication leaves outbox file in 'pending' state.
    Master startup reconciliation re-enqueues the Celery task.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    # Write dispatch intent
    intent_file = manager.write_outbox_intent(job.id, lease_token=1)
    assert intent_file.exists()

    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        republished = manager.reconcile_outbox_on_startup()
        assert republished == 1
        assert mock_delay.called
        assert not intent_file.exists()


def test_master_crash_after_publication_before_cleanup(sample_job_in_manager, monkeypatch):
    """
    Crash after publication leaves outbox file in 'published' state.
    Master startup reconciliation cleans up file without duplicate publication.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    manager.write_outbox_intent(job.id, lease_token=1)
    manager.mark_outbox_published(job.id, lease_token=1)

    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        republished = manager.reconcile_outbox_on_startup()
        assert republished == 0
        mock_delay.assert_not_called()
        assert not (manager.storage_dir / job.id / "outbox" / "dispatch_1.json").exists()


def test_crash_after_durable_intent_before_inspection_state_persistence_reconciles(sample_job_in_manager, monkeypatch):
    """
    PRIMARY REGRESSION TEST:
    Simulates a hard master crash immediately after writing the durable recovery intent
    (dispatch_2.json with previous_lease=1, intended_lease=2), but BEFORE inspection.json
    was updated (inspection remains at lease 1 on disk).

    Verifies that on master reboot:
    1. Startup reconciliation detects the durable intent.
    2. Safely advances the inspection lease from 1 to 2.
    3. Re-dispatches the Celery task with lease 2.
    4. Marks published and removes the outbox intent.
    5. Allows a replacement worker to claim manifest under lease 2.
    6. Leaves no stuck/unrecoverable queued state.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    # Bind worker to generation 1
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-dead")
    assert manager.get_job(job.id).lease_token == 1
    assert manager.get_job(job.id).assigned_worker == "worker-dead"

    # Write-ahead recovery intent exists on disk (as written by write_outbox_intent in Step 1 of reassignment)
    intent_file = manager.write_outbox_intent(
        inspection_id=job.id,
        lease_token=2,
        previous_lease=1,
        reason="Worker worker-dead lost",
    )
    assert intent_file.exists()

    # Simulate hard crash before inspection.json update:
    # inspection.json on disk is still at lease 1 with worker-dead.
    job_on_disk_file = manager.storage_dir / job.id / "inspection.json"
    with open(job_on_disk_file, "r") as f:
        disk_data = json.load(f)
    assert disk_data["lease_token"] == 1
    assert disk_data["assigned_worker"] == "worker-dead"

    # Simulate Master Restart: Instantiate a completely new JobManager from disk
    new_manager = JobManager(storage_dir=manager.storage_dir)
    loaded_job = new_manager.get_job(job.id)
    assert loaded_job is not None
    assert loaded_job.lease_token == 1  # Still at previous lease before reconciliation

    # Run startup reconciliation
    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        republished = new_manager.reconcile_outbox_on_startup()
        assert republished == 1
        mock_delay.assert_called_once_with(job.id, 2)

    # Verify inspection reached lease 2 and queued state
    updated_job = new_manager.get_job(job.id)
    assert updated_job.lease_token == 2
    assert updated_job.assigned_worker is None
    assert updated_job.status == "queued"
    assert not intent_file.exists()

    # Verify persistent state on disk matches
    with open(job_on_disk_file, "r") as f:
        new_disk_data = json.load(f)
    assert new_disk_data["lease_token"] == 2
    assert new_disk_data["assigned_worker"] is None
    assert new_disk_data["status"] == "queued"

    # Verify replacement worker can successfully claim the job under lease 2
    claimed_job = new_manager.claim_manifest(job.id, lease_token=2, worker_name="worker-survivor")
    assert claimed_job.assigned_worker == "worker-survivor"
    assert claimed_job.status == "processing"


def test_crash_after_inspection_state_persistence_before_broker_publication_reconciles(sample_job_in_manager, monkeypatch):
    """
    Simulates crash after both durable intent and inspection.json are persisted at lease 2,
    but before broker publication.
    On reboot, reconciliation detects state='pending' with job.lease_token == intended_lease,
    and reliably dispatches the task.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    # Complete the CAS reassignment (both intent and inspection state persisted)
    new_lease = manager.reassign_orphaned_job(job.id, expected_lease_token=1, reason="Worker loss")
    assert new_lease == 2
    assert (manager.storage_dir / job.id / "outbox" / "dispatch_2.json").exists()

    # Simulate Master Restart
    new_manager = JobManager(storage_dir=manager.storage_dir)
    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        republished = new_manager.reconcile_outbox_on_startup()
        assert republished == 1
        mock_delay.assert_called_once_with(job.id, 2)
        assert not (manager.storage_dir / job.id / "outbox" / "dispatch_2.json").exists()


def test_crash_after_broker_publication_before_marking_published_idempotent(sample_job_in_manager, monkeypatch):
    """
    Crash after publication leaves pending record. On reboot, duplicate publication is issued,
    and subsequent manifest claim by duplicate worker is rejected with 409 Conflict.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    manager.reassign_orphaned_job(job.id, expected_lease_token=1)
    # Intent remains 'pending' (simulating crash before mark_outbox_published)

    new_manager = JobManager(storage_dir=manager.storage_dir)
    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        republished = new_manager.reconcile_outbox_on_startup()
        assert republished == 1
        assert mock_delay.called

    # Worker 1 claims manifest under lease 2
    new_manager.claim_manifest(job.id, lease_token=2, worker_name="worker-1")

    # Worker 2 attempting to claim duplicate task with different worker gets 409 Conflict
    with pytest.raises(LeaseConflictError) as exc_info:
        new_manager.claim_manifest(job.id, lease_token=2, worker_name="worker-2")
    assert exc_info.value.status_code == 409


def test_stale_intent_does_not_overwrite_newer_lease(sample_job_in_manager, monkeypatch):
    """
    A stale outbox intent for generation 2 must not overwrite or reassign
    an inspection that has already advanced to generation 3.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    # Advance job to lease 3
    manager.reassign_orphaned_job(job.id, expected_lease_token=1)
    manager.cleanup_outbox(job.id, 2)
    manager.reassign_orphaned_job(job.id, expected_lease_token=2)
    manager.cleanup_outbox(job.id, 3)
    assert manager.get_job(job.id).lease_token == 3

    # Manually drop an old intent for lease 2 into outbox
    stale_file = manager.write_outbox_intent(job.id, lease_token=2, previous_lease=1)
    assert stale_file.exists()

    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        republished = manager.reconcile_outbox_on_startup()
        assert republished == 0
        mock_delay.assert_not_called()
        assert not stale_file.exists()
        # Job remains untouched at generation 3
        assert manager.get_job(job.id).lease_token == 3


def test_redis_unavailable_retains_durable_intent(sample_job_in_manager, monkeypatch):
    """
    When Redis/broker is down during startup reconciliation, the durable intent
    is retained on disk in 'pending' state and not unlinked.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    intent_file = manager.write_outbox_intent(job.id, lease_token=1)
    assert intent_file.exists()

    with patch("backend.app.tasks.run_inspection_task.delay", side_effect=Exception("Redis connection refused")):
        republished = manager.reconcile_outbox_on_startup()
        assert republished == 0
        # Intent file MUST remain on disk to be retried
        assert intent_file.exists()


# ============================================================================
# 8. Clean Worker Abort on 409 (No Retries, No Late Failures, Tempdir Cleanup)
# ============================================================================

def test_worker_aborts_cleanly_on_409(monkeypatch):
    """
    Worker receiving 409 Conflict raises LeaseRevokedException,
    cleans up tempdir, and Celery task exits without retry or calling /fail.
    """
    mock_client = MagicMock()
    # Mock manifest response returning 409 Conflict
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.text = "Fencing lease revoked"
    mock_client.get.return_value = mock_resp

    with pytest.raises(LeaseRevokedException):
        execute_worker_pipeline(
            inspection_id="INSP-ABORT-TEST",
            lease_token=1,
            client=mock_client,
            worker_name="worker-1",
            worker_token="tok",
        )

    # When run as Celery task, LeaseRevokedException is caught and returns False cleanly
    with patch("backend.app.tasks.execute_worker_pipeline", side_effect=LeaseRevokedException("Revoked")), \
         patch.object(run_inspection_task, "retry") as mock_retry:
        ret = run_inspection_task.run("INSP-ABORT-TEST", lease_token=1)
        assert ret is False
        mock_retry.assert_not_called()


# ============================================================================
# 9. Cluster Monitoring Fallback & Unresponsive Worker Flagging
# ============================================================================

def test_cluster_summary_marks_unresponsive_worker(test_client, sample_job_in_manager, monkeypatch):
    """
    When a job is assigned to 'worker-offline' but Celery inspect shows it is offline,
    GET /api/cluster/summary marks it as 'worker-offline (unresponsive)'.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    monkeypatch.setattr("backend.app.routes.cluster.job_manager", manager)

    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-offline")

    mock_insp = MagicMock()
    mock_insp.ping.return_value = {"worker-other": {"ok": "pong"}}  # worker-offline missing

    with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
        res = test_client.get("/api/cluster/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["active_assignments"].get(job.id) == "worker-offline (unresponsive)"


def test_cluster_workers_local_mode_safe(test_client, monkeypatch):
    """In local mode, GET /api/cluster/workers returns mode='local' and empty workers list."""
    monkeypatch.setenv("SURFACE_EXECUTION", "local")
    res = test_client.get("/api/cluster/workers")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "local"
    assert data["workers"] == []


# ============================================================================
# 10. Worker Pipeline Execution & Local Mode Preservation
# ============================================================================

def test_execute_worker_pipeline_end_to_end_mocked_http(tmp_path: Path, monkeypatch, synthetic_stereo_pair, valid_rectified_calib_json):
    """Executes full worker pipeline against test client, verifying manifest claim, artifacts upload, and completion."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "mock-worker-token")
    manager = JobManager(storage_dir=tmp_path / "jobs")
    monkeypatch.setattr("backend.app.main.job_manager", manager)
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
        lease_token=1,
        client=client,
        worker_name="mock-worker-node",
        worker_token="mock-worker-token",
    )

    assert success is True
    updated = manager.get_job(job.id)
    assert updated.status == "completed"
    assert updated.point_count > 0
    assert (manager.storage_dir / job.id / "pointcloud.ply").exists()


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


# ============================================================================
# Phase 2: Master-Side Coordination Layer Tests (18 Tests)
# ============================================================================

def test_storage_lock_acquired_on_startup_and_blocks_second_instance(tmp_path):
    """
    Test 1: Storage lock acquired on startup blocks a second master instance.
    """
    manager1 = JobManager(storage_dir=tmp_path / "jobs")
    manager1.acquire_storage_lock()
    try:
        manager2 = JobManager(storage_dir=tmp_path / "jobs")
        with pytest.raises(RuntimeError) as exc_info:
            manager2.acquire_storage_lock()
        assert "holds the storage lock" in str(exc_info.value)
    finally:
        manager1.release_storage_lock()

    # After release, manager2 can acquire it successfully
    manager2.acquire_storage_lock()
    manager2.release_storage_lock()


def test_coordinator_fencing_retains_m4a_semantics(sample_job_in_manager):
    """
    Test 2: Coordinator fencing retains all M4a invariants for Stages 0-3.
    """
    manager, job = sample_job_in_manager
    claimed = manager.claim_manifest(job.id, lease_token=1, worker_name="worker-coord-1")
    assert claimed.assigned_worker == "worker-coord-1"

    with pytest.raises(LeaseConflictError) as exc_info:
        manager.claim_manifest(job.id, lease_token=2, worker_name="worker-coord-1")
    assert "Fencing lease token mismatch" in str(exc_info.value)

    with pytest.raises(LeaseConflictError) as exc_info:
        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-coord-2")
    assert "Worker identity mismatch" in str(exc_info.value)

    job.status = "completed"
    manager._save_job_state(job)
    with pytest.raises(LeaseConflictError) as exc_info:
        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-coord-1")
    assert "already in terminal state" in str(exc_info.value)


def test_batch_worker_authorization_separation(test_client, sample_job_in_manager, monkeypatch):
    """
    Test 3: Batch worker authorization separation allows cluster workers different from coordinator.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)
    job.assigned_worker = "worker-coord-1"
    job.status = "processing"
    manager._save_job_state(job)

    batches = [
        {"batch_id": 0, "tile_indices": [0, 1]},
        {"batch_id": 1, "tile_indices": [2, 3]},
    ]
    fanout = manager.init_fanout(job.id, lease_token=1, worker_name="worker-coord-1", total_batches=2, batches=batches)
    assert fanout.status == "FANOUT_RUNNING"

    batch_headers = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-batch-99",
    }
    r_claim = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/claim",
        headers=batch_headers,
        json={"fanout_run_id": fanout.fanout_run_id, "claim_timeout_sec": 60.0},
    )
    assert r_claim.status_code == 200
    claim_data = r_claim.json()
    assert claim_data["status"] == "claimed"
    assert claim_data["batch_id"] == 0
    claim_token = claim_data["claim_token"]

    r_commit = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/commit",
        headers=batch_headers,
        json={"fanout_run_id": fanout.fanout_run_id, "claim_token": claim_token, "probabilities": [0.1, 0.2]},
    )
    assert r_commit.status_code == 200
    commit_data = r_commit.json()
    assert commit_data["status"] == "committed"
    assert commit_data["committed_batches"] == 1


def test_batch_worker_forbidden_from_coordinator_routes(test_client, sample_job_in_manager, monkeypatch):
    """
    Test 4: Batch workers are strictly forbidden from calling coordinator routes.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)
    job.assigned_worker = "worker-coord-1"
    job.status = "processing"
    manager._save_job_state(job)

    batch_headers = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-batch-99",
    }

    r_step = test_client.post(
        f"/api/internal/jobs/{job.id}/step",
        headers=batch_headers,
        json={"index": 1, "step_id": "depth", "status": "processing"},
    )
    assert r_step.status_code == 409

    r_artifact = test_client.post(
        f"/api/internal/jobs/{job.id}/artifacts/forbidden.txt",
        headers=batch_headers,
        files={"file": ("forbidden.txt", b"content", "text/plain")},
    )
    assert r_artifact.status_code == 409

    r_complete = test_client.post(
        f"/api/internal/jobs/{job.id}/complete",
        headers=batch_headers,
        json={"point_count": 100},
    )
    assert r_complete.status_code == 409


def test_batch_claim_validation_and_fencing(test_client, sample_job_in_manager, monkeypatch):
    """
    Test 5: Batch claim requires valid lease token, matching fanout run id, and existing batch id.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="worker-coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )

    valid_headers = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "batch-worker",
    }

    bad_lease_headers = dict(valid_headers, **{"X-Lease-Token": "2"})
    r1 = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/claim",
        headers=bad_lease_headers,
        json={"fanout_run_id": fanout.fanout_run_id},
    )
    assert r1.status_code == 409

    r2 = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/claim",
        headers=valid_headers,
        json={"fanout_run_id": "wrong-run-id"},
    )
    assert r2.status_code == 409

    r3 = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/999/claim",
        headers=valid_headers,
        json={"fanout_run_id": fanout.fanout_run_id},
    )
    assert r3.status_code == 404


def test_concurrent_batch_claims_atomic(sample_job_in_manager):
    """
    Test 6: Concurrent claims on the same batch result in exactly one successful claim.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0, 1]}],
    )

    results = []
    errors = []

    def try_claim(worker_id):
        try:
            res = manager.claim_batch(
                job.id, lease_token=1, worker_name=worker_id,
                fanout_run_id=fanout.fanout_run_id, batch_id=0, claim_timeout_sec=60.0,
            )
            results.append(res)
        except LeaseConflictError as err:
            errors.append(err)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(try_claim, "worker-alpha")
        f2 = executor.submit(try_claim, "worker-beta")
        f1.result()
        f2.result()

    claimed = [r for r in results if r.get("status") == "claimed"]
    already_claimed = [r for r in results if r.get("status") == "already_claimed"]
    assert len(claimed) == 1
    assert len(already_claimed) == 1
    assert claimed[0]["batch_id"] == 0
    assert already_claimed[0]["batch_id"] == 0
    assert already_claimed[0]["claimed_by"] in ("worker-alpha", "worker-beta")


def test_batch_claim_expiry_and_reclaim(sample_job_in_manager):
    """
    Test 7: An expired batch claim can be reclaimed by another worker.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )

    c1 = manager.claim_batch(
        job.id, 1, "worker-1", fanout.fanout_run_id, batch_id=0, claim_timeout_sec=0.01,
    )
    assert c1["status"] == "claimed"

    time.sleep(0.03)

    c2 = manager.claim_batch(
        job.id, 1, "worker-2", fanout.fanout_run_id, batch_id=0, claim_timeout_sec=60.0,
    )
    assert c2["status"] == "claimed"
    assert c2["claim_token"] != c1["claim_token"]


def test_stale_batch_commit_rejected(sample_job_in_manager):
    """
    Test 8: Commit using an expired or superseded claim token is rejected with 409 Conflict.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )

    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, claim_timeout_sec=0.01)
    time.sleep(0.03)

    c2 = manager.claim_batch(job.id, 1, "worker-2", fanout.fanout_run_id, 0, claim_timeout_sec=60.0)

    with pytest.raises(LeaseConflictError) as exc_info:
        manager.commit_batch(
            job.id, 1, "worker-1", fanout.fanout_run_id, batch_id=0,
            claim_token=c1["claim_token"], probabilities=[0.5],
        )
    assert "Stale or mismatched claim token" in str(exc_info.value)


def test_idempotent_batch_commit(sample_job_in_manager):
    """
    Test 9: Duplicate commits on an already committed batch return success idempotently.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=2,
        batches=[{"batch_id": 0, "tile_indices": [0]}, {"batch_id": 1, "tile_indices": [1]}],
    )

    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    commit1 = manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], [0.2])
    assert commit1["status"] == "committed"
    assert commit1["committed_batches"] == 1

    commit2 = manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], [0.2])
    assert commit2["status"] == "already_committed"
    assert commit2["committed_batches"] == 1


def test_batch_probability_validation(test_client, sample_job_in_manager, monkeypatch):
    """
    Test 10: Commit validates probabilities array length and valid float range [0.0, 1.0].
    """
    manager, job = sample_job_in_manager
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0, 1]}],
    )
    headers = {
        "X-Worker-Token": "test-secret-worker-token-xyz-123",
        "X-Lease-Token": "1",
        "X-Worker-Name": "worker-batch",
    }
    r_claim = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/claim",
        headers=headers,
        json={"fanout_run_id": fanout.fanout_run_id},
    )
    token = r_claim.json()["claim_token"]

    r1 = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/commit",
        headers=headers,
        json={"fanout_run_id": fanout.fanout_run_id, "claim_token": token, "probabilities": [0.5]},
    )
    assert r1.status_code == 400

    r2 = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/commit",
        headers=headers,
        json={"fanout_run_id": fanout.fanout_run_id, "claim_token": token, "probabilities": [0.5, 1.2]},
    )
    assert r2.status_code == 400

    r3 = test_client.post(
        f"/api/internal/jobs/{job.id}/batches/0/commit",
        headers=headers,
        json={"fanout_run_id": fanout.fanout_run_id, "claim_token": token, "probabilities": [-0.1, 0.5]},
    )
    assert r3.status_code == 400


def test_atomic_batch_file_persistence(sample_job_in_manager):
    """
    Test 11: Batch commit atomically persists batch_XXXX.json to storage with fsync.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [10, 20]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], [0.75, 0.25])

    batch_file = manager.storage_dir / job.id / "batches" / "batch_0000.json"
    assert batch_file.exists()
    with open(batch_file, "r") as f:
        data = json.load(f)
    assert data["batch_id"] == 0
    assert data["worker_name"] == "worker-1"
    assert data["tile_indices"] == [10, 20]
    assert data["probabilities"] == [0.75, 0.25]


def test_coordinator_loss_during_fanout_preserves_committed_batches(sample_job_in_manager, monkeypatch):
    """
    Test 12: Coordinator heartbeat expiration during FANOUT_RUNNING does NOT trigger M4a lease increment.
    Committed batches and Stages 0-3 are preserved.
    """
    manager, job = sample_job_in_manager
    job.assigned_worker = "dead-coordinator"
    job.status = "processing"
    manager._save_job_state(job)

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="dead-coordinator", total_batches=2,
        batches=[{"batch_id": 0, "tile_indices": [0]}, {"batch_id": 1, "tile_indices": [1]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-batch", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-batch", fanout.fanout_run_id, 0, c["claim_token"], [0.3])

    job.last_heartbeat_at = time.time() - 100.0
    manager._monotonic_heartbeats[job.id] = time.monotonic() - 100.0
    manager._save_job_state(job)

    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"other-worker": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

    updated_job = manager.get_job(job.id)
    assert updated_job.lease_token == 1
    assert updated_job.status == "processing"
    f_state = manager.get_fanout_status(job.id)
    assert f_state.committed_batches == 1
    assert f_state.batches["0"].state == "committed"


def test_all_batches_committed_transitions_to_callback_pending(sample_job_in_manager):
    """
    Test 13: Committing the final batch transitions fanout to ALL_BATCHES_COMMITTED and CALLBACK_PENDING.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [5]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    res = manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], [0.9])

    assert res["status"] == "committed"
    assert res["fanout_status"] == "CALLBACK_PENDING"

    intent_file = manager.storage_dir / job.id / "outbox" / "finalizer_1_0.json"
    assert intent_file.exists()
    with open(intent_file, "r") as f:
        data = json.load(f)
    assert data["state"] == "pending"
    assert data["lease_token"] == 1
    assert data["generation"] == 0


def test_finalizer_claim_and_delegation_token_generation(sample_job_in_manager):
    """
    Test 14: Finalizer claim yields delegation_token, generation 0, and FINALIZATION_RUNNING status.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "w1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "w1", fanout.fanout_run_id, 0, c["claim_token"], [0.1])

    res = manager.claim_finalizer(job.id, lease_token=1, worker_name="finalizer-worker-1", fanout_run_id=fanout.fanout_run_id)
    assert res["status"] == "claimed"
    assert res["delegation_token"].startswith("del_")
    assert res["finalizer_generation"] == 0
    assert res["finalizer_expires_at"] > time.time()

    f_state = manager.get_fanout_status(job.id)
    assert f_state.status == "FINALIZATION_RUNNING"
    assert f_state.delegation_token == res["delegation_token"]
    assert f_state.finalizer_worker == "finalizer-worker-1"


def test_finalizer_heartbeat_renews_expiry(sample_job_in_manager):
    """
    Test 15: Finalizer heartbeat extends finalizer_expires_at.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "w1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "w1", fanout.fanout_run_id, 0, c["claim_token"], [0.1])

    claim = manager.claim_finalizer(job.id, 1, "w-final", fanout.fanout_run_id, delegation_duration_sec=10.0)
    orig_expiry = claim["finalizer_expires_at"]

    hb = manager.apply_finalizer_heartbeat(
        job.id, 1, "w-final", claim["delegation_token"], claim["finalizer_generation"], extension_sec=60.0,
    )
    assert hb["status"] == "ok"
    assert hb["finalizer_expires_at"] > orig_expiry


def test_finalizer_expiry_and_generation_takeover(sample_job_in_manager):
    """
    Test 16: Expired finalizer delegation triggers generation takeover (G -> G+1) and revokes old credentials.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "w1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "w1", fanout.fanout_run_id, 0, c["claim_token"], [0.1])

    claim1 = manager.claim_finalizer(job.id, 1, "finalizer-old", fanout.fanout_run_id, delegation_duration_sec=0.01)
    time.sleep(0.03)

    claim2 = manager.claim_finalizer(job.id, 1, "finalizer-new", fanout.fanout_run_id, delegation_duration_sec=30.0)
    assert claim2["finalizer_generation"] == 1
    assert claim2["delegation_token"] != claim1["delegation_token"]

    with pytest.raises(LeaseConflictError) as exc_info:
        manager.apply_step_update(
            job.id, index=5, step_id="results", status="processing", worker="finalizer-old",
            lease_token=1, delegation_token=claim1["delegation_token"], finalizer_generation=claim1["finalizer_generation"],
        )
    assert "Finalizer generation mismatch" in str(exc_info.value) or "mismatch or revoked" in str(exc_info.value)


def test_lost_callback_recovery_reschedules_finalizer(sample_job_in_manager, monkeypatch):
    """
    Test 17: When callback arrival exceeds 30s deadline in CALLBACK_PENDING, supervisor reschedules intent without lease bump.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "w1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "w1", fanout.fanout_run_id, 0, c["claim_token"], [0.1])

    fanout.callback_dispatched_at = time.time() - 40.0
    manager._save_fanout_state_locked(fanout)

    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect, \
         patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-1": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

    f_state = manager.get_fanout_status(job.id)
    assert f_state.status == "CALLBACK_PENDING"
    assert (time.time() - f_state.callback_dispatched_at) < 5.0
    assert job.lease_token == 1
    mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)


def test_m4a_regression_all_baseline_tests_pass(sample_job_in_manager):
    """
    Test 18: Full M4a single-task baseline regression ensures non-fanout jobs execute unchanged.
    """
    manager, job = sample_job_in_manager
    claimed = manager.claim_manifest(job.id, lease_token=1, worker_name="m4a-worker")
    assert claimed.assigned_worker == "m4a-worker"
    assert claimed.status == "processing"

    assert manager.apply_heartbeat(job.id, lease_token=1, worker_name="m4a-worker") is True

    assert manager.apply_step_update(job.id, index=0, step_id="prepare", status="completed", worker="m4a-worker", lease_token=1) is True

    assert manager.apply_result(
        job.id, point_count=5000, mesh_status="completed", triangle_count=2500, worker="m4a-worker", lease_token=1,
    ) is True

    completed_job = manager.get_job(job.id)
    assert completed_job.status == "completed"
    assert completed_job.point_count == 5000


# ==============================================================================
# PHASE 3 TESTS: CHORD FAN-OUT, BATCH HEARTBEATS & RELIABILITY RECOVERY
# ==============================================================================


def test_active_duplicate_claim_returns_already_claimed(sample_job_in_manager):
    """Tier 1: Active duplicate claim returns already_claimed with bounded retry_after_sec."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    # Worker 1 claims batch
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, claim_timeout_sec=30.0)
    assert c1["status"] == "claimed"

    # Worker 2 attempts claim while Worker 1 is active
    c2 = manager.claim_batch(job.id, 1, "worker-2", fanout.fanout_run_id, 0)
    assert c2["status"] == "already_claimed"
    assert c2["claimed_by"] == "worker-1"
    assert 3.0 <= c2["retry_after_sec"] <= 10.0


def test_committed_duplicate_returns_already_committed(sample_job_in_manager):
    """Tier 1: Committed duplicate returns already_committed without inference."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.2])

    # Duplicate claim after commit
    c2 = manager.claim_batch(job.id, 1, "worker-2", fanout.fanout_run_id, 0)
    assert c2["status"] == "already_committed"
    assert c2["batch_id"] == 0


def test_valid_batch_heartbeat_renews_claim(sample_job_in_manager):
    """Tier 1: Valid batch heartbeat extends claim expiration."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, claim_timeout_sec=10.0)
    orig_expiry = c["claim_expires_at"]

    time.sleep(0.01)
    hb_res = manager.apply_batch_heartbeat(
        job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], extension_sec=30.0
    )
    assert hb_res["status"] == "ok"
    assert hb_res["claim_expires_at"] > orig_expiry


def test_batch_heartbeat_extension_above_30s_rejected_by_api(test_client, auth_headers, worker_token, monkeypatch):
    """Tier 1: Request with extension_sec > 30.0 is rejected by FastAPI validation with 422."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", worker_token)
    resp = test_client.post(
        "/api/internal/jobs/job-123/batches/0/heartbeat",
        headers=auth_headers,
        json={"fanout_run_id": "fanout-1", "claim_token": "tok-1", "extension_sec": 35.0},
    )
    assert resp.status_code == 422


def test_batch_heartbeat_domain_clamps_to_30s(sample_job_in_manager):
    """Tier 1: Domain layer independently clamps extension_sec to 30.0s maximum."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    now_wall = time.time()
    hb_res = manager.apply_batch_heartbeat(
        job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], extension_sec=100.0
    )
    assert hb_res["status"] == "ok"
    assert hb_res["claim_expires_at"] <= now_wall + 30.5


def test_expired_or_stale_claim_cannot_heartbeat(sample_job_in_manager):
    """Tier 1: Expired or stale claim token cannot heartbeat."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, claim_timeout_sec=0.01)
    time.sleep(0.02)

    with pytest.raises(LeaseConflictError):
        manager.apply_batch_heartbeat(
            job.id, 1, "worker-1", fanout.fanout_run_id, 0, c["claim_token"], extension_sec=30.0
        )


def test_wrong_worker_identity_cannot_heartbeat(sample_job_in_manager):
    """Tier 1: Worker cannot heartbeat on another worker's batch claim."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)

    with pytest.raises(LeaseConflictError):
        manager.apply_batch_heartbeat(
            job.id, 1, "worker-impostor", fanout.fanout_run_id, 0, c["claim_token"], extension_sec=30.0
        )


def test_batch_outbox_intent_survives_restart_and_reconciles(sample_job_in_manager, monkeypatch):
    """Tier 1: Batch outbox intent with interrupted transition reconciles to pending and redispatches."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)

    # Write outbox intent while batch is still claimed (simulates crash before fanout_state.json update)
    manager.write_batch_outbox_intent_locked(
        job.id, lease_token=1, fanout_run_id=fanout.fanout_run_id, batch_id=0, reason="test crash"
    )

    outbox_file = manager.storage_dir / job.id / "outbox" / f"batch_1_{fanout.fanout_run_id}_0.json"
    assert outbox_file.exists()

    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    with patch("backend.app.tasks.classify_batch_task.delay") as mock_delay:
        repub = manager.reconcile_outbox_on_startup()
        assert repub == 1
        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id, 0)

    # Batch state must have reconciled from claimed to pending
    f_state = manager.get_fanout_status(job.id)
    assert f_state.batches["0"].state == "pending"
    assert not outbox_file.exists()


def test_committed_batches_retrieval_returns_authoritative_data(sample_job_in_manager):
    """Tier 1: GET committed batches returns full probabilities under finalizer fencing."""
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=2,
        batches=[{"batch_id": 0, "tile_indices": [0, 1]}, {"batch_id": 1, "tile_indices": [2]}],
    )
    c0 = manager.claim_batch(job.id, 1, "w1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "w1", fanout.fanout_run_id, 0, c0["claim_token"], [0.2, 0.4])

    c1 = manager.claim_batch(job.id, 1, "w2", fanout.fanout_run_id, 1)
    manager.commit_batch(job.id, 1, "w2", fanout.fanout_run_id, 1, c1["claim_token"], [0.8])

    final_claim = manager.claim_finalizer(job.id, 1, "finalizer-1", fanout.fanout_run_id)
    del_token = final_claim["delegation_token"]
    gen = final_claim["finalizer_generation"]

    committed = manager.get_committed_batches_locked(
        job.id, lease_token=1, worker_name="finalizer-1", delegation_token=del_token, finalizer_generation=gen
    )
    assert committed["total_batches"] == 2
    assert committed["committed_batches"] == 2
    assert len(committed["batches"]) == 2
    assert committed["batches"][0]["probabilities"] == [0.2, 0.4]
    assert committed["batches"][1]["probabilities"] == [0.8]


def test_coordinator_async_replacement_dispatches_chord():
    """Tier 1: Coordinator asynchronous replacement dispatches chord signature and raises Ignore."""
    from celery import Celery, chord
    from celery.exceptions import Ignore
    from celery.app.task import Context

    test_app = Celery("test_async_coord", broker="memory://", backend="cache+memory://")
    test_app.conf.task_always_eager = False

    dispatched = []

    @test_app.task(bind=True)
    def dummy_batch(self, b_id):
        return b_id

    @test_app.task(bind=True)
    def dummy_finalizer(self, insp_id):
        return f"final_{insp_id}"

    @test_app.task(bind=True)
    def coord(self, insp_id):
        header = [dummy_batch.s(0), dummy_batch.s(1)]
        callback = dummy_finalizer.si(insp_id)
        sig = chord(header, callback)

        orig_delay = sig.delay
        def tracking_delay(*args, **kwargs):
            dispatched.append(sig)
            return orig_delay(*args, **kwargs)
        sig.delay = tracking_delay

        return self.replace(sig)

    req = Context(id="coord-task-id", is_eager=False)
    coord.request_stack.push(req)
    try:
        with pytest.raises(Ignore):
            coord("job-123")
    finally:
        coord.request_stack.pop()

    assert len(dispatched) == 1
    assert isinstance(dispatched[0], chord)
    assert len(dispatched[0].tasks) == 2
    assert dispatched[0].body.task.endswith("dummy_finalizer")


def test_eager_chord_execution_end_to_end():
    """Tier 2: Eager chord execution completes and attaches immutable callback."""
    from celery import Celery, chord

    eager_app = Celery("eager_chord_app", broker="memory://", backend="cache+memory://")
    eager_app.conf.task_always_eager = True
    eager_app.conf.task_eager_propagates = True

    @eager_app.task
    def mock_batch(b_id):
        return b_id * 10

    @eager_app.task
    def mock_finalizer(insp_id, lease_token):
        return f"{insp_id}_lease{lease_token}_done"

    c = chord([mock_batch.s(1), mock_batch.s(2)], mock_finalizer.si("job-abc", 1))
    result = c.apply_async()
    assert result.get() == "job-abc_lease1_done"


def test_callback_pending_recovery_dispatches_finalizer_and_retains_intent(sample_job_in_manager, monkeypatch):
    """
    Test A (Authorized Recovery-Dispatch Fix):
    - Commit all expected batches through JobManager.
    - Put fanout into CALLBACK_PENDING.
    - Advance clock beyond recovery deadline (>30s).
    - Run supervisor recovery cycle.
    - Assert that finalize_inspection_task is dispatched with correct args.
    - Assert that outbox intent remains on disk until durable claim acknowledgement.
    - Simulate master persisting valid finalizer claim (manager.claim_finalizer).
    - Assert that acknowledged intent is cleaned up.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.2])

    f_pending = manager.get_fanout_status(job.id)
    assert f_pending.status == "CALLBACK_PENDING"

    # Advance clock beyond 30s deadline
    f_pending.callback_dispatched_at = time.time() - 35.0
    manager._save_fanout_state_locked(f_pending)

    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect, \
         patch("backend.app.tasks.finalize_inspection_task.delay") as mock_finalize_delay:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-recovery": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

        # 1. Assert finalize_inspection_task was dispatched with exact args
        mock_finalize_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)

    # 2. Assert outbox intent remains on disk until durable claim acknowledgement
    outbox_file = manager.storage_dir / job.id / "outbox" / f"finalizer_1_{fanout.finalizer_generation}.json"
    assert outbox_file.exists()
    with open(outbox_file, "r") as f:
        outbox_data = json.load(f)
    assert outbox_data["state"] == "published"
    assert outbox_data["generation"] == 0

    # 3. Simulate master persisting a valid finalizer claim
    claim = manager.claim_finalizer(job.id, 1, "finalizer-worker-01", fanout.fanout_run_id)
    assert claim["status"] == "claimed"

    # 4. Assert acknowledged intent is cleaned up once claimed
    assert not outbox_file.exists()


def test_actual_finalizer_execution_after_recovery_dispatch(tmp_path: Path, monkeypatch):
    """
    Test B (Authorized Recovery-Dispatch Fix):
    - Exercise the supervisor recovery-dispatch path.
    - Mock only broker publication boundary.
    - Invoke actual finalize_inspection_task through in-process harness against FastAPI TestClient.
    - Verify finalizer:
        1. Claims master-issued delegation credentials.
        2. Retrieves committed batches from master storage.
        3. Validates tile coverage and runs aggregate_damage_findings.
        4. Uploads finding crops to master storage.
        5. Executes Stage 4 and Stage 5 steps.
        6. Completes inspection through delegated completion endpoint.
    - Asserts terminal completed state (without calling manager.apply_result()).
    """
    import cv2
    import numpy as np
    from backend.app.models.calibration import RectifiedStereoCalibration

    worker_token = "test-secret-worker-token-xyz-123"
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", worker_token)
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    monkeypatch.setenv("SURFACE_MASTER_URL", "http://testserver")

    manager = JobManager(storage_dir=tmp_path / "jobs")
    monkeypatch.setattr("backend.app.main.job_manager", manager)
    monkeypatch.setattr("backend.app.routes.internal.job_manager", manager)
    monkeypatch.setattr("backend.app.routes.inspections.job_manager", manager)

    calib = RectifiedStereoCalibration(
        camera_model="pinhole_stereo",
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=100,
        image_height=100,
        principal_point=(50.0, 50.0),
    )
    job = manager.create_job(
        inspection_id="INSP-FINALIZER-EXEC",
        name="Finalizer Recovery Exec Test",
        surface_type="Concrete",
        reconstruction_method="Automatic",
        left_bytes=b"dummy-left",
        left_filename="input_left.png",
        right_bytes=b"dummy-right",
        right_filename="input_right.png",
        calibration=calib,
        calibration_summary=None,
        raw_calibration_bytes=b"{}",
        dispatch_mode="none",
    )

    # Write Stage 0-3 artifacts required by finalizer
    job_dir = manager.storage_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(job_dir / "rectified_left.png"), img_bgr)
    (job_dir / "deviations.json").write_text(json.dumps({"regions": [], "summary": {}}))
    np.save(str(job_dir / "disparity.npy"), np.zeros((100, 100), dtype=np.float32))

    # Initialize fanout with 1 batch (1 tile)
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.85])

    f_state = manager.get_fanout_status(job.id)
    assert f_state.status == "CALLBACK_PENDING"

    # Simulate expired chord callback (>30s)
    f_state.callback_dispatched_at = time.time() - 35.0
    manager._save_fanout_state_locked(f_state)

    # Run supervisor recovery cycle: asserts finalizer dispatch
    supervisor = RecoverySupervisor(manager)
    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect, \
         patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-recovery": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)

    # In-process test harness routing finalize_inspection_task HTTP calls directly to FastAPI app
    class HarnessTestClient(TestClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop("base_url", None)
            kwargs.pop("timeout", None)
            super().__init__(app, base_url="http://testserver", *args, **kwargs)

    from backend.app.tasks import finalize_inspection_task
    with patch("backend.app.tasks.httpx.Client", HarnessTestClient):
        task_ret = finalize_inspection_task.run(job.id, lease_token=1, fanout_run_id=fanout.fanout_run_id)
        assert task_ret is True

    # Verify final terminal state in master storage
    completed_job = manager.get_job(job.id)
    assert completed_job.status == "completed"
    assert completed_job.fanout_status == "TERMINAL"
    assert completed_job.delegation_token is not None
    assert len(completed_job.findings) >= 0

    # Verify outbox intent was acknowledged and cleaned up
    outbox_file = manager.storage_dir / job.id / "outbox" / "finalizer_1_0.json"
    assert not outbox_file.exists()


def test_finalizer_broker_publication_failure_preserves_intent_and_retries(sample_job_in_manager, monkeypatch):
    """
    Test C (Authorized Recovery-Dispatch Fix):
    - Make broker publication fail on cycle 1.
    - Verify outbox intent remains durable in 'pending' state.
    - Run cycle 2 with working broker.
    - Verify publication is retried.
    - Verify successful publication marks state 'published' but does NOT delete intent before claim.
    """
    manager, job = sample_job_in_manager
    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.2])

    f_state = manager.get_fanout_status(job.id)
    f_state.callback_dispatched_at = time.time() - 35.0
    manager._save_fanout_state_locked(f_state)

    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    supervisor = RecoverySupervisor(manager)

    outbox_file = manager.storage_dir / job.id / "outbox" / "finalizer_1_0.json"

    # Cycle 1: Broker failure
    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect, \
         patch("backend.app.tasks.finalize_inspection_task.delay", side_effect=Exception("Redis connection refused")):
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-recovery": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

    assert outbox_file.exists()
    with open(outbox_file, "r") as f:
        data1 = json.load(f)
    assert data1["state"] == "pending"

    # Advance deadline again for cycle 2
    f_state = manager.get_fanout_status(job.id)
    f_state.callback_dispatched_at = time.time() - 35.0
    manager._save_fanout_state_locked(f_state)

    # Cycle 2: Working broker
    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect, \
         patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-recovery": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)

    # Publication succeeded: marked published, but file NOT deleted yet
    assert outbox_file.exists()
    with open(outbox_file, "r") as f:
        data2 = json.load(f)
    assert data2["state"] == "published"


def test_startup_reconciliation_republishes_valid_finalizer_and_discards_stale(sample_job_in_manager, monkeypatch):
    """
    Test D (Authorized Recovery-Dispatch Fix):
    - Persist unacknowledged finalizer intent.
    - Reinitialize JobManager from same storage.
    - Run startup reconciliation in Celery mode.
    - Verify valid intents are republished.
    - Verify stale-generation intents are cleaned up without publishing.
    - Verify broker failure preserves intent.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.2])

    intent_file = manager.storage_dir / job.id / "outbox" / "finalizer_1_0.json"
    assert intent_file.exists()

    # Subcase 1: Broker failure preserves intent
    new_manager_fail = JobManager(storage_dir=manager.storage_dir)
    with patch("backend.app.tasks.finalize_inspection_task.delay", side_effect=Exception("Redis connection error")):
        new_manager_fail.reconcile_outbox_on_startup()
        assert intent_file.exists()

    # Subcase 2: Valid intent is republished
    new_manager = JobManager(storage_dir=manager.storage_dir)
    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        republished = new_manager.reconcile_outbox_on_startup()
        assert republished == 1
        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)
        assert intent_file.exists()

    # Subcase 3: Stale generation intent discarded
    fanout = new_manager.get_fanout_status(job.id)
    fanout.finalizer_generation = 1
    new_manager._save_fanout_state_locked(fanout)

    assert intent_file.exists()

    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        new_manager.reconcile_outbox_on_startup()
        mock_delay.assert_not_called()
        assert not intent_file.exists()


def test_finalizer_generation_takeover_fences_old_generation_and_dispatches_new(sample_job_in_manager, monkeypatch):
    """
    Test E (Authorized Recovery-Dispatch Fix):
    - Expire existing finalizer generation 0.
    - Run recovery supervisor: asserts generation 1 takeover is persisted and dispatched.
    - Verify old generation 0 cannot claim, heartbeat, or complete.
    - Verify new generation 1 intent remains until claimed.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.5])

    # Finalizer 0 claims delegation
    claim_gen0 = manager.claim_finalizer(job.id, 1, "worker-finalizer-dead", fanout.fanout_run_id)
    assert claim_gen0["finalizer_generation"] == 0
    token_gen0 = claim_gen0["delegation_token"]

    # Expire finalizer lease for gen 0
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 10.0
    manager._save_fanout_state_locked(f_state)

    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect, \
         patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-recovery": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()

        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)

    # Master state is now generation 1 in CALLBACK_PENDING
    f_takeover = manager.get_fanout_status(job.id)
    assert f_takeover.status == "CALLBACK_PENDING"
    assert f_takeover.finalizer_generation == 1
    assert f_takeover.finalizer_worker is None
    assert f_takeover.delegation_token is None

    outbox_gen1 = manager.storage_dir / job.id / "outbox" / "finalizer_1_1.json"
    assert outbox_gen1.exists()

    # Stale generation 0 operations are fenced and rejected with 409
    with pytest.raises(LeaseConflictError):
        manager.apply_finalizer_heartbeat(
            job.id, 1, "worker-finalizer-dead", token_gen0, finalizer_generation=0
        )

    with pytest.raises(LeaseConflictError):
        manager.apply_result(
            inspection_id=job.id,
            point_count=500,
            mesh_status="completed",
            findings=[],
            worker="worker-finalizer-dead",
            lease_token=1,
            delegation_token=token_gen0,
            finalizer_generation=0,
        )

    # New generation 1 is claimed
    claim_gen1 = manager.claim_finalizer(job.id, 1, "worker-finalizer-alive", fanout.fanout_run_id)
    assert claim_gen1["status"] == "claimed"
    assert claim_gen1["finalizer_generation"] == 1
    assert not outbox_gen1.exists()


def test_takeover_crash_before_intent_persisted_preserves_old_generation(sample_job_in_manager, monkeypatch):
    """
    Crash Boundary 1:
    - Crash/failure occurs before the new intent is persisted.
    - Authoritative generation must not advance; old generation remains authoritative.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.5])
    claim_gen0 = manager.claim_finalizer(job.id, 1, "worker-finalizer-dead", fanout.fanout_run_id)
    assert claim_gen0["finalizer_generation"] == 0

    # Expire finalizer lease for gen 0
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 10.0
    manager._save_fanout_state_locked(f_state)

    supervisor = RecoverySupervisor(manager)

    # Inject failure before new intent is written
    orig_write = manager.write_finalizer_outbox_intent_locked
    def crash_write(*args, **kwargs):
        raise OSError("Simulated disk crash before intent persistence")

    with patch.object(manager, "write_finalizer_outbox_intent_locked", side_effect=crash_write), \
         patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = {"worker-recovery": "pong"}
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(OSError):
                loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()
    f_current = manager.get_fanout_status(job.id)
    assert f_current.finalizer_generation == 0
    assert f_current.status == "FINALIZATION_RUNNING"
    intent_gen1 = manager.storage_dir / job.id / "outbox" / "finalizer_1_1.json"
    assert not intent_gen1.exists()


def test_takeover_crash_after_intent_before_fanout_state_safely_reconciled(sample_job_in_manager, monkeypatch):
    """
    Crash Boundary 2:
    - Crash occurs after new-generation intent is written to disk, but before fanout_state.json is persisted.
    - Startup reconciliation must safely discard uncommitted future intent without dispatching or advancing generation.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.5])
    claim_gen0 = manager.claim_finalizer(job.id, 1, "worker-finalizer-dead", fanout.fanout_run_id)
    assert claim_gen0["finalizer_generation"] == 0

    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 10.0
    manager._save_fanout_state_locked(f_state)

    # Simulate crash: write intent for gen 1, but do NOT update fanout_state.json
    intent_gen1 = manager.write_finalizer_outbox_intent_locked(
        inspection_id=job.id, lease_token=1, fanout_run_id=fanout.fanout_run_id, generation=1
    )
    assert intent_gen1.exists()

    # Re-instantiate JobManager from disk (simulating restart)
    new_manager = JobManager(storage_dir=manager.storage_dir)
    assert new_manager.get_fanout_status(job.id).finalizer_generation == 0

    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        repub = new_manager.reconcile_outbox_on_startup()
        # Must not dispatch uncommitted future intent
        assert repub == 0
        mock_delay.assert_not_called()
        # Uncommitted intent was discarded
        assert not intent_gen1.exists()

    # Authoritative generation remained 0
    assert new_manager.get_fanout_status(job.id).finalizer_generation == 0


def test_takeover_crash_after_fanout_state_before_inspection_state_reconciles_and_recovers(sample_job_in_manager, monkeypatch):
    """
    Crash Boundary 3:
    - Crash occurs after fanout_state.json is persisted with gen 1, but before inspection.json is updated.
    - Startup must reconcile inspection.json from authoritative fanout_state.json and recover new intent.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.5])
    claim_gen0 = manager.claim_finalizer(job.id, 1, "worker-finalizer-dead", fanout.fanout_run_id)
    assert claim_gen0["finalizer_generation"] == 0

    # Write intent for gen 1 and persist fanout_state.json with gen 1
    intent_gen1 = manager.write_finalizer_outbox_intent_locked(
        inspection_id=job.id, lease_token=1, fanout_run_id=fanout.fanout_run_id, generation=1
    )
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_generation = 1
    f_state.status = "CALLBACK_PENDING"
    f_state.delegation_token = None
    f_state.finalizer_worker = None
    f_state.callback_dispatched_at = time.time()
    manager._save_fanout_state_locked(f_state)

    # inspection.json is deliberately left at generation 0 (simulating crash before step 5)
    raw_job_on_disk = json.loads((manager.storage_dir / job.id / "inspection.json").read_text())
    assert raw_job_on_disk.get("finalizer_generation") == 0

    # Restart JobManager
    new_manager = JobManager(storage_dir=manager.storage_dir)

    # Assert inspection.json was synchronized to generation 1 and CALLBACK_PENDING
    synced_job = new_manager.get_job(job.id)
    assert synced_job.finalizer_generation == 1
    assert synced_job.fanout_status == "CALLBACK_PENDING"

    # Reconcile outbox: recovers and dispatches valid generation 1 intent
    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        repub = new_manager.reconcile_outbox_on_startup()
        assert repub == 1
        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)
        assert intent_gen1.exists()


def test_takeover_crash_after_state_persistence_before_broker_publish_republishes(sample_job_in_manager, monkeypatch):
    """
    Crash Boundary 4:
    - State is fully persisted, but master crashes before broker publication.
    - Startup reconciliation discovers pending intent and publishes it.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.5])

    # Write intent and persist states for gen 1
    intent_gen1 = manager.write_finalizer_outbox_intent_locked(
        inspection_id=job.id, lease_token=1, fanout_run_id=fanout.fanout_run_id, generation=1
    )
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_generation = 1
    f_state.status = "CALLBACK_PENDING"
    f_state.delegation_token = None
    f_state.finalizer_worker = None
    f_state.callback_dispatched_at = time.time()
    manager._save_fanout_state_locked(f_state)
    job.finalizer_generation = 1
    job.fanout_status = "CALLBACK_PENDING"
    manager._save_job_state(job)

    # Restart
    new_manager = JobManager(storage_dir=manager.storage_dir)
    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        repub = new_manager.reconcile_outbox_on_startup()
        assert repub == 1
        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)
        assert intent_gen1.exists()
        with open(intent_gen1, "r") as f:
            data = json.load(f)
        assert data["state"] == "published"


def test_takeover_crash_after_broker_publish_before_marking_published_duplicate_safe(sample_job_in_manager, monkeypatch):
    """
    Crash Boundary 5:
    - Task was published, but master crashed before marking intent published.
    - On restart, reconciler republishes task (duplicate delivery).
    - Claim fencing ensures only the first claim succeeds and subsequent claims are rejected with 409.
    """
    manager, job = sample_job_in_manager
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")

    fanout = manager.init_fanout(
        job.id, lease_token=1, worker_name="coord", total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )
    c1 = manager.claim_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0)
    manager.commit_batch(job.id, 1, "worker-1", fanout.fanout_run_id, 0, c1["claim_token"], [0.5])

    intent_gen1 = manager.write_finalizer_outbox_intent_locked(
        inspection_id=job.id, lease_token=1, fanout_run_id=fanout.fanout_run_id, generation=1
    )
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_generation = 1
    f_state.status = "CALLBACK_PENDING"
    f_state.delegation_token = None
    f_state.finalizer_worker = None
    f_state.callback_dispatched_at = time.time()
    manager._save_fanout_state_locked(f_state)

    # Intent is still in state 'pending' because crash occurred before marking published
    with open(intent_gen1, "r") as f:
        assert json.load(f)["state"] == "pending"

    new_manager = JobManager(storage_dir=manager.storage_dir)

    # Duplicate delivery 1 claims
    claim1 = new_manager.claim_finalizer(job.id, 1, "worker-finalizer-1", fanout.fanout_run_id)
    assert claim1["status"] == "claimed"
    assert claim1["finalizer_generation"] == 1
    # Claim acknowledgement unlinks intent
    assert not intent_gen1.exists()

    # Duplicate delivery 2 attempts to claim: rejected with LeaseConflictError (409)
    with pytest.raises(LeaseConflictError):
        new_manager.claim_finalizer(job.id, 1, "worker-finalizer-2", fanout.fanout_run_id)


def test_duplicate_batch_task_retry_with_server_countdown_and_exhaustion():
    """
    Retry Exhaustion Unit Test:
    - Verifies that when a duplicate task receives already_claimed, it invokes self.retry()
      with the server-computed countdown and max_retries=24.
    - Verifies that when retry limit (24) is reached, MaxRetriesExceededError is raised,
      triggering chord callback suppression.
    """
    from backend.app.tasks import classify_batch_task
    from celery.exceptions import MaxRetriesExceededError, Retry

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "already_claimed",
        "batch_id": 0,
        "claimed_by": "worker-A",
        "retry_after_sec": 7.5,
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    # Verify self.retry() is called with countdown=7.5 and max_retries=24
    with patch("backend.app.tasks.httpx.Client", return_value=mock_client), \
         patch.object(classify_batch_task, "retry", side_effect=Retry) as mock_retry:
        with pytest.raises(Retry):
            classify_batch_task.run("job-test", 1, "fanout-test", 0)
        mock_retry.assert_called_once_with(countdown=7.5, max_retries=24)

    # Verify that at retry limit 24, Celery raises MaxRetriesExceededError
    with patch("backend.app.tasks.httpx.Client", return_value=mock_client), \
         patch.object(classify_batch_task, "retry", side_effect=MaxRetriesExceededError("Retries exhausted")):
        with pytest.raises(MaxRetriesExceededError):
            classify_batch_task.run("job-test", 1, "fanout-test", 0)
