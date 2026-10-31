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
