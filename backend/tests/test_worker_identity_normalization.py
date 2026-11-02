"""
Regression and forensic verification tests for worker identity normalization in SURFACE.

Verifies:
1. Canonical identity normalization ('worker-mac-1@hostname' -> 'worker-mac-1').
2. Identity equivalence (short name matches full Celery @hostname node identity).
3. Distinguishability (different short names remain distinct).
4. RecoverySupervisor decision table with Celery @hostname ping keys:
   - Healthy assigned worker is NOT classified as offline.
   - Healthy worker with short identity in pings is NOT classified as offline.
   - Genuinely missing worker is recognized as offline and enters recovery path.
5. Fencing leases and CAS reassignment remain strictly enforced across normalized identities.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch
import pytest

from backend.app import settings
from backend.app.services.job_manager import JobManager, LeaseConflictError
from backend.app.services.recovery_supervisor import RecoverySupervisor
from backend.app.models.inspection import InspectionResponse


# ============================================================================
# 1. Unit Tests: Normalization Function Properties
# ============================================================================

def test_normalize_worker_identity_strips_hostname():
    assert settings.normalize_worker_identity("worker-mac-1@Deeps-MacBook-Air.local") == "worker-mac-1"
    assert settings.normalize_worker_identity("worker-mac-2@ip-192-168-1-50") == "worker-mac-2"
    assert settings.normalize_worker_identity("worker-1@node.cluster.local") == "worker-1"


def test_normalize_worker_identity_preserves_short_name():
    assert settings.normalize_worker_identity("worker-mac-1") == "worker-mac-1"
    assert settings.normalize_worker_identity("worker-mac-2") == "worker-mac-2"
    assert settings.normalize_worker_identity("master-coordinator") == "master-coordinator"


def test_normalize_worker_identity_distinguishes_different_short_names():
    w1 = settings.normalize_worker_identity("worker-mac-1@Deeps-MacBook-Air.local")
    w2 = settings.normalize_worker_identity("worker-mac-2@Deeps-MacBook-Air.local")
    assert w1 != w2
    assert w1 == "worker-mac-1"
    assert w2 == "worker-mac-2"


def test_normalize_worker_identity_handles_none_and_empty():
    assert settings.normalize_worker_identity(None) is None
    assert settings.normalize_worker_identity("") is None
    assert settings.normalize_worker_identity("   ") is None


# ============================================================================
# 2. Supervisor-Level Tests: Actual Worker-Online Decision
# ============================================================================

@pytest.fixture
def clean_job_manager(tmp_path):
    mgr = JobManager(storage_dir=tmp_path / "storage")
    from backend.app.models.calibration import RectifiedStereoCalibration
    calib = RectifiedStereoCalibration(
        camera_model="pinhole_stereo",
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=320,
        image_height=240,
        principal_point=(160.0, 120.0),
    )
    job = mgr.create_job(
        inspection_id="INSP-2026-NORM01",
        name="Worker Normalization Test",
        surface_type="Concrete",
        reconstruction_method="Automatic",
        left_bytes=b"\x89PNG\r\n\x1a\nFakeLeft",
        left_filename="input_left.png",
        right_bytes=b"\x89PNG\r\n\x1a\nFakeRight",
        right_filename="input_right.png",
        calibration=calib,
        calibration_summary=None,
        raw_calibration_bytes=b"{}",
        dispatch_mode="none",
    )
    return mgr, job


def test_supervisor_does_not_classify_healthy_worker_with_hostname_as_offline(clean_job_manager, monkeypatch):
    """
    Forensic root-cause test for INSP-2026-D22E8E:
    When Celery inspect ping returns 'worker-mac-1@Deeps-MacBook-Air.local',
    and job.assigned_worker is 'worker-mac-1', the supervisor MUST recognize
    the worker as ONLINE and retain the healthy assignment without entering grace.
    """
    async def _run():
        manager, job = clean_job_manager
        supervisor = RecoverySupervisor(manager)
        monkeypatch.setenv("SURFACE_EXECUTION", "celery")
        monkeypatch.setenv("SURFACE_LEASE_TIMEOUT", "35.0")
        monkeypatch.setenv("SURFACE_OFFLINE_GRACE_PERIOD", "15.0")

        # Worker claims job with short identity
        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")
        assert manager.get_job(job.id).assigned_worker == "worker-mac-1"

        # Celery inspect ping returns worker with host suffix
        mock_insp = MagicMock()
        mock_insp.ping.return_value = {
            "worker-mac-1@Deeps-MacBook-Air.local": {"ok": "pong"},
            "worker-mac-2@Deeps-MacBook-Air.local": {"ok": "pong"},
        }

        with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
            # Run supervisor cycle
            await supervisor.check_recovery_cycle()

        # Invariant: Job assignment is retained and lease token is NOT bumped
        job_state = manager.get_job(job.id)
        assert job_state.lease_token == 1
        assert job_state.assigned_worker == "worker-mac-1"
        # No spurious grace period initiated
        assert job.id not in supervisor._grace_periods

    asyncio.run(_run())


def test_supervisor_retains_healthy_worker_with_exact_short_name(clean_job_manager, monkeypatch):
    """Supervisor retains assignment when Celery ping returns exact short name."""
    async def _run():
        manager, job = clean_job_manager
        supervisor = RecoverySupervisor(manager)
        monkeypatch.setenv("SURFACE_EXECUTION", "celery")

        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker-mac-1": {"ok": "pong"}}

        with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
            await supervisor.check_recovery_cycle()

        assert manager.get_job(job.id).lease_token == 1
        assert manager.get_job(job.id).assigned_worker == "worker-mac-1"
        assert job.id not in supervisor._grace_periods

    asyncio.run(_run())


def test_supervisor_distinguishes_assigned_worker_from_other_online_workers(clean_job_manager, monkeypatch):
    """
    When worker-mac-1 is assigned to a job, but only worker-mac-2 pings,
    supervisor recognizes that worker-mac-1 is genuinely offline.
    """
    async def _run():
        manager, job = clean_job_manager
        supervisor = RecoverySupervisor(manager)
        monkeypatch.setenv("SURFACE_EXECUTION", "celery")
        monkeypatch.setenv("SURFACE_LEASE_TIMEOUT", "35.0")
        monkeypatch.setenv("SURFACE_OFFLINE_GRACE_PERIOD", "15.0")

        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

        # Only worker-mac-2 is online
        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker-mac-2@Deeps-MacBook-Air.local": {"ok": "pong"}}

        with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp):
            await supervisor.check_recovery_cycle()

        # Worker-mac-1 is offline with fresh heartbeat -> enters grace period
        assert job.id in supervisor._grace_periods
        assert manager.get_job(job.id).lease_token == 1

    asyncio.run(_run())


def test_supervisor_genuinely_missing_worker_triggers_recovery(clean_job_manager, monkeypatch):
    """
    When the assigned worker is genuinely missing from Celery pings AND heartbeat is expired,
    the supervisor triggers immediate CAS lease recovery from generation 1 to 2.
    """
    async def _run():
        manager, job = clean_job_manager
        supervisor = RecoverySupervisor(manager)
        monkeypatch.setenv("SURFACE_EXECUTION", "celery")
        monkeypatch.setenv("SURFACE_LEASE_TIMEOUT", "10.0")

        manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")
        # Expire heartbeat
        manager._monotonic_heartbeats[job.id] = time.monotonic() - 25.0

        # Only worker-mac-2 is online
        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker-mac-2@Deeps-MacBook-Air.local": {"ok": "pong"}}

        with patch("backend.app.celery_app.celery_app.control.inspect", return_value=mock_insp), \
             patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
            await supervisor.check_recovery_cycle()

            # Lease token incremented to 2 and recovery task dispatched
            updated = manager.get_job(job.id)
            assert updated.lease_token == 2
            assert updated.assigned_worker is None
            assert mock_delay.called

    asyncio.run(_run())


# ============================================================================
# 3. Fencing and Reassignment Invariants with Normalized Identities
# ============================================================================

def test_fencing_allows_normalized_worker_identity_variants(clean_job_manager):
    """
    A worker claiming with 'worker-mac-1' can send heartbeats or updates
    using either 'worker-mac-1' or 'worker-mac-1@host', but conflicting workers are rejected.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    # Heartbeat with exact name succeeds
    assert manager.apply_heartbeat(job.id, lease_token=1, worker_name="worker-mac-1") is True

    # Heartbeat with full node identity succeeds
    assert manager.apply_heartbeat(job.id, lease_token=1, worker_name="worker-mac-1@Deeps-MacBook-Air.local") is True

    # Conflicting worker is rejected with LeaseConflictError
    with pytest.raises(LeaseConflictError) as exc_info:
        manager.apply_heartbeat(job.id, lease_token=1, worker_name="worker-mac-2")
    assert "Worker identity mismatch" in str(exc_info.value)

    # Conflicting worker with full node identity is also rejected
    with pytest.raises(LeaseConflictError) as exc_info:
        manager.apply_heartbeat(job.id, lease_token=1, worker_name="worker-mac-2@Deeps-MacBook-Air.local")
    assert "Worker identity mismatch" in str(exc_info.value)


def test_batch_claim_heartbeat_enforces_normalized_worker_identity(clean_job_manager):
    """Batch claim heartbeat accepts canonical worker variants and rejects rogue workers."""
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    # Initialize fanout
    fanout = manager.init_fanout(
        inspection_id=job.id,
        lease_token=1,
        worker_name="worker-mac-1",
        total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0, 1, 2]}],
    )

    # Worker-mac-2 claims batch 0
    claim_res = manager.claim_batch(
        inspection_id=job.id,
        lease_token=1,
        fanout_run_id=fanout.fanout_run_id,
        batch_id=0,
        worker_name="worker-mac-2@node",
    )
    claim_token = claim_res["claim_token"]

    # Heartbeat from worker-mac-2 succeeds
    ext_res = manager.apply_batch_heartbeat(
        inspection_id=job.id,
        lease_token=1,
        fanout_run_id=fanout.fanout_run_id,
        batch_id=0,
        claim_token=claim_token,
        worker_name="worker-mac-2",
        extension_sec=20.0,
    )
    assert ext_res["status"] == "ok"

    # Heartbeat from rogue worker-mac-1 is rejected
    with pytest.raises(LeaseConflictError) as exc_info:
        manager.apply_batch_heartbeat(
            inspection_id=job.id,
            lease_token=1,
            fanout_run_id=fanout.fanout_run_id,
            batch_id=0,
            claim_token=claim_token,
            worker_name="worker-mac-1",
            extension_sec=20.0,
        )
    assert "Batch 0 is claimed by 'worker-mac-2@node', not 'worker-mac-1'" in str(exc_info.value)
