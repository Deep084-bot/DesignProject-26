"""
Tests for M4b Finalizer Heartbeat Reliability Correction.
Verifies that FinalizerHeartbeatThread immediately pulses on startup, periodic heartbeats
renew delegation expiration windows, generation fencing is strictly enforced, and
authoritative takeovers preserve durable write-ahead outbox invariants.
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.models.calibration import RectifiedStereoCalibration
from backend.app.services.job_manager import JobManager, LeaseConflictError
from backend.app.services.recovery_supervisor import RecoverySupervisor
from backend.app.tasks import FinalizerHeartbeatThread
from backend.app import settings


@pytest.fixture
def clean_job_with_fanout(tmp_path: Path, monkeypatch):
    """Provides a fresh JobManager instance with a sample job advanced to ALL_BATCHES_COMMITTED."""
    monkeypatch.setenv("SURFACE_WORKER_TOKEN", "test-secret-token")
    monkeypatch.setenv("SURFACE_EXECUTION", "celery")
    monkeypatch.setenv("SURFACE_LEASE_TIMEOUT", "35.0")
    monkeypatch.setenv("SURFACE_OFFLINE_GRACE_PERIOD", "15.0")

    manager = JobManager(storage_dir=tmp_path / "jobs")
    calib = RectifiedStereoCalibration(
        camera_model="pinhole_stereo",
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=100,
        image_height=100,
        principal_point=(50.0, 50.0),
    )
    job = manager.create_job(
        inspection_id="INSP-FINALIZER-TEST",
        name="Finalizer Test",
        surface_type="Concrete",
        reconstruction_method="Automatic",
        left_bytes=b"dummy-left",
        left_filename="left.png",
        right_bytes=b"dummy-right",
        right_filename="right.png",
        calibration=calib,
        calibration_summary=None,
        raw_calibration_bytes=b"{}",
        dispatch_mode="none",
    )
    # Coordinator claims manifest
    manager.claim_manifest(job.id, lease_token=1, worker_name="coord-worker")

    # Initialize fanout with 1 batch
    fanout = manager.init_fanout(
        inspection_id=job.id,
        lease_token=1,
        worker_name="coord-worker",
        total_batches=1,
        batches=[{"batch_id": 0, "tile_indices": [0]}],
    )

    # Claim and commit batch 0
    claim = manager.claim_batch(job.id, lease_token=1, worker_name="batch-worker", fanout_run_id=fanout.fanout_run_id, batch_id=0)
    manager.commit_batch(
        job.id, lease_token=1, worker_name="batch-worker",
        fanout_run_id=fanout.fanout_run_id, batch_id=0,
        claim_token=claim["claim_token"], probabilities=[0.95],
    )

    # Reload fanout status
    f_state = manager.get_fanout_status(job.id)
    assert f_state.status in ("ALL_BATCHES_COMMITTED", "CALLBACK_PENDING")
    return manager, job, fanout


def _run_supervisor_cycle(supervisor: RecoverySupervisor, online_pings: dict):
    """Runs a single recovery supervisor cycle with mocked Celery ping response."""
    with patch("backend.app.celery_app.celery_app.control.inspect") as mock_inspect:
        mock_insp_inst = MagicMock()
        mock_insp_inst.ping.return_value = online_pings
        mock_inspect.return_value = mock_insp_inst

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(supervisor.check_recovery_cycle())
        finally:
            loop.close()


# ============================================================================
# 1. Immediate Initial Finalizer Heartbeat
# ============================================================================

def test_finalizer_heartbeat_thread_sends_immediate_initial_pulse():
    """
    FinalizerHeartbeatThread must pulse immediately on start without sleeping
    for the full heartbeat interval.
    """
    abort_evt = threading.Event()
    stop_evt = threading.Event()
    pulses = []

    class MockFinalizerThread(FinalizerHeartbeatThread):
        def _pulse(self) -> bool:
            pulses.append(time.monotonic())
            return True

    # Use large interval (100.0s) so periodic pulses would be far in the future
    t = MockFinalizerThread(
        inspection_id="INSP-IMM-PULSE",
        lease_token=1,
        worker_name="worker-mac-2",
        master_url="http://mock-master",
        worker_token="mock-token",
        delegation_token="del-mock",
        finalizer_generation=0,
        abort_event=abort_evt,
        stop_event=stop_evt,
        interval=100.0,
    )

    t0 = time.monotonic()
    t.start()
    time.sleep(0.05)
    stop_evt.set()
    t.join(timeout=1.0)

    assert len(pulses) >= 1, "FinalizerHeartbeatThread failed to pulse immediately on start"
    assert (pulses[0] - t0) < 0.5, "Initial pulse was delayed rather than immediate"


# ============================================================================
# 2. Periodic Finalizer Heartbeats
# ============================================================================

def test_finalizer_heartbeat_thread_periodic_pulses():
    """FinalizerHeartbeatThread sends multiple pulses at configured periodic intervals."""
    abort_evt = threading.Event()
    stop_evt = threading.Event()
    pulses = []

    class MockFinalizerThread(FinalizerHeartbeatThread):
        def _pulse(self) -> bool:
            pulses.append(time.monotonic())
            return True

    # Use fast interval (0.05s)
    t = MockFinalizerThread(
        inspection_id="INSP-PERIODIC-PULSE",
        lease_token=1,
        worker_name="worker-mac-2",
        master_url="http://mock-master",
        worker_token="mock-token",
        delegation_token="del-mock",
        finalizer_generation=0,
        abort_event=abort_evt,
        stop_event=stop_evt,
        interval=0.05,
    )

    t.start()
    time.sleep(0.18)
    stop_evt.set()
    t.join(timeout=1.0)

    assert len(pulses) >= 3, f"Expected >= 3 pulses, got {len(pulses)}"


# ============================================================================
# 3. Fresh Heartbeat Retains Generation 0
# ============================================================================

def test_fresh_finalizer_heartbeat_retains_generation_0(clean_job_with_fanout):
    """
    When finalizer delegation heartbeat is fresh, supervisor retains generation 0
    without triggering takeover or writing recovery intents.
    """
    manager, job, fanout = clean_job_with_fanout
    claim = manager.claim_finalizer(
        job.id, lease_token=1, worker_name="worker-mac-2", fanout_run_id=fanout.fanout_run_id, delegation_duration_sec=30.0
    )
    assert claim["finalizer_generation"] == 0

    # Pulse heartbeat
    hb_res = manager.apply_finalizer_heartbeat(
        inspection_id=job.id,
        lease_token=1,
        worker_name="worker-mac-2",
        delegation_token=claim["delegation_token"],
        finalizer_generation=0,
        extension_sec=30.0,
    )
    assert hb_res["status"] == "ok"

    supervisor = RecoverySupervisor(manager)
    _run_supervisor_cycle(supervisor, {"worker-mac-1@host": "pong", "worker-mac-2@host": "pong"})

    f_state = manager.get_fanout_status(job.id)
    assert f_state.status == "FINALIZATION_RUNNING"
    assert f_state.finalizer_generation == 0
    assert f_state.finalizer_worker == "worker-mac-2"

    # Verify zero outbox takeover files
    outbox_dir = manager.storage_dir / job.id / "outbox"
    if outbox_dir.exists():
        assert len(list(outbox_dir.glob("finalizer_*.json"))) == 0


# ============================================================================
# 4. Heartbeat Updates the Correct Finalizer Generation
# ============================================================================

def test_finalizer_heartbeat_updates_correct_generation(clean_job_with_fanout):
    """Heartbeat extends finalizer_expires_at specifically for the active generation."""
    manager, job, fanout = clean_job_with_fanout

    # Advance to generation 1
    manager.claim_finalizer(job.id, 1, "worker-dead", fanout.fanout_run_id, delegation_duration_sec=0.001)
    time.sleep(0.01)

    claim_gen1 = manager.claim_finalizer(job.id, 1, "worker-mac-1", fanout.fanout_run_id, delegation_duration_sec=15.0)
    assert claim_gen1["finalizer_generation"] == 1
    initial_exp = claim_gen1["finalizer_expires_at"]

    hb_res = manager.apply_finalizer_heartbeat(
        inspection_id=job.id,
        lease_token=1,
        worker_name="worker-mac-1",
        delegation_token=claim_gen1["delegation_token"],
        finalizer_generation=1,
        extension_sec=60.0,
    )
    assert hb_res["status"] == "ok"
    assert hb_res["finalizer_expires_at"] > initial_exp

    f_state = manager.get_fanout_status(job.id)
    assert f_state.finalizer_generation == 1
    assert f_state.finalizer_expires_at == hb_res["finalizer_expires_at"]


# ============================================================================
# 5. Stale Generation Cannot Refresh a Newer Generation
# ============================================================================

def test_stale_generation_cannot_refresh_newer_generation(clean_job_with_fanout):
    """A heartbeat from an older generation is rejected with LeaseConflictError."""
    manager, job, fanout = clean_job_with_fanout

    claim_gen0 = manager.claim_finalizer(job.id, 1, "worker-mac-2", fanout.fanout_run_id, delegation_duration_sec=0.001)
    time.sleep(0.01)

    claim_gen1 = manager.claim_finalizer(job.id, 1, "worker-mac-1", fanout.fanout_run_id, delegation_duration_sec=30.0)
    assert claim_gen1["finalizer_generation"] == 1

    # Worker with stale gen 0 attempts heartbeat
    with pytest.raises(LeaseConflictError) as exc_info:
        manager.apply_finalizer_heartbeat(
            inspection_id=job.id,
            lease_token=1,
            worker_name="worker-mac-2",
            delegation_token=claim_gen0["delegation_token"],
            finalizer_generation=0,
            extension_sec=30.0,
        )
    assert "Finalizer delegation token mismatch or revoked" in str(exc_info.value) or "generation mismatch" in str(exc_info.value)


# ============================================================================
# 6. Heartbeat HTTP Failure Is Handled Correctly
# ============================================================================

def test_finalizer_heartbeat_http_failure_handled_correctly():
    """Network or transient HTTP failures do not set abort_event; 409 Conflict sets abort_event."""
    abort_evt = threading.Event()
    stop_evt = threading.Event()

    # Case A: Network connection failure
    t = FinalizerHeartbeatThread(
        inspection_id="INSP-NET-ERR",
        lease_token=1,
        worker_name="worker-mac-2",
        master_url="http://invalid-nonexistent-domain-xyz.local",
        worker_token="mock-token",
        delegation_token="del-mock",
        finalizer_generation=0,
        abort_event=abort_evt,
        stop_event=stop_evt,
        interval=100.0,
    )
    t.start()
    time.sleep(0.05)
    stop_evt.set()
    t.join(timeout=1.0)
    assert not abort_evt.is_set(), "Network error should not abort finalizer"

    # Case B: 409 Conflict response
    abort_evt2 = threading.Event()
    stop_evt2 = threading.Event()
    t2 = FinalizerHeartbeatThread(
        inspection_id="INSP-409-CONFLICT",
        lease_token=1,
        worker_name="worker-mac-2",
        master_url="http://mock-master",
        worker_token="mock-token",
        delegation_token="del-mock",
        finalizer_generation=0,
        abort_event=abort_evt2,
        stop_event=stop_evt2,
        interval=100.0,
    )

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 409
        mock_resp.text = "Delegation revoked"
        mock_post.return_value = mock_resp

        t2.start()
        time.sleep(0.05)
        t2.join(timeout=1.0)
        assert abort_evt2.is_set(), "409 Conflict must trigger abort_event"


# ============================================================================
# 7. Thread Stops and Joins Cleanly
# ============================================================================

def test_finalizer_heartbeat_thread_stops_and_joins_cleanly():
    """Setting stop_event terminates the finalizer heartbeat thread promptly."""
    abort_evt = threading.Event()
    stop_evt = threading.Event()

    class MockThread(FinalizerHeartbeatThread):
        def _pulse(self) -> bool:
            return True

    t = MockThread(
        inspection_id="INSP-CLEAN-STOP",
        lease_token=1,
        worker_name="worker-mac-2",
        master_url="http://mock-master",
        worker_token="mock-token",
        delegation_token="del-mock",
        finalizer_generation=0,
        abort_event=abort_evt,
        stop_event=stop_evt,
        interval=60.0,
    )
    t.start()
    assert t.is_alive()
    stop_evt.set()
    t.join(timeout=1.0)
    assert not t.is_alive(), "Thread did not stop after stop_event set"


# ============================================================================
# 8. Finalizer Completing Before Expiry Does Not Trigger Takeover
# ============================================================================

def test_finalizer_completing_before_delegation_expiry_no_takeover(clean_job_with_fanout):
    """When a finalizer completes inspection before delegation window expires, no takeover occurs."""
    manager, job, fanout = clean_job_with_fanout
    claim = manager.claim_finalizer(job.id, 1, "worker-mac-2", fanout.fanout_run_id, delegation_duration_sec=30.0)

    # Complete job successfully
    completed = manager.apply_result(
        inspection_id=job.id,
        point_count=100,
        mesh_status="completed",
        duration_sec=1.5,
        worker="worker-mac-2",
        lease_token=1,
        delegation_token=claim["delegation_token"],
        finalizer_generation=0,
    )
    assert completed is True

    supervisor = RecoverySupervisor(manager)
    _run_supervisor_cycle(supervisor, {})

    j = manager.get_job(job.id)
    assert j.status == "completed"
    assert j.finalizer_generation == 0
    assert j.finalizer_worker == "worker-mac-2"


# ============================================================================
# 9. Genuine Delegation Expiry Permits One Authoritative Takeover
# ============================================================================

def test_genuine_delegation_expiry_permits_one_authoritative_takeover(clean_job_with_fanout):
    """When delegation window actually expires, supervisor performs exactly one WAL takeover to gen 1."""
    manager, job, fanout = clean_job_with_fanout
    claim = manager.claim_finalizer(job.id, 1, "worker-mac-2", fanout.fanout_run_id, delegation_duration_sec=0.001)

    # Expire delegation
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 5.0
    manager._save_fanout_state_locked(f_state)

    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        _run_supervisor_cycle(supervisor, {"worker-mac-1@host": "pong", "worker-mac-2@host": "pong"})
        mock_delay.assert_called_once_with(job.id, 1, fanout.fanout_run_id)

    f_state2 = manager.get_fanout_status(job.id)
    assert f_state2.finalizer_generation == 1
    assert f_state2.status == "CALLBACK_PENDING"
    assert f_state2.delegation_token is None
    assert f_state2.finalizer_worker is None


# ============================================================================
# 10. Duplicate Supervisor Cycles Do Not Create Duplicate Recovery Dispatches
# ============================================================================

def test_duplicate_supervisor_cycles_do_not_duplicate_finalizer_dispatches(clean_job_with_fanout):
    """Once finalizer is recovered to CALLBACK_PENDING, subsequent cycles do not double-dispatch."""
    manager, job, fanout = clean_job_with_fanout
    manager.claim_finalizer(job.id, 1, "worker-mac-2", fanout.fanout_run_id, delegation_duration_sec=0.001)

    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 5.0
    manager._save_fanout_state_locked(f_state)

    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.tasks.finalize_inspection_task.delay") as mock_delay:
        # Cycle 1: Triggers takeover
        _run_supervisor_cycle(supervisor, {})
        assert mock_delay.call_count == 1

        # Cycle 2: Within 30s callback arrival deadline -> NO redispatch
        _run_supervisor_cycle(supervisor, {})
        assert mock_delay.call_count == 1


# ============================================================================
# 11. Existing Write-Ahead Outbox Ordering Remains Intact
# ============================================================================

def test_existing_write_ahead_finalizer_outbox_ordering_intact(clean_job_with_fanout):
    """
    Finalizer takeover writes intent file before state update.
    Claiming new generation unlinks write-ahead intent file.
    """
    manager, job, fanout = clean_job_with_fanout
    manager.claim_finalizer(job.id, 1, "worker-mac-2", fanout.fanout_run_id, delegation_duration_sec=0.001)

    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 5.0
    manager._save_fanout_state_locked(f_state)

    supervisor = RecoverySupervisor(manager)
    with patch("backend.app.tasks.finalize_inspection_task.delay"):
        _run_supervisor_cycle(supervisor, {})

    intent_file = manager.storage_dir / job.id / "outbox" / "finalizer_1_1.json"
    assert intent_file.exists(), "Write-ahead finalizer intent must exist"
    with open(intent_file, "r") as f:
        data = json.load(f)
    assert data["generation"] == 1
    assert data["state"] == "published"

    # New worker claims generation 1 -> unlinks intent
    claim_gen1 = manager.claim_finalizer(job.id, 1, "worker-mac-1", fanout.fanout_run_id)
    assert claim_gen1["finalizer_generation"] == 1
    assert not intent_file.exists(), "Claiming finalizer must clean up outbox intent"


# ============================================================================
# 12. Old Finalizer Generations Remain Fenced After Takeover
# ============================================================================

def test_old_finalizer_generations_remain_fenced_after_takeover(clean_job_with_fanout):
    """After generation 0 is superseded by generation 1, any update with gen 0 credentials is 409 rejected."""
    manager, job, fanout = clean_job_with_fanout
    claim_gen0 = manager.claim_finalizer(job.id, 1, "worker-mac-2", fanout.fanout_run_id, delegation_duration_sec=0.001)

    # Expire and takeover to gen 1
    f_state = manager.get_fanout_status(job.id)
    f_state.finalizer_expires_at = time.time() - 5.0
    manager._save_fanout_state_locked(f_state)

    claim_gen1 = manager.claim_finalizer(job.id, 1, "worker-mac-1", fanout.fanout_run_id)
    assert claim_gen1["finalizer_generation"] == 1

    # Gen 0 tries to apply step update
    with pytest.raises(LeaseConflictError) as exc_step:
        manager.apply_step_update(
            inspection_id=job.id,
            index=4,
            step_id="damage",
            status="completed",
            worker="worker-mac-2",
            lease_token=1,
            delegation_token=claim_gen0["delegation_token"],
            finalizer_generation=0,
        )
    assert "revoked" in str(exc_step.value) or "mismatch" in str(exc_step.value)

    # Gen 0 tries to complete job
    with pytest.raises(LeaseConflictError) as exc_comp:
        manager.apply_result(
            inspection_id=job.id,
            point_count=100,
            mesh_status="completed",
            duration_sec=2.0,
            worker="worker-mac-2",
            lease_token=1,
            delegation_token=claim_gen0["delegation_token"],
            finalizer_generation=0,
        )
    assert "revoked" in str(exc_comp.value) or "mismatch" in str(exc_comp.value)
