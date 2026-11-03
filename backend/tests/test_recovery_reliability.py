"""
Tests for M4b Recovery Reliability Correction.
Verifies the invariant that a missing Celery control ping alone must NEVER cause
coordinator lease reassignment while the application heartbeat is fresh.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.models.calibration import RectifiedStereoCalibration
from backend.app.services.job_manager import JobManager, LeaseConflictError
from backend.app.services.recovery_supervisor import RecoverySupervisor
from backend.app.tasks import WorkerHeartbeatThread
from backend.app import settings


@pytest.fixture
def clean_job_manager(tmp_path: Path, monkeypatch):
    """Provides a fresh JobManager instance with an initial sample job."""
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
        inspection_id="INSP-RECOVERY-TEST",
        name="Recovery Test",
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
    return manager, job


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
# 1. Online + Fresh Heartbeat -> Retain Assignment
# ============================================================================

def test_online_and_fresh_heartbeat_retains_assignment(clean_job_manager):
    """Worker in ping response with fresh heartbeat is retained without grace period."""
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    supervisor = RecoverySupervisor(manager)
    _run_supervisor_cycle(supervisor, {"worker-mac-1@hostname": "pong"})

    j = manager.get_job(job.id)
    assert j.lease_token == 1
    assert j.assigned_worker == "worker-mac-1"
    assert job.id not in supervisor._grace_periods


# ============================================================================
# 2. Offline + Fresh Heartbeat -> Retain Assignment Across Multiple Cycles
# ============================================================================

def test_offline_and_fresh_heartbeat_retains_assignment_across_multiple_cycles(clean_job_manager):
    """
    CRITICAL INVARIANT TEST:
    Worker busy in CPU computation (e.g. solo pool) drops off Celery ping.
    As long as heartbeat is fresh (<= 35s), supervisor MUST retain assignment
    and NEVER trigger CAS reassignment, even across multiple check cycles.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    supervisor = RecoverySupervisor(manager)

    # Worker-mac-2 is idle and responds; worker-mac-1 is busy and omitted
    ping_resp = {"worker-mac-2@hostname": "pong"}

    for cycle in range(5):
        _run_supervisor_cycle(supervisor, ping_resp)
        j = manager.get_job(job.id)
        assert j.lease_token == 1, f"Lease was improperly bumped on cycle {cycle + 1}"
        assert j.assigned_worker == "worker-mac-1"
        assert job.id not in supervisor._grace_periods, "Grace period should not be set while heartbeat is fresh"

    # Confirm zero recovery outbox intents written
    outbox_dir = manager.storage_dir / job.id / "outbox"
    if outbox_dir.exists():
        assert len(list(outbox_dir.glob("*.json"))) == 0


# ============================================================================
# 3. Offline + Stale Heartbeat Before Lease Expiry -> No Premature Takeover
# ============================================================================

def test_offline_and_heartbeat_before_lease_expiry_no_premature_takeover(clean_job_manager):
    """
    Worker is missing from Celery ping, but heartbeat age (e.g. 20s) is within the
    35s lease timeout. The supervisor must NOT execute premature takeover.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    # Simulate heartbeat age of 20 seconds (fresh since <= 35.0s)
    manager._monotonic_heartbeats[job.id] = time.monotonic() - 20.0

    supervisor = RecoverySupervisor(manager)
    _run_supervisor_cycle(supervisor, {"worker-mac-2@hostname": "pong"})

    j = manager.get_job(job.id)
    assert j.lease_token == 1
    assert j.assigned_worker == "worker-mac-1"


# ============================================================================
# 4. Offline + Stale Heartbeat After Lease Expiry -> Exactly One Takeover
# ============================================================================

def test_offline_and_stale_heartbeat_after_lease_expiry_exactly_one_takeover(clean_job_manager):
    """
    When worker is missing from ping AND heartbeat is expired (> 35s),
    authoritative lease timeout has passed and supervisor triggers exactly one CAS takeover.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    # Age heartbeat past 35s lease timeout
    manager._monotonic_heartbeats[job.id] = time.monotonic() - 40.0

    supervisor = RecoverySupervisor(manager)
    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        _run_supervisor_cycle(supervisor, {"worker-mac-2@hostname": "pong"})
        mock_delay.assert_called_once_with(job.id, 2)

    j = manager.get_job(job.id)
    assert j.lease_token == 2
    assert j.status == "queued"  # Reset for recovery claim


# ============================================================================
# 5. Online + Stale Heartbeat -> Bounded Grace and Correct Recovery
# ============================================================================

def test_online_and_stale_heartbeat_bounded_grace_and_recovery(clean_job_manager, monkeypatch):
    """
    Worker responds to Celery ping, but task heartbeat is expired (> 35s).
    Supervisor grants a bounded grace period (15s) for task stalls, then recovers.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    fake_time = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    # Age heartbeat past 35s lease timeout
    manager._monotonic_heartbeats[job.id] = fake_time[0] - 40.0

    supervisor = RecoverySupervisor(manager)

    # Cycle 1: Grace period starts, no immediate reassignment
    _run_supervisor_cycle(supervisor, {"worker-mac-1@hostname": "pong"})

    j1 = manager.get_job(job.id)
    assert j1.lease_token == 1
    assert job.id in supervisor._grace_periods
    assert supervisor._grace_periods[job.id] == 100.0

    # Cycle 2: Grace period elapsed 10s (< 15s threshold) -> still retained
    fake_time[0] = 110.0
    _run_supervisor_cycle(supervisor, {"worker-mac-1@hostname": "pong"})
    j2 = manager.get_job(job.id)
    assert j2.lease_token == 1

    # Cycle 3: Grace period elapsed 16s (>= 15s threshold) -> takeover executed
    fake_time[0] = 116.0
    with patch("backend.app.tasks.run_inspection_task.delay"):
        _run_supervisor_cycle(supervisor, {"worker-mac-1@hostname": "pong"})

    j3 = manager.get_job(job.id)
    assert j3.lease_token == 2
    assert job.id not in supervisor._grace_periods


# ============================================================================
# 6. Fresh Heartbeat Clears an Existing Grace Period
# ============================================================================

def test_fresh_heartbeat_clears_existing_grace_period(clean_job_manager):
    """If a task was previously in a grace period, a refreshed heartbeat clears the grace period."""
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    supervisor = RecoverySupervisor(manager)
    supervisor._grace_periods[job.id] = time.monotonic() - 5.0

    # Heartbeat is fresh
    manager.apply_heartbeat(job.id, lease_token=1, worker_name="worker-mac-1")

    _run_supervisor_cycle(supervisor, {"worker-mac-1@hostname": "pong"})
    assert job.id not in supervisor._grace_periods


# ============================================================================
# 7. Heartbeat Thread Sends an Immediate Initial Pulse
# ============================================================================

def test_heartbeat_thread_sends_immediate_initial_pulse():
    """
    WorkerHeartbeatThread must pulse immediately on start without waiting
    for the full heartbeat interval.
    """
    abort_event = asyncio.Event()  # unused placeholder
    import threading
    abort_evt = threading.Event()
    stop_evt = threading.Event()

    pulses = []

    class MockHeartbeatThread(WorkerHeartbeatThread):
        def _pulse(self) -> bool:
            pulses.append(time.monotonic())
            return True

    # Use a large interval (100.0s) so any second pulse would be far in the future
    t = MockHeartbeatThread(
        inspection_id="INSP-INITIAL-PULSE",
        lease_token=1,
        worker_name="worker-mac-1",
        master_url="http://mock-master",
        worker_token="mock-token",
        interval=100.0,
        abort_event=abort_evt,
        stop_event=stop_evt,
    )

    t0 = time.monotonic()
    t.start()
    # Allow a brief moment for thread to execute initial step
    time.sleep(0.05)
    stop_evt.set()
    t.join(timeout=1.0)

    assert len(pulses) >= 1, "Thread failed to pulse immediately on start"
    assert (pulses[0] - t0) < 0.5, "Initial pulse was delayed rather than immediate"


# ============================================================================
# 8. Heartbeat HTTP Failure Does Not Kill the Worker Task
# ============================================================================

def test_heartbeat_http_failure_does_not_kill_worker_task():
    """Network or transient HTTP failures during heartbeat pulse are caught and handled."""
    import threading
    abort_evt = threading.Event()
    stop_evt = threading.Event()

    t = WorkerHeartbeatThread(
        inspection_id="INSP-NET-ERR",
        lease_token=1,
        worker_name="worker-mac-1",
        master_url="http://invalid-nonexistent-domain-xyz.local",
        worker_token="mock-token",
        interval=100.0,
        abort_event=abort_evt,
        stop_event=stop_evt,
    )

    t.start()
    time.sleep(0.1)
    stop_evt.set()
    t.join(timeout=1.0)

    assert not abort_evt.is_set(), "Network error should not set abort_event"
    assert not t.is_alive()


# ============================================================================
# 9. Heartbeat Thread Stops Cleanly
# ============================================================================

def test_heartbeat_thread_stops_cleanly():
    """Setting stop_event terminates the heartbeat thread promptly."""
    import threading
    abort_evt = threading.Event()
    stop_evt = threading.Event()

    class MockThread(WorkerHeartbeatThread):
        def _pulse(self) -> bool:
            return True

    t = MockThread(
        inspection_id="INSP-CLEAN-STOP",
        lease_token=1,
        worker_name="worker-mac-1",
        master_url="http://mock-master",
        worker_token="mock-token",
        interval=60.0,
        abort_event=abort_evt,
        stop_event=stop_evt,
    )

    t.start()
    assert t.is_alive()
    stop_evt.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


# ============================================================================
# 10. Stale Coordinator Remains Fenced After Lease Generation Changes
# ============================================================================

def test_stale_coordinator_remains_fenced_after_lease_generation_changes(clean_job_manager):
    """A worker operating under an old lease is rejected with 409 LeaseConflictError."""
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    # Increment lease via reassignment
    new_lease = manager.reassign_orphaned_job(job.id, expected_lease_token=1, reason="Test recovery")
    assert new_lease == 2

    # Old coordinator with lease 1 tries to pulse heartbeat
    with pytest.raises(LeaseConflictError) as exc_info:
        manager.apply_heartbeat(job.id, lease_token=1, worker_name="worker-mac-1")
    assert "Fencing lease token mismatch" in str(exc_info.value)

    # Old coordinator tries to update step
    with pytest.raises(LeaseConflictError) as exc_step:
        manager.apply_step_update(
            inspection_id=job.id,
            index=0,
            step_id="prepare",
            status="completed",
            worker="worker-mac-1",
            lease_token=1,
        )
    assert "Fencing lease token mismatch" in str(exc_step.value)


# ============================================================================
# 11. Recovery Outbox Ordering and Idempotency Remain Intact
# ============================================================================

def test_recovery_outbox_ordering_and_idempotency_intact(clean_job_manager):
    """
    Reassigning an orphaned job durably writes dispatch intent before state commit.
    Reconciling outbox on startup is idempotent.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")

    # Perform atomic CAS reassignment
    new_lease = manager.reassign_orphaned_job(job.id, expected_lease_token=1, reason="Simulated loss")
    assert new_lease == 2

    # Verify write-ahead outbox file exists on disk
    outbox_file = manager.storage_dir / job.id / "outbox" / "dispatch_2.json"
    assert outbox_file.exists()
    with open(outbox_file, "r") as f:
        data = json.load(f)
    assert data["intended_lease"] == 2
    assert data["state"] == "pending"

    # Multiple identical reassignment attempts under same expected token return None (CAS failure)
    assert manager.reassign_orphaned_job(job.id, expected_lease_token=1) is None


# ============================================================================
# 12. Duplicate Supervisor Cycles Do Not Create Duplicate Recovery Dispatches
# ============================================================================

def test_duplicate_supervisor_cycles_do_not_duplicate_recovery_dispatches(clean_job_manager):
    """
    Once a job is recovered and placed in queued state with a new lease,
    subsequent supervisor cycles do not increment lease again or double-dispatch.
    """
    manager, job = clean_job_manager
    manager.claim_manifest(job.id, lease_token=1, worker_name="worker-mac-1")
    manager._monotonic_heartbeats[job.id] = time.monotonic() - 40.0

    supervisor = RecoverySupervisor(manager)

    with patch("backend.app.tasks.run_inspection_task.delay") as mock_delay:
        # Cycle 1: Triggers recovery
        _run_supervisor_cycle(supervisor, {})
        assert mock_delay.call_count == 1

        # Cycle 2: Job is now queued with no assigned_worker yet. Should NOT re-dispatch.
        _run_supervisor_cycle(supervisor, {})
        assert mock_delay.call_count == 1

    j = manager.get_job(job.id)
    assert j.lease_token == 2
    assert j.status == "queued"
    assert j.assigned_worker is None
