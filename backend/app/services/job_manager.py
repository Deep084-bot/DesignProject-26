"""
Job Manager and Background Execution Service for Surface.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.1 & 4.2.
Manages inspection job lifecycles:
- Isolated filesystem workspace per job (backend/storage/jobs/{inspection_id})
- Safe file naming and artifact containment
- Master-authoritative fencing leases and worker identity binding
- Periodic heartbeats and monotonic elapsed time tracking
- Crash-safe file-backed outbox for reliable dispatch
- Two-phase atomic artifact staging and replacement
- Thread-safe inspection state tracking and persistent JSON synchronization
- Unified state updates via apply_step_update, apply_result, apply_failure
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import fcntl
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

from ..models.inspection import (
    InspectionResponse,
    ImageFileMeta,
    CalibrationSummary,
    Finding,
    FanoutRunState,
    BatchRecord,
    BatchDefinition,
    FanoutStatus,
)
from ..models.calibration import StereoCalibration
from ..pipeline.runner import run_pipeline, PipelineExecutionResult
from .. import settings


# Base storage directory
BASE_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "jobs"


class LeaseConflictError(Exception):
    """Raised when an operation is attempted with an invalid, expired, or mismatched lease token."""
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class JobManager:
    def __init__(self, storage_dir: Path = BASE_STORAGE_DIR, max_workers: int = 2):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="surface-worker")
        self._lock = threading.Lock()
        self._jobs: Dict[str, InspectionResponse] = {}
        self._fanout_states: Dict[str, FanoutRunState] = {}
        self._monotonic_heartbeats: Dict[str, float] = {}
        self._lock_file_fd = None
        self._load_existing_jobs()

    def acquire_storage_lock(self):
        """
        Acquires an exclusive advisory OS-level lock on the storage directory.
        Fails fast if another Surface master process is already running.
        """
        lock_path = self.storage_dir.parent / ".master.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self._lock_file_fd is None:
            self._lock_file_fd = open(lock_path, "a+")
        try:
            fcntl.flock(self._lock_file_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(
                f"Another Surface master process is already running and holds the storage lock at {lock_path}."
            ) from exc

    def release_storage_lock(self):
        """Releases the exclusive advisory storage lock if held."""
        if self._lock_file_fd is not None:
            try:
                fcntl.flock(self._lock_file_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._lock_file_fd.close()
            except Exception:
                pass
            self._lock_file_fd = None

    def _load_existing_jobs(self):
        """Scans storage directory and loads prior inspections and fanout states into memory."""
        with self._lock:
            for job_dir in self.storage_dir.iterdir():
                if job_dir.is_dir():
                    meta_file = job_dir / "inspection.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            inspection = InspectionResponse(**data)
                            self._jobs[inspection.id] = inspection
                        except Exception:
                            # Skip corrupted historical records
                            pass
                    fanout_file = job_dir / "fanout_state.json"
                    if fanout_file.exists():
                        try:
                            with open(fanout_file, "r", encoding="utf-8") as f:
                                f_data = json.load(f)
                            fanout = FanoutRunState(**f_data)
                            self._fanout_states[fanout.inspection_id] = fanout
                            if fanout.inspection_id in self._jobs:
                                job = self._jobs[fanout.inspection_id]
                                needs_sync = (
                                    job.fanout_status != fanout.status
                                    or job.fanout_run_id != fanout.fanout_run_id
                                    or job.delegation_token != fanout.delegation_token
                                    or job.finalizer_worker != fanout.finalizer_worker
                                    or job.finalizer_generation != fanout.finalizer_generation
                                )
                                job.fanout_status = fanout.status
                                job.fanout_run_id = fanout.fanout_run_id
                                job.delegation_token = fanout.delegation_token
                                job.finalizer_worker = fanout.finalizer_worker
                                job.finalizer_generation = fanout.finalizer_generation
                                if needs_sync:
                                    self._save_job_state(job)
                        except Exception:
                            pass

    def _save_job_state(self, inspection: InspectionResponse):
        """Persists current inspection model to its job directory atomically."""
        job_dir = self.storage_dir / inspection.id
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_file = job_dir / "inspection.json"
        tmp_file = job_dir / f".inspection_{uuid.uuid4().hex}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(inspection.model_dump(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, meta_file)
        try:
            dir_fd = os.open(str(job_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

    def _save_fanout_state_locked(self, fanout: FanoutRunState):
        """Persists authoritative fan-out state to fanout_state.json atomically."""
        job_dir = self.storage_dir / fanout.inspection_id
        job_dir.mkdir(parents=True, exist_ok=True)
        final_file = job_dir / "fanout_state.json"
        tmp_file = job_dir / f".fanout_{uuid.uuid4().hex}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(fanout.model_dump(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, final_file)
        try:
            dir_fd = os.open(str(job_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

    def _validate_coordinator_lease_locked(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: Optional[str] = None,
    ):
        """
        Validates coordinator ownership and fencing lease within self._lock.
        Raises KeyError if inspection does not exist.
        Raises LeaseConflictError if job is in a terminal state, lease token mismatches,
        or worker identity mismatches.
        """
        if inspection_id not in self._jobs:
            raise KeyError(f"Inspection '{inspection_id}' not found.")
        job = self._jobs[inspection_id]
        if job.status in ("completed", "failed"):
            raise LeaseConflictError(f"Job '{inspection_id}' is already in terminal state '{job.status}'.")
        if job.lease_token != lease_token:
            raise LeaseConflictError(
                f"Fencing lease token mismatch: job lease is {job.lease_token}, but request provided {lease_token}."
            )
        if worker_name and job.assigned_worker:
            assigned_norm = settings.normalize_worker_identity(job.assigned_worker)
            req_norm = settings.normalize_worker_identity(worker_name)
            if assigned_norm != req_norm:
                raise LeaseConflictError(
                    f"Worker identity mismatch: job '{inspection_id}' is bound to worker '{job.assigned_worker}', "
                    f"but request came from '{worker_name}'."
                )

    def _validate_batch_operation_locked(
        self,
        inspection_id: str,
        lease_token: int,
        fanout_run_id: str,
        batch_id: int,
    ):
        """
        Validates batch-level authority within self._lock.
        Allows any cluster worker with valid worker token and active lease.
        Enforces matching fanout_run_id and FANOUT_RUNNING status.
        """
        if inspection_id not in self._jobs:
            raise KeyError(f"Inspection '{inspection_id}' not found.")
        job = self._jobs[inspection_id]
        if job.status in ("completed", "failed"):
            raise LeaseConflictError(f"Job '{inspection_id}' is already in terminal state '{job.status}'.")
        if job.lease_token != lease_token:
            raise LeaseConflictError(
                f"Fencing lease token mismatch: job lease is {job.lease_token}, but request provided {lease_token}."
            )
        fanout = self._fanout_states.get(inspection_id)
        if not fanout or fanout.lease_token != lease_token or fanout.fanout_run_id != fanout_run_id:
            raise LeaseConflictError(
                f"Fanout mismatch: run id '{fanout_run_id}' or lease {lease_token} not active for job '{inspection_id}'."
            )
        if fanout.status != "FANOUT_RUNNING":
            if (
                fanout.status in ("ALL_BATCHES_COMMITTED", "CALLBACK_PENDING")
                or (str(batch_id) in fanout.batches and fanout.batches[str(batch_id)].state == "committed")
            ):
                pass
            else:
                raise LeaseConflictError(
                    f"Cannot perform batch operation in fanout state '{fanout.status}' (expected FANOUT_RUNNING)."
                )
        if str(batch_id) not in fanout.batches:
            raise KeyError(f"Batch {batch_id} not found in fanout definition for job '{inspection_id}'.")

    def _validate_finalizer_lease_locked(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        delegation_token: str,
        finalizer_generation: int,
    ):
        """
        Validates finalizer delegation credentials within self._lock.
        Requires matching lease token, delegation token, generation, and non-expired window.
        """
        if inspection_id not in self._jobs:
            raise KeyError(f"Inspection '{inspection_id}' not found.")
        job = self._jobs[inspection_id]
        if job.status in ("completed", "failed"):
            raise LeaseConflictError(f"Job '{inspection_id}' is already in terminal state '{job.status}'.")
        if job.lease_token != lease_token:
            raise LeaseConflictError(
                f"Fencing lease token mismatch: job lease is {job.lease_token}, but request provided {lease_token}."
            )
        fanout = self._fanout_states.get(inspection_id)
        if not fanout or fanout.lease_token != lease_token:
            raise LeaseConflictError(f"No active fanout state matching lease {lease_token} for job '{inspection_id}'.")
        if fanout.status != "FINALIZATION_RUNNING":
            raise LeaseConflictError(
                f"Finalizer operation rejected: fanout status is '{fanout.status}', expected 'FINALIZATION_RUNNING'."
            )
        if fanout.delegation_token != delegation_token:
            raise LeaseConflictError("Finalizer delegation token mismatch or revoked.")
        if fanout.finalizer_worker and worker_name:
            finalizer_norm = settings.normalize_worker_identity(fanout.finalizer_worker)
            req_norm = settings.normalize_worker_identity(worker_name)
            if finalizer_norm != req_norm:
                raise LeaseConflictError(
                    f"Finalizer worker mismatch: delegated to '{fanout.finalizer_worker}', but request from '{worker_name}'."
                )
        if fanout.finalizer_generation != finalizer_generation:
            raise LeaseConflictError(
                f"Finalizer generation mismatch: current generation is {fanout.finalizer_generation}, "
                f"but request provided {finalizer_generation}."
            )
        now_wall = time.time()
        if fanout.finalizer_expires_at is not None and now_wall > fanout.finalizer_expires_at:
            raise LeaseConflictError(
                f"Finalizer delegation has expired (expired at {fanout.finalizer_expires_at}, now {now_wall})."
            )

    def _validate_lease_locked(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: Optional[str] = None,
        delegation_token: Optional[str] = None,
        finalizer_generation: Optional[int] = None,
    ):
        """
        Unified lease validation within self._lock.
        If delegation_token is provided, delegates to _validate_finalizer_lease_locked.
        Otherwise, validates coordinator lease via _validate_coordinator_lease_locked.
        """
        if delegation_token is not None:
            if not worker_name or finalizer_generation is None:
                raise LeaseConflictError("Finalizer operation requires worker_name and finalizer_generation.")
            self._validate_finalizer_lease_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                worker_name=worker_name,
                delegation_token=delegation_token,
                finalizer_generation=finalizer_generation,
            )
        else:
            self._validate_coordinator_lease_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                worker_name=worker_name,
            )

    def claim_manifest(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
    ) -> InspectionResponse:
        """
        Claims a job manifest and binds worker identity to the current lease token.
        Raises KeyError if inspection does not exist.
        Raises LeaseConflictError on lease or worker mismatch.
        """
        with self._lock:
            self._validate_lease_locked(inspection_id, lease_token, worker_name)
            job = self._jobs[inspection_id]
            canonical_worker = settings.normalize_worker_identity(worker_name) or worker_name
            job.assigned_worker = canonical_worker
            now_epoch = time.time()
            job.last_heartbeat_at = now_epoch
            self._monotonic_heartbeats[inspection_id] = time.monotonic()
            if job.status == "queued":
                job.status = "processing"
            self._save_job_state(job)
            return job

    def apply_heartbeat(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
    ) -> bool:
        """
        Records a worker heartbeat for an active lease.
        Raises KeyError or LeaseConflictError on validation failure.
        """
        with self._lock:
            self._validate_lease_locked(inspection_id, lease_token, worker_name)
            job = self._jobs[inspection_id]
            job.last_heartbeat_at = time.time()
            self._monotonic_heartbeats[inspection_id] = time.monotonic()
            self._save_job_state(job)
            return True

    def get_heartbeat_age(self, inspection_id: str) -> Optional[float]:
        """
        Returns elapsed seconds since the last heartbeat for this inspection.
        Uses monotonic time if recorded in memory; falls back to wall-clock time
        relative to last_heartbeat_at for cross-process / post-restart continuity.
        Returns None if no heartbeat was ever recorded.
        """
        with self._lock:
            if inspection_id in self._monotonic_heartbeats:
                return round(time.monotonic() - self._monotonic_heartbeats[inspection_id], 3)
            job = self._jobs.get(inspection_id)
            if job and job.last_heartbeat_at is not None:
                return round(max(0.0, time.time() - job.last_heartbeat_at), 3)
            return None

    def apply_step_update(
        self,
        inspection_id: str,
        index: int,
        step_id: str,
        status: str,
        worker: Optional[str] = None,
        duration_sec: Optional[float] = None,
        message: Optional[str] = None,
        lease_token: Optional[int] = None,
        delegation_token: Optional[str] = None,
        finalizer_generation: Optional[int] = None,
    ) -> bool:
        """
        Thread-safe application of a stage progress update.
        If lease_token is provided, validates fencing lease inside self._lock.
        Ignores late updates if job is already in terminal state when lease_token is None.
        Raises LeaseConflictError on fencing mismatch if lease_token is specified.
        Returns True if updated, False if ignored or not found.
        """
        with self._lock:
            if inspection_id not in self._jobs:
                return False
            job = self._jobs[inspection_id]

            if lease_token is not None:
                self._validate_lease_locked(
                    inspection_id=inspection_id,
                    lease_token=lease_token,
                    worker_name=worker,
                    delegation_token=delegation_token,
                    finalizer_generation=finalizer_generation,
                )
            else:
                if job.status in ("completed", "failed"):
                    return False

            if job.status == "queued" and status in ("active", "processing"):
                job.status = "processing"

            job.current_step_index = index
            if 0 <= index < len(job.steps):
                step = job.steps[index]
                step.status = status  # type: ignore
                if worker is not None:
                    step.worker = worker
                if duration_sec is not None:
                    step.duration_sec = duration_sec

            if status == "failed" and message:
                job.failure_message = message

            # Refresh heartbeat timestamp on step activity
            job.last_heartbeat_at = time.time()
            self._monotonic_heartbeats[inspection_id] = time.monotonic()

            self._save_job_state(job)
            return True

    def apply_result(
        self,
        inspection_id: str,
        point_count: int,
        mesh_status: str,
        mesh_error: Optional[str] = None,
        triangle_count: Optional[int] = None,
        deviation_summary: Optional[Dict[str, Any]] = None,
        findings: Optional[List[Any]] = None,
        classifier_status: Optional[str] = None,
        classifier_error: Optional[str] = None,
        duration_sec: float = 0.0,
        worker: Optional[str] = None,
        lease_token: Optional[int] = None,
        delegation_token: Optional[str] = None,
        finalizer_generation: Optional[int] = None,
    ) -> bool:
        """
        Thread-safe application of successful inspection results.
        If lease_token is provided, validates fencing lease.
        Duplicate completion on an already completed job returns True idempotently.
        """
        with self._lock:
            if inspection_id not in self._jobs:
                return False
            job = self._jobs[inspection_id]

            if lease_token is not None:
                if job.status == "completed":
                    return True
                self._validate_lease_locked(
                    inspection_id=inspection_id,
                    lease_token=lease_token,
                    worker_name=worker,
                    delegation_token=delegation_token,
                    finalizer_generation=finalizer_generation,
                )
            else:
                if job.status in ("completed", "failed"):
                    return False

            job.status = "completed"
            job.current_step_index = len(job.steps) - 1
            job.processing_duration_sec = duration_sec
            job.point_count = point_count
            job.point_cloud_url = f"/api/inspections/{inspection_id}/artifacts/pointcloud.ply"
            job.mesh_status = mesh_status  # type: ignore
            job.mesh_error = mesh_error

            job_dir = self.storage_dir / inspection_id
            if (job_dir / "mesh.ply").exists() or (mesh_status == "completed" and triangle_count):
                job.mesh_url = f"/api/inspections/{inspection_id}/artifacts/mesh.ply"
                job.triangle_count = triangle_count
            else:
                job.mesh_url = None
                job.triangle_count = None

            if (job_dir / "deviation_heatmap.png").exists() or (deviation_summary is not None):
                job.deviation_heatmap_url = f"/api/inspections/{inspection_id}/artifacts/deviation_heatmap.png"
            else:
                job.deviation_heatmap_url = None

            job.deviation_summary = deviation_summary
            job.failure_message = None

            # Process findings, converting any dict inputs to Finding models
            parsed_findings: List[Finding] = []
            for item in findings or []:
                if isinstance(item, dict):
                    finding_obj = Finding(**item)
                else:
                    finding_obj = item
                if finding_obj.image_crop_url and "/api/inspections/" in finding_obj.image_crop_url:
                    parts = finding_obj.image_crop_url.split("/")
                    if len(parts) >= 5 and parts[2] == "inspections":
                        parts[3] = inspection_id
                        finding_obj.image_crop_url = "/".join(parts)
                parsed_findings.append(finding_obj)

            job.findings = parsed_findings
            job.classifier_status = classifier_status  # type: ignore
            job.classifier_error = classifier_error

            for s in job.steps:
                s.status = "completed"
                if worker and not s.worker:
                    s.worker = worker

            # Synchronize fanout state to TERMINAL if active
            fanout = self._fanout_states.get(inspection_id)
            if fanout:
                fanout.status = "TERMINAL"
                fanout.finalized_at = time.time()
                self._save_fanout_state_locked(fanout)
                job.fanout_status = "TERMINAL"

            self._save_job_state(job)
            return True

    def apply_failure(
        self,
        inspection_id: str,
        message: str,
        failed_stage_index: Optional[int] = None,
        worker: Optional[str] = None,
        lease_token: Optional[int] = None,
        delegation_token: Optional[str] = None,
        finalizer_generation: Optional[int] = None,
    ) -> bool:
        """
        Thread-safe application of a job failure.
        If lease_token is provided, validates fencing lease.
        Ignores failure update if job is already in terminal state.
        """
        with self._lock:
            if inspection_id not in self._jobs:
                return False
            job = self._jobs[inspection_id]

            if lease_token is not None:
                if job.status in ("completed", "failed"):
                    return False
                self._validate_lease_locked(
                    inspection_id=inspection_id,
                    lease_token=lease_token,
                    worker_name=worker,
                    delegation_token=delegation_token,
                    finalizer_generation=finalizer_generation,
                )
            else:
                if job.status in ("completed", "failed"):
                    return False

            job.status = "failed"
            job.failure_message = message
            if failed_stage_index is not None and 0 <= failed_stage_index < len(job.steps):
                job.steps[failed_stage_index].status = "failed"
                if worker:
                    job.steps[failed_stage_index].worker = worker

            # Synchronize fanout state to TERMINAL if active
            fanout = self._fanout_states.get(inspection_id)
            if fanout:
                fanout.status = "TERMINAL"
                fanout.failure_reason = message
                self._save_fanout_state_locked(fanout)
                job.fanout_status = "TERMINAL"

            self._save_job_state(job)
            return True

    def commit_artifact(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        artifact_name: str,
        staging_path: Path,
        delegation_token: Optional[str] = None,
        finalizer_generation: Optional[int] = None,
    ) -> bool:
        """
        Validates fencing lease within self._lock and atomically commits
        a staged artifact file to the job directory via os.replace.
        Cleans up staging_path on validation failure.
        """
        with self._lock:
            try:
                self._validate_lease_locked(
                    inspection_id=inspection_id,
                    lease_token=lease_token,
                    worker_name=worker_name,
                    delegation_token=delegation_token,
                    finalizer_generation=finalizer_generation,
                )
            except Exception:
                staging_path.unlink(missing_ok=True)
                raise

            job = self._jobs[inspection_id]
            target_path = self.storage_dir / inspection_id / artifact_name
            os.replace(staging_path, target_path)

            job.last_heartbeat_at = time.time()
            self._monotonic_heartbeats[inspection_id] = time.monotonic()
            return True

    def write_outbox_intent(
        self,
        inspection_id: str,
        lease_token: int,
        previous_lease: Optional[int] = None,
        reason: str = "",
    ) -> Path:
        """
        Writes crash-safe write-ahead dispatch intent file using atomic staging, fsync, and os.replace.
        Guarantees that intent exists durably before lease transition is committed.
        """
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        outbox_dir.mkdir(parents=True, exist_ok=True)
        final_file = outbox_dir / f"dispatch_{lease_token}.json"
        dispatch_id = uuid.uuid4().hex
        tmp_file = outbox_dir / f".dispatch_{lease_token}_{dispatch_id}.tmp"

        prev_lease = previous_lease if previous_lease is not None else max(0, lease_token - 1)
        payload = {
            "inspection_id": inspection_id,
            "dispatch_id": dispatch_id,
            "previous_lease": prev_lease,
            "intended_lease": lease_token,
            "created_at": time.time(),
            "state": "pending",
            "reason": reason,
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, final_file)
        try:
            dir_fd = os.open(str(outbox_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
        return final_file

    def mark_outbox_published(self, inspection_id: str, lease_token: int):
        """Atomically updates dispatch record state to 'published'."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"dispatch_{lease_token}.json"
        if not final_file.exists():
            return

        existing_data = {}
        try:
            with open(final_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            pass

        tmp_file = outbox_dir / f".dispatch_{lease_token}_{uuid.uuid4().hex}.tmp"
        payload = {
            "inspection_id": inspection_id,
            "dispatch_id": existing_data.get("dispatch_id", uuid.uuid4().hex),
            "previous_lease": existing_data.get("previous_lease", max(0, lease_token - 1)),
            "intended_lease": existing_data.get("intended_lease", lease_token),
            "created_at": existing_data.get("created_at", time.time()),
            "state": "published",
            "published_at": time.time(),
            "reason": existing_data.get("reason", ""),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, final_file)
        try:
            dir_fd = os.open(str(outbox_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

    def cleanup_outbox(self, inspection_id: str, lease_token: int):
        """Removes the outbox file once publication has succeeded."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"dispatch_{lease_token}.json"
        final_file.unlink(missing_ok=True)

    def write_finalizer_outbox_intent_locked(
        self,
        inspection_id: str,
        lease_token: int,
        fanout_run_id: str,
        generation: int = 0,
    ) -> Path:
        """
        Durably records a finalizer task dispatch intent before broker publication.
        """
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        outbox_dir.mkdir(parents=True, exist_ok=True)
        final_file = outbox_dir / f"finalizer_{lease_token}_{generation}.json"
        tmp_file = outbox_dir / f".finalizer_{lease_token}_{generation}_{uuid.uuid4().hex}.tmp"
        payload = {
            "inspection_id": inspection_id,
            "lease_token": lease_token,
            "fanout_run_id": fanout_run_id,
            "generation": generation,
            "created_at": time.time(),
            "state": "pending",
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, final_file)
        try:
            dir_fd = os.open(str(outbox_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
        return final_file

    def mark_finalizer_outbox_published(self, inspection_id: str, lease_token: int, generation: int = 0):
        """Atomically marks finalizer intent as published."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"finalizer_{lease_token}_{generation}.json"
        if not final_file.exists():
            return
        try:
            with open(final_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["state"] = "published"
            data["published_at"] = time.time()
            tmp_file = outbox_dir / f".finalizer_{lease_token}_{generation}_{uuid.uuid4().hex}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, final_file)
            try:
                dir_fd = os.open(str(outbox_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
        except Exception:
            pass

    def cleanup_finalizer_outbox(self, inspection_id: str, lease_token: int, generation: int = 0):
        """Removes the finalizer outbox record after successful dispatch."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"finalizer_{lease_token}_{generation}.json"
        final_file.unlink(missing_ok=True)

    def write_batch_outbox_intent_locked(
        self,
        inspection_id: str,
        lease_token: int,
        fanout_run_id: str,
        batch_id: int,
        reason: str = "",
    ) -> Path:
        """
        Durably records a batch task redispatch intent before broker publication.
        Uses temporary staging, flush, fsync, atomic os.replace, and parent directory fsync.
        """
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        outbox_dir.mkdir(parents=True, exist_ok=True)
        final_file = outbox_dir / f"batch_{lease_token}_{fanout_run_id}_{batch_id}.json"
        tmp_file = outbox_dir / f".batch_{lease_token}_{fanout_run_id}_{batch_id}_{uuid.uuid4().hex}.tmp"
        payload = {
            "inspection_id": inspection_id,
            "lease_token": lease_token,
            "fanout_run_id": fanout_run_id,
            "batch_id": batch_id,
            "previous_state": "claimed",
            "intended_state": "pending",
            "reason": reason,
            "created_at": time.time(),
            "state": "pending",
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, final_file)
        try:
            dir_fd = os.open(str(outbox_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
        return final_file

    def cleanup_batch_outbox(
        self,
        inspection_id: str,
        lease_token: int,
        fanout_run_id: str,
        batch_id: int,
    ):
        """Removes the batch outbox record after successful broker publication."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"batch_{lease_token}_{fanout_run_id}_{batch_id}.json"
        final_file.unlink(missing_ok=True)

    def reconcile_outbox_on_startup(self) -> int:
        """
        Scans storage for uncommitted/pending outbox records on master startup or supervisor tick.
        Handles:
        A. No intent: no recovery action.
        B. Durable intent exists, inspection still has previous lease: completes lease transition and dispatches.
        C. Durable intent exists, inspection already has intended lease: publishes or cleans up idempotently.
        D. Intent exists but inspection has newer lease: cleans up stale intent safely.
        E. Crash after broker publication: republishes (tolerates at-least-once duplicate delivery).
        F. Crash after marking published: cleans up file without duplicate publication.
        G. Redis unavailable: retains intent file and defers publication.
        Returns the number of re-published jobs.
        """
        republished = 0
        if not self.storage_dir.exists():
            return republished

        for job_dir in self.storage_dir.iterdir():
            if not job_dir.is_dir():
                continue
            outbox_dir = job_dir / "outbox"
            if not outbox_dir.exists() or not outbox_dir.is_dir():
                continue

            for item in list(outbox_dir.iterdir()):
                # Clean up dangling temporary files
                if (
                    item.name.startswith(".dispatch_")
                    or item.name.startswith(".finalizer_")
                    or item.name.startswith(".batch_")
                ) and item.name.endswith(".tmp"):
                    item.unlink(missing_ok=True)
                    continue

                if item.name.startswith("batch_") and item.name.endswith(".json"):
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        state = data.get("state")
                        insp_id = data.get("inspection_id")
                        intent_lease = data.get("lease_token", 1)
                        intent_run_id = data.get("fanout_run_id")
                        b_id = data.get("batch_id")

                        if state == "published":
                            item.unlink(missing_ok=True)
                            continue

                        should_dispatch = False
                        with self._lock:
                            job = self._jobs.get(insp_id)
                            if job is None or job.status in ("completed", "failed") or job.lease_token != intent_lease:
                                item.unlink(missing_ok=True)
                                continue

                            fanout = self._fanout_states.get(insp_id)
                            if not fanout or fanout.lease_token != intent_lease or fanout.fanout_run_id != intent_run_id:
                                item.unlink(missing_ok=True)
                                continue

                            batch = fanout.batches.get(str(b_id))
                            if not batch or batch.state == "committed":
                                item.unlink(missing_ok=True)
                                continue

                            # Interrupted transition: if still claimed, complete transition to pending
                            if batch.state == "claimed":
                                batch.state = "pending"
                                batch.claimed_by = None
                                batch.claim_token = None
                                batch.claim_expires_at = None
                                self._save_fanout_state_locked(fanout)

                            if batch.state == "pending" and fanout.status == "FANOUT_RUNNING":
                                should_dispatch = True

                        if should_dispatch and settings.get_execution_mode() == "celery":
                            try:
                                from ..tasks import classify_batch_task
                                classify_batch_task.delay(insp_id, intent_lease, intent_run_id, b_id)
                                republished += 1
                                self.cleanup_batch_outbox(insp_id, intent_lease, intent_run_id, b_id)
                            except Exception:
                                pass
                        elif not should_dispatch:
                            item.unlink(missing_ok=True)
                    except Exception:
                        item.unlink(missing_ok=True)
                    continue

                if item.name.startswith("finalizer_") and item.name.endswith(".json"):
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, OSError) as read_err:
                        logger.warning(f"Discarding unreadable finalizer intent {item.name}: {read_err}")
                        item.unlink(missing_ok=True)
                        continue

                    insp_id = data.get("inspection_id")
                    intent_lease = data.get("lease_token", 1)
                    intent_gen = data.get("generation", 0)
                    intent_run_id = data.get("fanout_run_id")
                    intent_state = data.get("state", "pending")

                    should_dispatch = False
                    with self._lock:
                        job = self._jobs.get(insp_id)
                        if job is None or job.status in ("completed", "failed") or job.lease_token != intent_lease:
                            item.unlink(missing_ok=True)
                            continue

                        fanout = self._fanout_states.get(insp_id)
                        if not fanout or fanout.lease_token != intent_lease or fanout.fanout_run_id != intent_run_id:
                            item.unlink(missing_ok=True)
                            continue

                        # Generation check:
                        # An intent is valid for dispatch if and only if it matches the authoritative committed generation.
                        # - intent_gen < fanout.finalizer_generation: Stale older generation (superseded by newer takeover).
                        # - intent_gen > fanout.finalizer_generation: Uncommitted future intent written before a crash prior
                        #   to fanout_state.json persistence. Must NOT be dispatched or advance generation; safely discarded.
                        if intent_gen != fanout.finalizer_generation:
                            logger.info(
                                f"Startup reconciliation: discarding mismatched finalizer intent {item.name} "
                                f"(intent gen {intent_gen} != authoritative gen {fanout.finalizer_generation})."
                            )
                            item.unlink(missing_ok=True)
                            continue

                        # Check if already durably claimed by an active, unexpired finalizer of this generation
                        is_actively_claimed = (
                            fanout.status == "FINALIZATION_RUNNING"
                            and fanout.finalizer_generation == intent_gen
                            and fanout.delegation_token is not None
                            and fanout.finalizer_expires_at is not None
                            and time.time() <= fanout.finalizer_expires_at
                        )
                        if is_actively_claimed:
                            item.unlink(missing_ok=True)
                            continue

                        # Valid unacknowledged finalizer intent for pending or expired callback
                        now_wall = time.time()
                        is_expired_finalizer = (
                            fanout.status == "FINALIZATION_RUNNING"
                            and fanout.finalizer_expires_at is not None
                            and now_wall > fanout.finalizer_expires_at
                        )
                        is_pending_callback = fanout.status in ("CALLBACK_PENDING", "ALL_BATCHES_COMMITTED")

                        if is_pending_callback or is_expired_finalizer:
                            if intent_state == "pending":
                                should_dispatch = True
                            elif intent_state == "published":
                                if fanout.callback_dispatched_at is None or (now_wall - fanout.callback_dispatched_at) > 30.0:
                                    should_dispatch = True
                        else:
                            # Not in a state that expects a finalizer
                            item.unlink(missing_ok=True)
                            continue

                    if should_dispatch and settings.get_execution_mode() == "celery":
                        try:
                            from ..tasks import finalize_inspection_task
                            finalize_inspection_task.delay(insp_id, intent_lease, intent_run_id)
                            republished += 1
                            self.mark_finalizer_outbox_published(insp_id, intent_lease, intent_gen)
                            with self._lock:
                                f_curr = self._fanout_states.get(insp_id)
                                if f_curr:
                                    f_curr.callback_dispatched_at = time.time()
                                    self._save_fanout_state_locked(f_curr)
                            logger.info(f"Startup reconciliation: republished finalizer for {insp_id} (gen {intent_gen}).")
                        except Exception as exc:
                            logger.error(f"Startup reconciliation: failed to publish finalizer for {insp_id}: {exc}")
                    continue

                if item.name.startswith("dispatch_") and item.name.endswith(".json"):
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        state = data.get("state")
                        insp_id = data.get("inspection_id")
                        intended_lease = data.get("intended_lease", data.get("lease_token", 1))
                        previous_lease = data.get("previous_lease", max(0, intended_lease - 1))

                        if state == "published":
                            # Requirement F: already published before crash, clean up without republishing
                            item.unlink(missing_ok=True)
                            continue

                        # state == "pending"
                        with self._lock:
                            job = self._jobs.get(insp_id)
                            if job is None or job.status in ("completed", "failed"):
                                # Job does not exist or already finished; clean up stale intent
                                item.unlink(missing_ok=True)
                                continue

                            # Requirement D: Stale intent where inspection already advanced to a newer lease
                            if job.lease_token > intended_lease:
                                item.unlink(missing_ok=True)
                                continue

                            # Requirement B: Interrupted transition where inspection still has previous lease
                            if job.lease_token == previous_lease:
                                job.lease_token = intended_lease
                                job.assigned_worker = None
                                job.last_heartbeat_at = None
                                job.recovery_pending = False
                                self._monotonic_heartbeats.pop(insp_id, None)
                                for s in job.steps:
                                    s.status = "waiting"
                                    s.duration_sec = None
                                    s.worker = None
                                job.current_step_index = 0
                                job.status = "queued"
                                self._save_job_state(job)

                            # Requirement C: Job is at intended lease and queued/processing
                            should_publish = (
                                job.status in ("queued", "processing")
                                and job.lease_token == intended_lease
                            )

                        if should_publish and settings.get_execution_mode() == "celery":
                            try:
                                from ..tasks import run_inspection_task
                                run_inspection_task.delay(insp_id, intended_lease)
                                republished += 1
                                self.mark_outbox_published(insp_id, intended_lease)
                                self.cleanup_outbox(insp_id, intended_lease)
                            except Exception:
                                # Requirement G: Redis unavailable; retain intent file and defer publication
                                pass
                        elif not should_publish:
                            item.unlink(missing_ok=True)
                    except Exception:
                        item.unlink(missing_ok=True)

        # Reconcile fan-out states: if ALL_BATCHES_COMMITTED without pending finalizer intent, repair it
        with self._lock:
            for fanout in list(self._fanout_states.values()):
                if fanout.status == "ALL_BATCHES_COMMITTED":
                    self.write_finalizer_outbox_intent_locked(
                        inspection_id=fanout.inspection_id,
                        lease_token=fanout.lease_token,
                        fanout_run_id=fanout.fanout_run_id,
                        generation=0,
                    )
                    fanout.status = "CALLBACK_PENDING"
                    fanout.callback_dispatched_at = time.time()
                    self._save_fanout_state_locked(fanout)
                    job = self._jobs.get(fanout.inspection_id)
                    if job:
                        job.fanout_status = "CALLBACK_PENDING"
                        self._save_job_state(job)

        return republished

    def reassign_orphaned_job(
        self,
        inspection_id: str,
        expected_lease_token: int,
        reason: str = "",
    ) -> Optional[int]:
        """
        Performs atomic CAS reassignment for an orphaned or dead-worker job using write-ahead recovery intent.
        1. Under lock, verifies expected_lease_token and recovery eligibility.
        2. Durably writes write-ahead recovery dispatch intent FIRST (dispatch_{new_lease}.json).
        3. Updates in-memory state and atomically persists inspection.json with the new lease generation.
        Returns the new lease_token, or None on CAS mismatch or terminal state.
        """
        with self._lock:
            if inspection_id not in self._jobs:
                return None
            job = self._jobs[inspection_id]
            if job.status in ("completed", "failed"):
                return None
            if job.lease_token != expected_lease_token:
                # Concurrent bump or mismatch
                return None

            fanout = self._fanout_states.get(inspection_id)
            if fanout and fanout.status in (
                "FANOUT_RUNNING",
                "ALL_BATCHES_COMMITTED",
                "CALLBACK_PENDING",
                "FINALIZATION_RUNNING",
            ):
                # Job is in fan-out or finalizer phase; preserve committed progress and do not reset coordinator
                return None

            new_lease = expected_lease_token + 1

            # 1. Write-Ahead Recovery Intent: durably write dispatch intent BEFORE committing lease in inspection.json
            self.write_outbox_intent(
                inspection_id=inspection_id,
                lease_token=new_lease,
                previous_lease=expected_lease_token,
                reason=reason,
            )

            # 2. Update job state and persist inspection.json
            job.lease_token = new_lease
            job.assigned_worker = None
            job.last_heartbeat_at = None
            job.recovery_pending = False
            self._monotonic_heartbeats.pop(inspection_id, None)

            # Reset pipeline steps back to initial state
            for s in job.steps:
                s.status = "waiting"
                s.duration_sec = None
                s.worker = None
            job.current_step_index = 0
            job.status = "queued"

            self._save_job_state(job)
            return new_lease

    def init_fanout(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        total_batches: int,
        batches: List[Any],
    ) -> FanoutRunState:
        """
        Initializes a distributed fan-out run for Stage 4 under coordinator lease validation.
        Creates durable fanout_state.json and updates derived inspection fields.
        """
        with self._lock:
            self._validate_coordinator_lease_locked(inspection_id, lease_token, worker_name)
            job = self._jobs[inspection_id]

            # Idempotent re-initialization check for same lease
            existing = self._fanout_states.get(inspection_id)
            if existing and existing.lease_token == lease_token and existing.status == "FANOUT_RUNNING":
                return existing

            fanout_run_id = f"fanout_{inspection_id}_{uuid.uuid4().hex[:8]}"
            batch_records: Dict[str, BatchRecord] = {}
            for item in batches:
                if isinstance(item, dict):
                    bid = int(item["batch_id"])
                    t_indices = list(item["tile_indices"])
                else:
                    bid = int(item.batch_id)
                    t_indices = list(item.tile_indices)
                batch_records[str(bid)] = BatchRecord(
                    batch_id=bid,
                    tile_indices=t_indices,
                    state="pending",
                )

            fanout = FanoutRunState(
                inspection_id=inspection_id,
                fanout_run_id=fanout_run_id,
                lease_token=lease_token,
                status="FANOUT_RUNNING",
                total_batches=total_batches,
                committed_batches=0,
                batches=batch_records,
                created_at=time.time(),
            )
            self._save_fanout_state_locked(fanout)
            self._fanout_states[inspection_id] = fanout

            job.fanout_status = "FANOUT_RUNNING"
            job.fanout_run_id = fanout_run_id
            job.last_heartbeat_at = time.time()
            self._monotonic_heartbeats[inspection_id] = time.monotonic()
            self._save_job_state(job)
            return fanout

    def claim_batch(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        fanout_run_id: str,
        batch_id: int,
        claim_timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Claims a batch for processing by an authenticated cluster worker.
        Validates lease token, fanout run id, and batch existence.
        Permits re-claim if previous claim has expired or was held by same worker.
        If actively claimed by another worker, returns 'already_claimed' with bounded retry_after_sec.
        """
        with self._lock:
            self._validate_batch_operation_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                fanout_run_id=fanout_run_id,
                batch_id=batch_id,
            )
            fanout = self._fanout_states[inspection_id]
            batch = fanout.batches[str(batch_id)]
            now_wall = time.time()

            if batch.state == "committed":
                return {
                    "status": "already_committed",
                    "batch_id": batch_id,
                    "tile_indices": batch.tile_indices,
                }

            if batch.state == "claimed":
                is_expired = (
                    batch.claim_expires_at is not None
                    and (now_wall >= batch.claim_expires_at or (batch.claimed_at and now_wall < batch.claimed_at - 5.0))
                )
                claimed_norm = settings.normalize_worker_identity(batch.claimed_by)
                req_norm = settings.normalize_worker_identity(worker_name)
                if not is_expired and claimed_norm != req_norm:
                    rem_sec = max(0.0, batch.claim_expires_at - now_wall)
                    retry_after = min(10.0, max(3.0, rem_sec / 2.0))
                    return {
                        "status": "already_claimed",
                        "batch_id": batch_id,
                        "claimed_by": batch.claimed_by,
                        "claim_expires_at": batch.claim_expires_at,
                        "retry_after_sec": round(retry_after, 2),
                    }

            claim_token = uuid.uuid4().hex
            batch.state = "claimed"
            batch.claimed_by = worker_name
            batch.claimed_at = now_wall
            batch.claim_expires_at = now_wall + claim_timeout_sec
            batch.claim_token = claim_token
            batch.claim_attempts += 1

            self._save_fanout_state_locked(fanout)
            return {
                "status": "claimed",
                "batch_id": batch_id,
                "claim_token": claim_token,
                "claim_expires_at": batch.claim_expires_at,
                "tile_indices": batch.tile_indices,
            }

    def commit_batch(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        fanout_run_id: str,
        batch_id: int,
        claim_token: str,
        probabilities: List[float],
    ) -> Dict[str, Any]:
        """
        Durably commits batch classification probabilities under lock.
        Validates claim token, probabilities format/bounds, and writes batch artifact atomically.
        Transitions fanout to ALL_BATCHES_COMMITTED and CALLBACK_PENDING when all batches are committed.
        """
        with self._lock:
            self._validate_batch_operation_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                fanout_run_id=fanout_run_id,
                batch_id=batch_id,
            )
            fanout = self._fanout_states[inspection_id]
            batch = fanout.batches[str(batch_id)]
            now_wall = time.time()

            # Input validation
            if not isinstance(probabilities, list):
                raise ValueError("probabilities must be a list of floats.")
            if len(probabilities) != len(batch.tile_indices):
                raise ValueError(
                    f"Probabilities count mismatch: expected {len(batch.tile_indices)}, got {len(probabilities)}."
                )
            for p in probabilities:
                if not isinstance(p, (int, float)) or p < 0.0 or p > 1.0 or p != p:
                    raise ValueError(f"Probability value {p} must be a valid float between 0.0 and 1.0.")

            # Idempotent re-commit
            if batch.state == "committed":
                return {
                    "status": "already_committed",
                    "batch_id": batch_id,
                    "committed_batches": fanout.committed_batches,
                    "fanout_status": fanout.status,
                }

            if batch.state != "claimed":
                raise LeaseConflictError(f"Batch {batch_id} is in state '{batch.state}', not claimed.")

            if batch.claim_token != claim_token:
                raise LeaseConflictError(f"Stale or mismatched claim token for batch {batch_id}.")

            if batch.claim_expires_at and now_wall >= batch.claim_expires_at:
                raise LeaseConflictError(f"Claim for batch {batch_id} expired at {batch.claim_expires_at}.")

            # Durably write batch file via staging + fsync + os.replace
            batches_dir = self.storage_dir / inspection_id / "batches"
            batches_dir.mkdir(parents=True, exist_ok=True)
            final_batch_file = batches_dir / f"batch_{batch_id:04d}.json"
            tmp_batch_file = batches_dir / f".batch_{batch_id:04d}_{uuid.uuid4().hex}.tmp"
            record_payload = {
                "batch_id": batch_id,
                "worker_name": worker_name,
                "claim_token": claim_token,
                "committed_at": now_wall,
                "tile_indices": batch.tile_indices,
                "probabilities": [float(p) for p in probabilities],
            }
            with open(tmp_batch_file, "w", encoding="utf-8") as f:
                json.dump(record_payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_batch_file, final_batch_file)
            try:
                bdir_fd = os.open(str(batches_dir), os.O_RDONLY)
                try:
                    os.fsync(bdir_fd)
                finally:
                    os.close(bdir_fd)
            except Exception:
                pass

            # Update batch record
            batch.state = "committed"
            batch.committed_at = now_wall
            fanout.committed_batches += 1

            # Check completion of all batches
            if fanout.committed_batches >= fanout.total_batches:
                fanout.status = "ALL_BATCHES_COMMITTED"
                fanout.all_committed_at = now_wall
                self.write_finalizer_outbox_intent_locked(
                    inspection_id=inspection_id,
                    lease_token=lease_token,
                    fanout_run_id=fanout.fanout_run_id,
                    generation=0,
                )
                fanout.status = "CALLBACK_PENDING"
                fanout.callback_dispatched_at = now_wall

            self._save_fanout_state_locked(fanout)
            job = self._jobs[inspection_id]
            job.fanout_status = fanout.status
            self._save_job_state(job)

            return {
                "status": "committed",
                "batch_id": batch_id,
                "committed_batches": fanout.committed_batches,
                "fanout_status": fanout.status,
            }

    def claim_finalizer(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        fanout_run_id: str,
        delegation_duration_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Claims finalizer delegation for post-fanout Stage 5 completion.
        Returns delegation_token, finalizer_generation, and finalizer_expires_at.
        """
        with self._lock:
            if inspection_id not in self._jobs:
                raise KeyError(f"Inspection '{inspection_id}' not found.")
            job = self._jobs[inspection_id]
            if job.status in ("completed", "failed"):
                raise LeaseConflictError(f"Job '{inspection_id}' is already in terminal state '{job.status}'.")
            if job.lease_token != lease_token:
                raise LeaseConflictError(
                    f"Fencing lease token mismatch: job lease is {job.lease_token}, but request provided {lease_token}."
                )
            fanout = self._fanout_states.get(inspection_id)
            if not fanout or fanout.lease_token != lease_token or fanout.fanout_run_id != fanout_run_id:
                raise LeaseConflictError("Active fanout run does not match request credentials.")

            now_wall = time.time()
            if fanout.status in ("NOT_STARTED", "FANOUT_RUNNING"):
                raise LeaseConflictError(
                    f"Cannot claim finalizer in state '{fanout.status}': batches are not yet all committed."
                )
            if fanout.status == "TERMINAL":
                raise LeaseConflictError("Job is already in terminal state.")

            if fanout.status == "FINALIZATION_RUNNING":
                is_expired = fanout.finalizer_expires_at is not None and now_wall > fanout.finalizer_expires_at
                if is_expired:
                    fanout.finalizer_generation += 1
                elif settings.normalize_worker_identity(fanout.finalizer_worker) != settings.normalize_worker_identity(worker_name):
                    raise LeaseConflictError(
                        f"Finalizer delegation is currently held by '{fanout.finalizer_worker}' "
                        f"until {fanout.finalizer_expires_at}."
                    )
            elif fanout.status == "CALLBACK_PENDING":
                fanout.callback_arrived_at = now_wall
                fanout.status = "FINALIZATION_RUNNING"

            canonical_worker = settings.normalize_worker_identity(worker_name) or worker_name
            delegation_token = f"del_{uuid.uuid4().hex}"
            fanout.delegation_token = delegation_token
            fanout.finalizer_worker = canonical_worker
            fanout.finalizer_claimed_at = now_wall
            fanout.finalizer_expires_at = now_wall + delegation_duration_sec

            self._save_fanout_state_locked(fanout)

            job.fanout_status = "FINALIZATION_RUNNING"
            job.delegation_token = delegation_token
            job.finalizer_worker = canonical_worker
            job.finalizer_generation = fanout.finalizer_generation
            self._save_job_state(job)

            # Durable claim acknowledgement: clean up finalizer outbox intent(s) for this generation
            self.cleanup_finalizer_outbox(inspection_id, lease_token, fanout.finalizer_generation)
            if fanout.finalizer_generation > 0:
                for prev_g in range(fanout.finalizer_generation):
                    self.cleanup_finalizer_outbox(inspection_id, lease_token, prev_g)

            return {
                "status": "claimed",
                "delegation_token": delegation_token,
                "finalizer_generation": fanout.finalizer_generation,
                "finalizer_expires_at": fanout.finalizer_expires_at,
            }

    def apply_finalizer_heartbeat(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        delegation_token: str,
        finalizer_generation: int,
        extension_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Renews finalizer delegation lease window.
        """
        with self._lock:
            self._validate_finalizer_lease_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                worker_name=worker_name,
                delegation_token=delegation_token,
                finalizer_generation=finalizer_generation,
            )
            fanout = self._fanout_states[inspection_id]
            now_wall = time.time()
            fanout.finalizer_expires_at = now_wall + extension_sec
            self._save_fanout_state_locked(fanout)
            return {
                "status": "ok",
                "finalizer_expires_at": fanout.finalizer_expires_at,
            }

    def get_fanout_status(self, inspection_id: str) -> Optional[FanoutRunState]:
        """Returns the active fanout run state for an inspection, if any."""
        with self._lock:
            return self._fanout_states.get(inspection_id)

    def apply_batch_heartbeat(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        fanout_run_id: str,
        batch_id: int,
        claim_token: str,
        extension_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Renews an active batch claim window for an authenticated worker under lock.
        Enforces a hard ceiling of 30.0 seconds.
        """
        with self._lock:
            self._validate_batch_operation_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                fanout_run_id=fanout_run_id,
                batch_id=batch_id,
            )
            fanout = self._fanout_states[inspection_id]
            batch = fanout.batches[str(batch_id)]
            now_wall = time.time()

            if batch.state != "claimed":
                raise LeaseConflictError(f"Batch {batch_id} is in state '{batch.state}', not claimed.")
            claimed_norm = settings.normalize_worker_identity(batch.claimed_by)
            req_norm = settings.normalize_worker_identity(worker_name)
            if claimed_norm != req_norm:
                raise LeaseConflictError(f"Batch {batch_id} is claimed by '{batch.claimed_by}', not '{worker_name}'.")
            if batch.claim_token != claim_token:
                raise LeaseConflictError(f"Stale or mismatched claim token for batch {batch_id}.")
            if batch.claim_expires_at is not None and now_wall >= batch.claim_expires_at:
                raise LeaseConflictError(
                    f"Claim for batch {batch_id} expired at {batch.claim_expires_at} (now {now_wall})."
                )

            safe_extension = min(30.0, max(1.0, float(extension_sec)))
            batch.claim_expires_at = now_wall + safe_extension
            self._save_fanout_state_locked(fanout)
            return {
                "status": "ok",
                "batch_id": batch_id,
                "claim_expires_at": batch.claim_expires_at,
            }

    def get_committed_batches_locked(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        delegation_token: str,
        finalizer_generation: int,
    ) -> Dict[str, Any]:
        """
        Retrieves committed batch classification results for delegated finalizer.
        Validates finalizer lease, generation, completeness, and value bounds.
        """
        with self._lock:
            self._validate_finalizer_lease_locked(
                inspection_id=inspection_id,
                lease_token=lease_token,
                worker_name=worker_name,
                delegation_token=delegation_token,
                finalizer_generation=finalizer_generation,
            )
            fanout = self._fanout_states[inspection_id]
            if fanout.committed_batches < fanout.total_batches:
                raise LeaseConflictError(
                    f"Cannot retrieve committed batches: only {fanout.committed_batches}/{fanout.total_batches} committed."
                )

            batches_dir = self.storage_dir / inspection_id / "batches"
            results = []
            for b_idx in range(fanout.total_batches):
                b_str = str(b_idx)
                if b_str not in fanout.batches:
                    raise KeyError(f"Batch {b_idx} missing in fanout definition.")
                batch_rec = fanout.batches[b_str]
                if batch_rec.state != "committed":
                    raise LeaseConflictError(f"Batch {b_idx} is not in committed state.")

                batch_file = batches_dir / f"batch_{b_idx:04d}.json"
                if not batch_file.exists():
                    raise FileNotFoundError(f"Missing durable batch file: {batch_file.name}")

                with open(batch_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("batch_id") != b_idx:
                    raise ValueError(f"Corrupt batch file {batch_file.name}: batch_id mismatch.")
                probs = data.get("probabilities", [])
                tile_indices = data.get("tile_indices", [])
                if len(probs) != len(tile_indices):
                    raise ValueError(f"Corrupt batch file {batch_file.name}: probabilities count mismatch.")
                for p in probs:
                    if not isinstance(p, (int, float)) or p < 0.0 or p > 1.0 or p != p:
                        raise ValueError(f"Corrupt batch file {batch_file.name}: invalid probability value {p}.")

                results.append({
                    "batch_id": b_idx,
                    "tile_indices": tile_indices,
                    "probabilities": [float(p) for p in probs],
                })

            return {
                "inspection_id": inspection_id,
                "fanout_run_id": fanout.fanout_run_id,
                "total_batches": fanout.total_batches,
                "committed_batches": fanout.committed_batches,
                "batches": results,
            }

    def create_job(
        self,
        inspection_id: str,
        name: str,
        surface_type: str,
        reconstruction_method: str,
        left_bytes: bytes,
        left_filename: str,
        right_bytes: bytes,
        right_filename: str,
        calibration: StereoCalibration,
        calibration_summary: Optional[CalibrationSummary],
        raw_calibration_bytes: Optional[bytes] = None,
        dispatch_mode: Optional[str] = None,
    ) -> InspectionResponse:
        """
        Initializes a job directory, writes sanitized input files, records job state,
        and optionally schedules local background processing depending on execution mode.
        """
        job_dir = self.storage_dir / inspection_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize extension
        left_ext = Path(left_filename).suffix.lower() or ".jpg"
        right_ext = Path(right_filename).suffix.lower() or ".jpg"
        if left_ext not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            left_ext = ".png"
        if right_ext not in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            right_ext = ".png"

        left_path = job_dir / f"input_left{left_ext}"
        right_path = job_dir / f"input_right{right_ext}"

        with open(left_path, "wb") as f:
            f.write(left_bytes)
        with open(right_path, "wb") as f:
            f.write(right_bytes)

        if raw_calibration_bytes:
            with open(job_dir / "calibration.json", "wb") as f:
                f.write(raw_calibration_bytes)

        left_meta = ImageFileMeta(
            name=left_filename,
            url=f"/api/inspections/{inspection_id}/artifacts/{left_path.name}",
            size_bytes=len(left_bytes),
        )
        right_meta = ImageFileMeta(
            name=right_filename,
            url=f"/api/inspections/{inspection_id}/artifacts/{right_path.name}",
            size_bytes=len(right_bytes),
        )

        inspection = InspectionResponse.create_initial(
            inspection_id=inspection_id,
            name=name,
            surface_type=surface_type,
            reconstruction_method=reconstruction_method,
            left_meta=left_meta,
            right_meta=right_meta,
            calib_summary=calibration_summary,
        )

        with self._lock:
            self._jobs[inspection_id] = inspection
            self._save_job_state(inspection)

        # Determine execution dispatch
        mode = dispatch_mode or settings.get_execution_mode()
        if mode == "local":
            self.executor.submit(
                self._execute_pipeline_task,
                inspection_id,
                job_dir,
                left_path,
                right_path,
                calibration,
            )

        return inspection

    def _execute_pipeline_task(
        self,
        inspection_id: str,
        job_dir: Path,
        left_path: Path,
        right_path: Path,
        calibration: StereoCalibration,
    ):
        """Worker function executing the stereo pipeline in local thread-pool mode."""
        worker_name = settings.get_worker_name()

        with self._lock:
            if inspection_id not in self._jobs:
                return
            self._jobs[inspection_id].status = "processing"
            self._save_job_state(self._jobs[inspection_id])

        stage_start_times: Dict[int, float] = {}

        def on_step_change(step_idx: int, stage_id: str, status: str, msg: Optional[str] = None):
            duration = None
            now = time.time()
            if status in ("active", "processing"):
                stage_start_times[step_idx] = now
            elif status in ("completed", "failed") and step_idx in stage_start_times:
                duration = round(now - stage_start_times[step_idx], 3)

            self.apply_step_update(
                inspection_id=inspection_id,
                index=step_idx,
                step_id=stage_id,
                status=status,
                worker=worker_name,
                duration_sec=duration,
                message=msg,
            )

        try:
            result: PipelineExecutionResult = run_pipeline(
                job_dir=job_dir,
                left_image_path=left_path,
                right_image_path=right_path,
                calibration=calibration,
                on_step_change=on_step_change,
            )
        except Exception as exc:
            self.apply_failure(
                inspection_id=inspection_id,
                message=str(exc),
                worker=worker_name,
            )
            return

        if result.success:
            self.apply_result(
                inspection_id=inspection_id,
                point_count=result.point_count,
                mesh_status=result.mesh_status or "completed",
                mesh_error=result.mesh_error,
                triangle_count=result.triangle_count,
                deviation_summary=result.deviation_summary,
                findings=result.findings or [],
                classifier_status=result.classifier_status,
                classifier_error=result.classifier_error,
                duration_sec=result.duration_sec,
                worker=worker_name,
            )
        else:
            self.apply_failure(
                inspection_id=inspection_id,
                message=result.failure_message or "Pipeline execution failed",
                failed_stage_index=result.failed_stage_index,
                worker=worker_name,
            )

    def get_job(self, inspection_id: str) -> Optional[InspectionResponse]:
        with self._lock:
            return self._jobs.get(inspection_id)

    def list_jobs(self) -> List[InspectionResponse]:
        with self._lock:
            # Sort by creation time descending
            return sorted(
                list(self._jobs.values()),
                key=lambda j: j.created_at,
                reverse=True,
            )

    def get_artifact_path(self, inspection_id: str, artifact_name: str) -> Optional[Path]:
        """Safely resolves an artifact path strictly within the job directory."""
        job_dir = (self.storage_dir / inspection_id).resolve()
        if not job_dir.exists() or not job_dir.is_dir():
            return None

        # Disallow directory traversal
        target_path = (job_dir / artifact_name).resolve()
        if not str(target_path).startswith(str(job_dir)):
            return None

        if target_path.exists() and target_path.is_file():
            return target_path
        return None


# Global singleton instance
job_manager = JobManager()
