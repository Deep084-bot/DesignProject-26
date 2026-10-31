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
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..models.inspection import (
    InspectionResponse,
    ImageFileMeta,
    CalibrationSummary,
    Finding,
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
        self._monotonic_heartbeats: Dict[str, float] = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self):
        """Scans storage directory and loads prior inspections into memory."""
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

    def _validate_lease_locked(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: Optional[str] = None,
    ):
        """
        Validates ownership and fencing lease within self._lock.
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
        if worker_name and job.assigned_worker and job.assigned_worker != worker_name:
            raise LeaseConflictError(
                f"Worker identity mismatch: job '{inspection_id}' is bound to worker '{job.assigned_worker}', "
                f"but request came from '{worker_name}'."
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
            job.assigned_worker = worker_name
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
                self._validate_lease_locked(inspection_id, lease_token, worker)
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
                self._validate_lease_locked(inspection_id, lease_token, worker)
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

            self._save_job_state(job)
            return True

    def apply_failure(
        self,
        inspection_id: str,
        message: str,
        failed_stage_index: Optional[int] = None,
        worker: Optional[str] = None,
        lease_token: Optional[int] = None,
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
                self._validate_lease_locked(inspection_id, lease_token, worker)
            else:
                if job.status in ("completed", "failed"):
                    return False

            job.status = "failed"
            job.failure_message = message
            if failed_stage_index is not None and 0 <= failed_stage_index < len(job.steps):
                job.steps[failed_stage_index].status = "failed"
                if worker:
                    job.steps[failed_stage_index].worker = worker

            self._save_job_state(job)
            return True

    def commit_artifact(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        artifact_name: str,
        staging_path: Path,
    ) -> bool:
        """
        Validates fencing lease within self._lock and atomically commits
        a staged artifact file to the job directory via os.replace.
        Cleans up staging_path on validation failure.
        """
        with self._lock:
            try:
                self._validate_lease_locked(inspection_id, lease_token, worker_name)
            except Exception:
                staging_path.unlink(missing_ok=True)
                raise

            job = self._jobs[inspection_id]
            target_path = self.storage_dir / inspection_id / artifact_name
            os.replace(staging_path, target_path)

            job.last_heartbeat_at = time.time()
            self._monotonic_heartbeats[inspection_id] = time.monotonic()
            return True

    def write_outbox_intent(self, inspection_id: str, lease_token: int) -> Path:
        """Writes crash-safe dispatch intent file using atomic staging and os.replace."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        outbox_dir.mkdir(parents=True, exist_ok=True)
        final_file = outbox_dir / f"dispatch_{lease_token}.json"
        tmp_file = outbox_dir / f".dispatch_{lease_token}_{uuid.uuid4().hex}.tmp"

        payload = {
            "inspection_id": inspection_id,
            "lease_token": lease_token,
            "created_at": time.time(),
            "state": "pending",
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, final_file)
        return final_file

    def mark_outbox_published(self, inspection_id: str, lease_token: int):
        """Atomically updates dispatch record state to 'published'."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"dispatch_{lease_token}.json"
        if not final_file.exists():
            return

        tmp_file = outbox_dir / f".dispatch_{lease_token}_{uuid.uuid4().hex}.tmp"
        payload = {
            "inspection_id": inspection_id,
            "lease_token": lease_token,
            "created_at": time.time(),
            "state": "published",
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, final_file)

    def cleanup_outbox(self, inspection_id: str, lease_token: int):
        """Removes the outbox file once publication has succeeded."""
        outbox_dir = self.storage_dir / inspection_id / "outbox"
        final_file = outbox_dir / f"dispatch_{lease_token}.json"
        final_file.unlink(missing_ok=True)

    def reconcile_outbox_on_startup(self) -> int:
        """
        Scans storage for uncommitted/pending outbox records on master startup,
        re-publishing pending tasks and cleaning up completed or orphaned temp files.
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
                if item.name.startswith(".dispatch_") and item.name.endswith(".tmp"):
                    item.unlink(missing_ok=True)
                    continue

                if item.name.startswith("dispatch_") and item.name.endswith(".json"):
                    try:
                        with open(item, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        state = data.get("state")
                        insp_id = data.get("inspection_id")
                        lease = data.get("lease_token", 1)

                        if state == "published":
                            item.unlink(missing_ok=True)
                        elif state == "pending":
                            with self._lock:
                                job = self._jobs.get(insp_id)
                                should_publish = (
                                    job is not None
                                    and job.status in ("queued", "processing")
                                    and job.lease_token == lease
                                )

                            if should_publish and settings.get_execution_mode() == "celery":
                                try:
                                    from ..tasks import run_inspection_task
                                    run_inspection_task.delay(insp_id, lease)
                                    republished += 1
                                    self.mark_outbox_published(insp_id, lease)
                                    item.unlink(missing_ok=True)
                                except Exception:
                                    pass
                            else:
                                item.unlink(missing_ok=True)
                    except Exception:
                        item.unlink(missing_ok=True)
        return republished

    def reassign_orphaned_job(
        self,
        inspection_id: str,
        expected_lease_token: int,
        reason: str = "",
    ) -> Optional[int]:
        """
        Performs atomic CAS reassignment for an orphaned or dead-worker job.
        Increments lease_token by 1, clears assigned_worker, resets steps to waiting,
        writes crash-safe outbox intent, and returns the new lease_token.
        Returns None if job not found, in terminal state, or expected_lease_token mismatched.
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

            new_lease = expected_lease_token + 1
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
            self.write_outbox_intent(inspection_id, new_lease)
            return new_lease

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
