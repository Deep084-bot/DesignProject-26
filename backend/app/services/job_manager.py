"""
Job Manager and Background Execution Service for Surface.

Manages inspection job lifecycles:
- Isolated filesystem workspace per job (backend/storage/jobs/{inspection_id})
- Safe file naming (avoids arbitrary user paths)
- Asynchronous pipeline execution using Python ThreadPoolExecutor
- Thread-safe inspection state tracking and persistent JSON synchronization
- Clean modular design ready for eventual migration to Celery/Redis
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil

from ..models.inspection import (
    InspectionResponse,
    ImageFileMeta,
    CalibrationSummary,
    DEFAULT_STAGES,
    PipelineStep,
)
from ..models.calibration import StereoCalibration
from ..pipeline.runner import run_pipeline, PipelineExecutionResult


# Base storage directory
BASE_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "jobs"


class JobManager:
    def __init__(self, storage_dir: Path = BASE_STORAGE_DIR, max_workers: int = 2):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="surface-worker")
        self._lock = threading.Lock()
        self._jobs: Dict[str, InspectionResponse] = {}
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
        """Persists current inspection model to its job directory."""
        job_dir = self.storage_dir / inspection.id
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_file = job_dir / "inspection.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(inspection.model_dump(), f, indent=2)

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
    ) -> InspectionResponse:
        """
        Initializes a job directory, writes sanitized input files, and schedules background processing.
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

        # Dispatch background worker
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
        """Worker function executing the stereo pipeline."""
        with self._lock:
            if inspection_id not in self._jobs:
                return
            self._jobs[inspection_id].status = "processing"
            self._save_job_state(self._jobs[inspection_id])

        def on_step_change(step_idx: int, stage_id: str, status: str, msg: Optional[str] = None):
            with self._lock:
                if inspection_id not in self._jobs:
                    return
                job = self._jobs[inspection_id]
                job.current_step_index = step_idx
                if step_idx < len(job.steps):
                    job.steps[step_idx].status = status  # type: ignore
                if status == "failed" and msg:
                    job.failure_message = msg
                self._save_job_state(job)

        # Run pipeline synchronously in worker thread
        result: PipelineExecutionResult = run_pipeline(
            job_dir=job_dir,
            left_image_path=left_path,
            right_image_path=right_path,
            calibration=calibration,
            on_step_change=on_step_change,
        )

        with self._lock:
            if inspection_id not in self._jobs:
                return
            job = self._jobs[inspection_id]
            job.processing_duration_sec = result.duration_sec

            if result.success:
                job.status = "completed"
                job.current_step_index = 5
                job.point_count = result.point_count
                job.point_cloud_url = f"/api/inspections/{inspection_id}/artifacts/pointcloud.ply"
                job.mesh_status = result.mesh_status
                job.mesh_error = result.mesh_error
                if result.mesh_path and result.mesh_path.exists():
                    job.mesh_url = f"/api/inspections/{inspection_id}/artifacts/mesh.ply"
                    job.triangle_count = result.triangle_count
                else:
                    job.mesh_url = None
                    job.triangle_count = None
                job.failure_message = None
                # Milestone 1: findings list remains empty [] (no fabricated defects)
                job.findings = []
                for s in job.steps:
                    s.status = "completed"
            else:
                job.status = "failed"
                job.failure_message = result.failure_message
                if result.failed_stage_index is not None and result.failed_stage_index < len(job.steps):
                    job.steps[result.failed_stage_index].status = "failed"

            self._save_job_state(job)

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
