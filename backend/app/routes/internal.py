"""
Internal API Routes for Worker Coordination in SURFACE.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2.
All routes require header `X-Worker-Token` validated against SURFACE_WORKER_TOKEN
via constant-time comparison (secrets.compare_digest).
If SURFACE_WORKER_TOKEN is empty, all routes return 403 Forbidden.
"""

from pathlib import Path
import secrets
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Header, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field

from ..services.job_manager import job_manager
from .. import settings


def verify_worker_token(x_worker_token: Optional[str] = Header(None, alias="X-Worker-Token")):
    """
    Enforces secret token verification for internal worker communication.
    Rejects all requests with 403 if token is not configured or fails verification.
    """
    configured_token = settings.get_worker_token()
    if not configured_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="SURFACE_WORKER_TOKEN is not configured on master.",
        )
    if not x_worker_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required 'X-Worker-Token' header.",
        )
    if not secrets.compare_digest(x_worker_token, configured_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid worker token provided.",
        )


router = APIRouter(
    prefix="/api/internal",
    tags=["internal"],
    dependencies=[Depends(verify_worker_token)],
)


class StepUpdateRequest(BaseModel):
    index: int
    step_id: str
    status: str
    worker: Optional[str] = None
    duration_sec: Optional[float] = None
    message: Optional[str] = None


class CompleteJobRequest(BaseModel):
    point_count: int
    mesh_status: str = "completed"
    mesh_error: Optional[str] = None
    triangle_count: Optional[int] = None
    deviation_summary: Optional[Dict[str, Any]] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    classifier_status: Optional[str] = None
    classifier_error: Optional[str] = None
    duration_sec: float = 0.0
    worker: Optional[str] = None


class FailJobRequest(BaseModel):
    message: str
    failed_stage_index: Optional[int] = None
    worker: Optional[str] = None


@router.get("/jobs/{inspection_id}/manifest")
async def get_job_manifest(inspection_id: str):
    """
    Returns file manifest (input images and calibration file) for worker download.
    """
    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    job_dir = (job_manager.storage_dir / inspection_id).resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storage for inspection '{inspection_id}' not found.",
        )

    left_name = None
    right_name = None
    for item in job_dir.iterdir():
        if item.is_file():
            if item.name.startswith("input_left"):
                left_name = item.name
            elif item.name.startswith("input_right"):
                right_name = item.name

    if not left_name or not right_name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Input images not found in storage for inspection '{inspection_id}'.",
        )

    calib_name = "calibration.json"
    if not (job_dir / calib_name).exists():
        # Fall back if no calibration.json exists
        calib_name = "calibration.json"

    return {
        "inspection_id": inspection_id,
        "left_image": left_name,
        "right_image": right_name,
        "calibration": calib_name,
    }


@router.post("/jobs/{inspection_id}/artifacts/{artifact_name:path}")
async def upload_job_artifact(
    inspection_id: str,
    artifact_name: str,
    file: UploadFile = File(...),
):
    """
    Receives an artifact file from a worker and stores it safely within the job directory.
    Validates against directory traversal and rejects unsafe filenames.
    """
    if (
        not artifact_name
        or "/" in artifact_name
        or "\\" in artifact_name
        or ".." in artifact_name
        or artifact_name.startswith(".")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid artifact name '{artifact_name}': path traversal and hidden files are disallowed.",
        )

    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    job_dir = (job_manager.storage_dir / inspection_id).resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storage for inspection '{inspection_id}' not found.",
        )

    target_path = (job_dir / artifact_name).resolve()
    if not str(target_path).startswith(str(job_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal detected.",
        )

    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)

    return {
        "status": "uploaded",
        "inspection_id": inspection_id,
        "artifact": artifact_name,
        "size_bytes": len(content),
    }


@router.post("/jobs/{inspection_id}/step")
async def update_job_step(inspection_id: str, payload: StepUpdateRequest):
    """
    Applies a pipeline step progress update from a worker.
    Ignores updates if job is already finished (completed or failed).
    """
    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    applied = job_manager.apply_step_update(
        inspection_id=inspection_id,
        index=payload.index,
        step_id=payload.step_id,
        status=payload.status,
        worker=payload.worker,
        duration_sec=payload.duration_sec,
        message=payload.message,
    )

    return {"status": "ok" if applied else "ignored"}


@router.post("/jobs/{inspection_id}/complete")
async def complete_job(inspection_id: str, payload: CompleteJobRequest):
    """
    Applies final successful results from a worker.
    Ignores duplicate completion if job is already finished.
    """
    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    applied = job_manager.apply_result(
        inspection_id=inspection_id,
        point_count=payload.point_count,
        mesh_status=payload.mesh_status,
        mesh_error=payload.mesh_error,
        triangle_count=payload.triangle_count,
        deviation_summary=payload.deviation_summary,
        findings=payload.findings,
        classifier_status=payload.classifier_status,
        classifier_error=payload.classifier_error,
        duration_sec=payload.duration_sec,
        worker=payload.worker,
    )

    return {"status": "ok" if applied else "ignored"}


@router.post("/jobs/{inspection_id}/fail")
async def fail_job(inspection_id: str, payload: FailJobRequest):
    """
    Applies failure status and error details from a worker.
    Ignores update if job is already finished.
    """
    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    applied = job_manager.apply_failure(
        inspection_id=inspection_id,
        message=payload.message,
        failed_stage_index=payload.failed_stage_index,
        worker=payload.worker,
    )

    return {"status": "ok" if applied else "ignored"}
