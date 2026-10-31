"""
FastAPI Routes for Surface Stereo Inspections.

Implements:
- POST /api/inspections: Upload stereo pair & calibration, validate inputs, spawn reconstruction.
- GET /api/inspections/{inspection_id}: Fetch true inspection progress and status.
- GET /api/inspections: Fetch all inspections.
- GET /api/inspections/{inspection_id}/artifacts/{artifact_name}: Retrieve point cloud and pipeline artifacts.
"""

from typing import List, Optional
import uuid
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse

from ..models.inspection import InspectionResponse
from ..pipeline.calibration_parser import parse_calibration, CalibrationValidationError
from ..services.job_manager import job_manager
from .. import settings


router = APIRouter(prefix="/api/inspections", tags=["inspections"])


@router.post("", response_model=InspectionResponse, status_code=status.HTTP_201_CREATED)
async def create_inspection(
    left_image: UploadFile = File(..., description="Left camera stereo photograph"),
    right_image: UploadFile = File(..., description="Right camera stereo photograph"),
    calibration_file: Optional[UploadFile] = File(None, description="Camera calibration JSON"),
    name: str = Form("Structural surface inspection"),
    surface_type: str = Form("Concrete surface"),
    reconstruction_method: str = Form("Automatic surface reconstruction"),
):
    """
    Validates uploaded stereo pair and calibration parameters, then initializes
    an isolated reconstruction job.
    """
    # 1. Read image contents
    left_bytes = await left_image.read()
    right_bytes = await right_image.read()

    if not left_bytes or len(left_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Left image file is empty.",
        )
    if not right_bytes or len(right_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Right image file is empty.",
        )

    # 2. Decode and validate image readability
    left_arr = np.frombuffer(left_bytes, np.uint8)
    right_arr = np.frombuffer(right_bytes, np.uint8)

    left_cv = cv2.imdecode(left_arr, cv2.IMREAD_COLOR)
    if left_cv is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid left_image: '{left_image.filename}' could not be decoded as an image. "
                   "Supported formats: JPEG, PNG, TIFF, BMP.",
        )

    right_cv = cv2.imdecode(right_arr, cv2.IMREAD_COLOR)
    if right_cv is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid right_image: '{right_image.filename}' could not be decoded as an image. "
                   "Supported formats: JPEG, PNG, TIFF, BMP.",
        )

    # 3. Validate image dimensions match
    hl, wl = left_cv.shape[:2]
    hr, wr = right_cv.shape[:2]
    if hl != hr or wl != wr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mismatched image dimensions: Left image is {wl}×{hl} px, but right image is {wr}×{hr} px. "
                   "Stereo baseline reconstruction requires matching image dimensions.",
        )

    # 4. Validate calibration
    if calibration_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stereo reconstruction requires camera calibration. Please provide either "
                   "(A) rectified camera parameters ('baseline_distance_mm' and 'focal_length_px') or "
                   "(B) full stereo rig calibration ('left_camera_matrix', 'right_camera_matrix', 'rotation_matrix', 'translation_vector').",
        )

    calib_bytes = await calibration_file.read()
    if not calib_bytes or len(calib_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded calibration file is empty.",
        )

    try:
        calibration_model, calibration_summary = parse_calibration(
            raw_content=calib_bytes,
            file_name=calibration_file.filename or "camera_calibration.json",
            image_width=wl,
            image_height=hl,
        )
    except CalibrationValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # 5. Generate distinct inspection ID
    unique_suffix = uuid.uuid4().hex[:6].upper()
    current_year = datetime.now(timezone.utc).year
    inspection_id = f"INSP-{current_year}-{unique_suffix}"

    # 6. Create job in storage
    inspection = job_manager.create_job(
        inspection_id=inspection_id,
        name=name.strip() or "Structural surface inspection",
        surface_type=surface_type.strip() or "Concrete surface",
        reconstruction_method=reconstruction_method.strip() or "Automatic surface reconstruction",
        left_bytes=left_bytes,
        left_filename=left_image.filename or "left_view.jpg",
        right_bytes=right_bytes,
        right_filename=right_image.filename or "right_view.jpg",
        calibration=calibration_model,
        calibration_summary=calibration_summary,
        raw_calibration_bytes=calib_bytes,
    )

    # 7. In Celery mode, enqueue task to Celery / Redis using crash-safe outbox
    if settings.get_execution_mode() == "celery":
        try:
            job_manager.write_outbox_intent(inspection_id, inspection.lease_token, previous_lease=0)
            from ..tasks import run_inspection_task
            run_inspection_task.delay(inspection_id, inspection.lease_token)
            job_manager.mark_outbox_published(inspection_id, inspection.lease_token)
            job_manager.cleanup_outbox(inspection_id, inspection.lease_token)
        except Exception as exc:
            job_manager.apply_failure(
                inspection_id=inspection_id,
                message=f"Failed to enqueue Celery task: {str(exc)}",
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Task queue unavailable: {str(exc)}",
            )

    return inspection


@router.get("", response_model=List[InspectionResponse])
async def list_inspections():
    """Returns the list of all historical and current inspections."""
    return job_manager.list_jobs()


@router.get("/{inspection_id}", response_model=InspectionResponse)
async def get_inspection(inspection_id: str):
    """Returns the authentic status and artifacts of a specific inspection."""
    inspection = job_manager.get_job(inspection_id)
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection with ID '{inspection_id}' was not found.",
        )
    return inspection


@router.api_route("/{inspection_id}/artifacts/{artifact_name}", methods=["GET", "HEAD"])
async def get_artifact(inspection_id: str, artifact_name: str):
    """
    Serves a generated artifact file (PLY point cloud, disparity map, rectified image, etc.)
    with path sanitization.
    """
    artifact_path = job_manager.get_artifact_path(inspection_id, artifact_name)
    if not artifact_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact_name}' not found for inspection '{inspection_id}'.",
        )

    suffix = artifact_path.suffix.lower()
    media_types = {
        ".ply": "application/octet-stream",
        ".npy": "application/octet-stream",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=artifact_path.name,
    )
