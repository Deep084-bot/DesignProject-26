"""
Internal API Routes for Worker Coordination in SURFACE.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2.
All routes require:
- Header `X-Worker-Token` validated against SURFACE_WORKER_TOKEN via secrets.compare_digest (403 on mismatch)
- Header `X-Lease-Token` validated against current job lease (409 on mismatch)
- Header `X-Worker-Name` bound upon manifest claim and validated on every operation (409 on mismatch)
"""

from pathlib import Path
import secrets
import uuid
from typing import Optional, Dict, Any, List, Tuple
from fastapi import APIRouter, Depends, Header, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field

from ..services.job_manager import job_manager, LeaseConflictError
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


def verify_fencing_headers(
    x_lease_token: Optional[str] = Header(None, alias="X-Lease-Token"),
    x_worker_name: Optional[str] = Header(None, alias="X-Worker-Name"),
) -> Tuple[int, str]:
    """
    Extracts and validates fencing lease token and worker identity headers.
    Returns (lease_token: int, worker_name: str).
    Rejects with 409 Conflict if either header is missing or malformed.
    """
    if not x_lease_token:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Missing required 'X-Lease-Token' fencing header.",
        )
    try:
        lease = int(x_lease_token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Invalid 'X-Lease-Token' value '{x_lease_token}': must be an integer.",
        )
    if not x_worker_name or not x_worker_name.strip():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Missing required 'X-Worker-Name' header.",
        )
    return lease, x_worker_name.strip()


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


class BatchInitItem(BaseModel):
    batch_id: int
    tile_indices: List[int]
    tiles_count: Optional[int] = None


class FanoutInitRequest(BaseModel):
    total_batches: int
    batches: List[BatchInitItem]


class BatchClaimRequest(BaseModel):
    fanout_run_id: str
    claim_timeout_sec: float = 30.0


class BatchHeartbeatRequest(BaseModel):
    fanout_run_id: str
    claim_token: str
    extension_sec: float = Field(
        default=30.0,
        ge=1.0,
        le=30.0,
        description="Lease extension duration in seconds (capped at 30.0s).",
    )


class BatchCommittedItem(BaseModel):
    batch_id: int
    tile_indices: List[int]
    probabilities: List[float]


class CommittedBatchesResponse(BaseModel):
    inspection_id: str
    fanout_run_id: str
    total_batches: int
    committed_batches: int
    batches: List[BatchCommittedItem]


class BatchCommitRequest(BaseModel):
    fanout_run_id: str
    claim_token: str
    probabilities: List[float]


class FinalizerClaimRequest(BaseModel):
    fanout_run_id: str
    delegation_duration_sec: float = 30.0


class FinalizerHeartbeatRequest(BaseModel):
    delegation_token: str
    finalizer_generation: int
    extension_sec: float = 30.0


@router.get("/jobs/{inspection_id}/manifest")
async def get_job_manifest(
    inspection_id: str,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Claims job manifest and binds worker identity to the specified lease token.
    Returns file manifest (input images and calibration file) for worker download.
    Returns 409 Conflict on lease token or worker mismatch.
    """
    lease_token, worker_name = fencing

    try:
        job = job_manager.claim_manifest(inspection_id, lease_token, worker_name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
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

    return {
        "inspection_id": inspection_id,
        "lease_token": job.lease_token,
        "assigned_worker": job.assigned_worker,
        "left_image": left_name,
        "right_image": right_name,
        "calibration": calib_name,
    }


@router.post("/jobs/{inspection_id}/heartbeat")
async def post_job_heartbeat(
    inspection_id: str,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Records a periodic worker heartbeat.
    Returns 409 Conflict if the worker's lease has expired or been reassigned.
    """
    lease_token, worker_name = fencing

    try:
        job_manager.apply_heartbeat(inspection_id, lease_token, worker_name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return {
        "status": "ok",
        "inspection_id": inspection_id,
        "lease_token": lease_token,
        "worker": worker_name,
    }


def _parse_generation(x_gen: Optional[str]) -> Optional[int]:
    if x_gen is None:
        return None
    try:
        return int(x_gen)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Invalid 'X-Finalizer-Generation' value '{x_gen}': must be an integer.",
        )


@router.post("/jobs/{inspection_id}/artifacts/{artifact_name:path}")
async def upload_job_artifact(
    inspection_id: str,
    artifact_name: str,
    file: UploadFile = File(...),
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
    x_delegation_token: Optional[str] = Header(None, alias="X-Delegation-Token"),
    x_finalizer_generation: Optional[str] = Header(None, alias="X-Finalizer-Generation"),
):
    """
    Receives an artifact file from a worker and commits it safely using two-phase atomic replacement.
    Streams to a staging file without holding the manager lock, then verifies the lease and atomically
    replaces the target file inside the lock boundary.
    """
    lease_token, worker_name = fencing
    finalizer_gen = _parse_generation(x_finalizer_generation)

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

    # 1. Stream content to isolated staging file (unlocked)
    staging_path = job_dir / f".staging_{uuid.uuid4().hex}_{artifact_name}"
    total_bytes = 0
    try:
        with open(staging_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)

        # 2. Commit inside lock boundary with lease validation
        job_manager.commit_artifact(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            artifact_name=artifact_name,
            staging_path=staging_path,
            delegation_token=x_delegation_token,
            finalizer_generation=finalizer_gen,
        )
    except KeyError:
        staging_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        staging_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise

    return {
        "status": "uploaded",
        "inspection_id": inspection_id,
        "artifact": artifact_name,
        "size_bytes": total_bytes,
    }


@router.post("/jobs/{inspection_id}/step")
async def update_job_step(
    inspection_id: str,
    payload: StepUpdateRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
    x_delegation_token: Optional[str] = Header(None, alias="X-Delegation-Token"),
    x_finalizer_generation: Optional[str] = Header(None, alias="X-Finalizer-Generation"),
):
    """
    Applies a pipeline step progress update from a worker under lease validation.
    Returns 409 Conflict if lease is expired or mismatched.
    """
    lease_token, worker_name = fencing
    finalizer_gen = _parse_generation(x_finalizer_generation)

    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    try:
        applied = job_manager.apply_step_update(
            inspection_id=inspection_id,
            index=payload.index,
            step_id=payload.step_id,
            status=payload.status,
            worker=payload.worker or worker_name,
            duration_sec=payload.duration_sec,
            message=payload.message,
            lease_token=lease_token,
            delegation_token=x_delegation_token,
            finalizer_generation=finalizer_gen,
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return {"status": "ok" if applied else "ignored"}


@router.post("/jobs/{inspection_id}/complete")
async def complete_job(
    inspection_id: str,
    payload: CompleteJobRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
    x_delegation_token: Optional[str] = Header(None, alias="X-Delegation-Token"),
    x_finalizer_generation: Optional[str] = Header(None, alias="X-Finalizer-Generation"),
):
    """
    Applies final successful results from a worker under lease validation.
    Returns 409 Conflict if lease is expired or mismatched.
    """
    lease_token, worker_name = fencing
    finalizer_gen = _parse_generation(x_finalizer_generation)

    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    try:
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
            worker=payload.worker or worker_name,
            lease_token=lease_token,
            delegation_token=x_delegation_token,
            finalizer_generation=finalizer_gen,
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return {"status": "ok" if applied else "ignored"}


@router.post("/jobs/{inspection_id}/fail")
async def fail_job(
    inspection_id: str,
    payload: FailJobRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
    x_delegation_token: Optional[str] = Header(None, alias="X-Delegation-Token"),
    x_finalizer_generation: Optional[str] = Header(None, alias="X-Finalizer-Generation"),
):
    """
    Applies failure status and error details from a worker under lease validation.
    Returns 409 Conflict if lease is expired or mismatched.
    """
    lease_token, worker_name = fencing
    finalizer_gen = _parse_generation(x_finalizer_generation)

    job = job_manager.get_job(inspection_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    try:
        applied = job_manager.apply_failure(
            inspection_id=inspection_id,
            message=payload.message,
            failed_stage_index=payload.failed_stage_index,
            worker=payload.worker or worker_name,
            lease_token=lease_token,
            delegation_token=x_delegation_token,
            finalizer_generation=finalizer_gen,
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return {"status": "ok" if applied else "ignored"}


@router.post("/jobs/{inspection_id}/fanout/init")
async def init_fanout_run(
    inspection_id: str,
    payload: FanoutInitRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Initializes a fan-out run for Stage 4 under coordinator lease validation.
    """
    lease_token, worker_name = fencing
    try:
        fanout = job_manager.init_fanout(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            total_batches=payload.total_batches,
            batches=payload.batches,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return fanout


@router.post("/jobs/{inspection_id}/batches/{batch_id}/claim")
async def claim_inspection_batch(
    inspection_id: str,
    batch_id: int,
    payload: BatchClaimRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Claims a batch for processing by an authenticated cluster worker.
    """
    lease_token, worker_name = fencing
    try:
        result = job_manager.claim_batch(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            fanout_run_id=payload.fanout_run_id,
            batch_id=batch_id,
            claim_timeout_sec=payload.claim_timeout_sec,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' or batch '{batch_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return result


@router.post("/jobs/{inspection_id}/batches/{batch_id}/commit")
async def commit_inspection_batch(
    inspection_id: str,
    batch_id: int,
    payload: BatchCommitRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Durably commits batch probabilities under lock and atomic persistence.
    """
    lease_token, worker_name = fencing
    try:
        result = job_manager.commit_batch(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            fanout_run_id=payload.fanout_run_id,
            batch_id=batch_id,
            claim_token=payload.claim_token,
            probabilities=payload.probabilities,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' or batch '{batch_id}' not found.",
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return result


@router.post("/jobs/{inspection_id}/batches/{batch_id}/heartbeat")
async def post_batch_heartbeat(
    inspection_id: str,
    batch_id: int,
    payload: BatchHeartbeatRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Renews an active batch claim window for an authenticated worker.
    """
    lease_token, worker_name = fencing
    try:
        result = job_manager.apply_batch_heartbeat(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            fanout_run_id=payload.fanout_run_id,
            batch_id=batch_id,
            claim_token=payload.claim_token,
            extension_sec=payload.extension_sec,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' or batch '{batch_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return result


@router.get("/jobs/{inspection_id}/batches/committed", response_model=CommittedBatchesResponse)
async def get_committed_batches(
    inspection_id: str,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
    x_delegation_token: Optional[str] = Header(None, alias="X-Delegation-Token"),
    x_finalizer_generation: Optional[str] = Header(None, alias="X-Finalizer-Generation"),
):
    """
    Retrieves authoritative committed batch results for delegated finalizer.
    """
    lease_token, worker_name = fencing
    finalizer_gen = _parse_generation(x_finalizer_generation)
    if not x_delegation_token or finalizer_gen is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Committed batches retrieval requires 'X-Delegation-Token' and 'X-Finalizer-Generation'.",
        )
    try:
        result = job_manager.get_committed_batches_locked(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            delegation_token=x_delegation_token,
            finalizer_generation=finalizer_gen,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve committed batches: {err}",
        )

    return result


@router.post("/jobs/{inspection_id}/finalizer/claim")
async def claim_finalizer_delegation(
    inspection_id: str,
    payload: FinalizerClaimRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Claims finalizer delegation for post-fanout Stage 5 completion.
    """
    lease_token, worker_name = fencing
    try:
        result = job_manager.claim_finalizer(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            fanout_run_id=payload.fanout_run_id,
            delegation_duration_sec=payload.delegation_duration_sec,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return result


@router.post("/jobs/{inspection_id}/finalizer/heartbeat")
async def post_finalizer_heartbeat(
    inspection_id: str,
    payload: FinalizerHeartbeatRequest,
    fencing: Tuple[int, str] = Depends(verify_fencing_headers),
):
    """
    Renews finalizer delegation lease window.
    """
    lease_token, worker_name = fencing
    try:
        result = job_manager.apply_finalizer_heartbeat(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            delegation_token=payload.delegation_token,
            finalizer_generation=payload.finalizer_generation,
            extension_sec=payload.extension_sec,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )
    except LeaseConflictError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        )

    return result


@router.get("/jobs/{inspection_id}/fanout/status")
async def get_inspection_fanout_status(inspection_id: str):
    """
    Returns current fan-out state for an inspection.
    """
    fanout = job_manager.get_fanout_status(inspection_id)
    if not fanout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No fan-out state found for inspection '{inspection_id}'.",
        )
    return fanout
