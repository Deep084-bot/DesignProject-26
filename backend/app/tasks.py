"""
Celery Worker Tasks for Distributed Pipeline Execution in SURFACE.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2.
Worker task downloads inputs from master via authenticated HTTP,
runs run_pipeline in an isolated temporary workspace,
reports stage progress, uploads output artifacts, and registers completion/failure.
Retries up to 2 times on network errors, never on pipeline failures.
Always cleans up its temporary directory.
"""

from pathlib import Path
import tempfile
import time
import logging
from typing import Optional
import cv2
import httpx

from .celery_app import celery_app
from .pipeline.calibration_parser import parse_calibration, CalibrationValidationError
from .pipeline.runner import run_pipeline, PipelineExecutionResult
from . import settings

logger = logging.getLogger(__name__)


def execute_worker_pipeline(
    inspection_id: str,
    client: Optional[httpx.Client] = None,
    worker_name: Optional[str] = None,
    master_url: Optional[str] = None,
    worker_token: Optional[str] = None,
) -> bool:
    """
    Core worker execution logic decoupled from the Celery task wrapper
    to facilitate direct testing and reuse.
    """
    master_url = master_url or settings.get_master_url()
    worker_token = worker_token or settings.get_worker_token()
    worker_name = worker_name or settings.get_worker_name()
    headers = {"X-Worker-Token": worker_token}

    close_client = False
    if client is None:
        client = httpx.Client(base_url=master_url, timeout=60.0)
        close_client = True

    try:
        # 1. Fetch manifest
        manifest_resp = client.get(f"/api/internal/jobs/{inspection_id}/manifest", headers=headers)
        manifest_resp.raise_for_status()
        manifest = manifest_resp.json()

        left_name = manifest["left_image"]
        right_name = manifest["right_image"]
        calib_name = manifest.get("calibration", "calibration.json")

        # 2. Run pipeline in isolated temporary directory
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)

            # Download input files
            left_resp = client.get(f"/api/inspections/{inspection_id}/artifacts/{left_name}")
            left_resp.raise_for_status()
            local_left = temp_dir / left_name
            local_left.write_bytes(left_resp.content)

            right_resp = client.get(f"/api/inspections/{inspection_id}/artifacts/{right_name}")
            right_resp.raise_for_status()
            local_right = temp_dir / right_name
            local_right.write_bytes(right_resp.content)

            calib_resp = client.get(f"/api/inspections/{inspection_id}/artifacts/{calib_name}")
            calib_resp.raise_for_status()
            calib_bytes = calib_resp.content
            (temp_dir / calib_name).write_bytes(calib_bytes)

            # Read image to obtain actual dimensions for calibration parsing
            left_img = cv2.imread(str(local_left))
            if left_img is None:
                fail_payload = {
                    "message": f"Worker could not decode left image '{left_name}'",
                    "worker": worker_name,
                }
                client.post(f"/api/internal/jobs/{inspection_id}/fail", headers=headers, json=fail_payload)
                return False

            hl, wl = left_img.shape[:2]
            try:
                calibration_model, _ = parse_calibration(
                    raw_content=calib_bytes,
                    file_name=calib_name,
                    image_width=wl,
                    image_height=hl,
                )
            except CalibrationValidationError as cal_err:
                fail_payload = {
                    "message": f"Calibration validation error: {str(cal_err)}",
                    "worker": worker_name,
                }
                client.post(f"/api/internal/jobs/{inspection_id}/fail", headers=headers, json=fail_payload)
                return False

            # Progress tracking callback
            stage_start_times: dict[int, float] = {}

            def on_step_change(step_idx: int, stage_id: str, status: str, msg: Optional[str] = None):
                duration = None
                now = time.time()
                if status in ("active", "processing"):
                    stage_start_times[step_idx] = now
                elif status in ("completed", "failed") and step_idx in stage_start_times:
                    duration = round(now - stage_start_times[step_idx], 3)

                try:
                    client.post(
                        f"/api/internal/jobs/{inspection_id}/step",
                        headers=headers,
                        json={
                            "index": step_idx,
                            "step_id": stage_id,
                            "status": status,
                            "worker": worker_name,
                            "duration_sec": duration,
                            "message": msg,
                        },
                    )
                except Exception as step_err:
                    logger.warning(f"Worker failed to post step update: {step_err}")

            # 3. Execute pipeline
            try:
                result: PipelineExecutionResult = run_pipeline(
                    job_dir=temp_dir,
                    left_image_path=local_left,
                    right_image_path=local_right,
                    calibration=calibration_model,
                    on_step_change=on_step_change,
                )
            except Exception as pipe_err:
                try:
                    client.post(
                        f"/api/internal/jobs/{inspection_id}/fail",
                        headers=headers,
                        json={
                            "message": f"Pipeline failure: {str(pipe_err)}",
                            "worker": worker_name,
                        },
                    )
                except Exception:
                    pass
                return False

            # 4. Handle pipeline outcome
            if result.success:
                # Upload all generated artifacts
                skip_files = {left_name, right_name, calib_name}
                for f in temp_dir.iterdir():
                    if f.is_file() and f.name not in skip_files:
                        with open(f, "rb") as fp:
                            client.post(
                                f"/api/internal/jobs/{inspection_id}/artifacts/{f.name}",
                                headers=headers,
                                files={"file": (f.name, fp, "application/octet-stream")},
                            )

                # Register final completion
                complete_payload = {
                    "point_count": result.point_count,
                    "mesh_status": result.mesh_status or "completed",
                    "mesh_error": result.mesh_error,
                    "triangle_count": result.triangle_count,
                    "deviation_summary": result.deviation_summary,
                    "findings": [f.model_dump() for f in (result.findings or [])],
                    "classifier_status": result.classifier_status,
                    "classifier_error": result.classifier_error,
                    "duration_sec": result.duration_sec,
                    "worker": worker_name,
                }
                client.post(
                    f"/api/internal/jobs/{inspection_id}/complete",
                    headers=headers,
                    json=complete_payload,
                )
                return True
            else:
                fail_payload = {
                    "message": result.failure_message or "Pipeline execution failed",
                    "failed_stage_index": result.failed_stage_index,
                    "worker": worker_name,
                }
                client.post(
                    f"/api/internal/jobs/{inspection_id}/fail",
                    headers=headers,
                    json=fail_payload,
                )
                return False

    finally:
        if close_client:
            client.close()


@celery_app.task(bind=True, max_retries=2)
def run_inspection_task(self, inspection_id: str):
    """
    Celery task entry point for distributed inspection processing.
    Retries up to 2 times on network errors communicating with master.
    """
    try:
        return execute_worker_pipeline(inspection_id=inspection_id)
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error(f"Network error in run_inspection_task({inspection_id}): {exc}. Retrying...")
        raise self.retry(exc=exc, countdown=5)
    except Exception as exc:
        logger.error(f"Non-recoverable error in run_inspection_task({inspection_id}): {exc}")
        # Report failure if possible
        try:
            headers = {"X-Worker-Token": settings.get_worker_token()}
            with httpx.Client(base_url=settings.get_master_url(), timeout=10.0) as client:
                client.post(
                    f"/api/internal/jobs/{inspection_id}/fail",
                    headers=headers,
                    json={"message": f"Task runner exception: {str(exc)}", "worker": settings.get_worker_name()},
                )
        except Exception:
            pass
        return False
