"""
Celery Worker Tasks for Distributed Pipeline Execution in SURFACE.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2.
Worker task downloads inputs from master via authenticated HTTP,
runs run_pipeline in an isolated temporary workspace,
reports periodic heartbeats and stage progress under master fencing leases,
uploads output artifacts, and registers completion/failure.
Aborts cleanly without retry on HTTP 409 (lease revoked).
Retries up to 2 times on transient network errors, never on pipeline failures.
Always cleans up its temporary directory.
"""

from pathlib import Path
import tempfile
import time
import logging
import threading
from typing import Optional
import cv2
import httpx

from .celery_app import celery_app
from .pipeline.calibration_parser import parse_calibration, CalibrationValidationError
from .pipeline.runner import run_pipeline, PipelineExecutionResult
from . import settings

logger = logging.getLogger(__name__)


class LeaseRevokedException(Exception):
    """Raised when the master informs the worker that its fencing lease is no longer valid."""
    pass


class WorkerHeartbeatThread(threading.Thread):
    """
    Background daemon thread sending periodic heartbeats to the master
    independently of OpenCV / Open3D computation.
    """
    def __init__(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        master_url: str,
        worker_token: str,
        interval: float,
        abort_event: threading.Event,
        stop_event: threading.Event,
    ):
        super().__init__(name=f"heartbeat-{inspection_id}", daemon=True)
        self.inspection_id = inspection_id
        self.lease_token = lease_token
        self.worker_name = worker_name
        self.master_url = master_url
        self.worker_token = worker_token
        self.interval = interval
        self.abort_event = abort_event
        self.stop_event = stop_event

    def run(self):
        headers = {
            "X-Worker-Token": self.worker_token,
            "X-Lease-Token": str(self.lease_token),
            "X-Worker-Name": self.worker_name,
        }
        url = f"{self.master_url}/api/internal/jobs/{self.inspection_id}/heartbeat"

        while not self.stop_event.is_set() and not self.abort_event.is_set():
            # Wait for interval in small slices to respond promptly to stop_event
            start_wait = time.monotonic()
            while (time.monotonic() - start_wait) < self.interval:
                if self.stop_event.is_set() or self.abort_event.is_set():
                    return
                time.sleep(0.5)

            if self.stop_event.is_set() or self.abort_event.is_set():
                return

            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, headers=headers)
                    if resp.status_code == 409:
                        logger.warning(
                            f"Heartbeat received 409 Conflict for job {self.inspection_id} "
                            f"(lease {self.lease_token}). Lease revoked. Signaling abort."
                        )
                        self.abort_event.set()
                        return
                    elif resp.status_code >= 400:
                        logger.warning(f"Heartbeat HTTP error {resp.status_code}: {resp.text}")
            except Exception as exc:
                logger.warning(f"Heartbeat network error for job {self.inspection_id}: {exc}")


def _check_http_response(resp: httpx.Response, inspection_id: str, lease_token: int):
    """Checks response status and raises LeaseRevokedException on 409 Conflict."""
    if resp.status_code == 409:
        raise LeaseRevokedException(
            f"Master rejected operation with 409 Conflict: lease {lease_token} for job "
            f"{inspection_id} is expired or reassigned. Detail: {resp.text}"
        )
    resp.raise_for_status()


def execute_worker_pipeline(
    inspection_id: str,
    lease_token: int = 1,
    client: Optional[httpx.Client] = None,
    worker_name: Optional[str] = None,
    master_url: Optional[str] = None,
    worker_token: Optional[str] = None,
) -> bool:
    """
    Core worker execution logic decoupled from the Celery task wrapper.
    Passes fencing lease tokens on every internal operation.
    Aborts immediately upon receiving 409 Conflict.
    """
    master_url = master_url or settings.get_master_url()
    worker_token = worker_token or settings.get_worker_token()
    worker_name = worker_name or settings.get_worker_name()
    headers = {
        "X-Worker-Token": worker_token,
        "X-Lease-Token": str(lease_token),
        "X-Worker-Name": worker_name,
    }

    close_client = False
    if client is None:
        client = httpx.Client(base_url=master_url, timeout=60.0)
        close_client = True

    abort_event = threading.Event()
    stop_heartbeat = threading.Event()
    heartbeat_thread: Optional[WorkerHeartbeatThread] = None

    try:
        # 1. Fetch and claim manifest under current lease
        manifest_resp = client.get(f"/api/internal/jobs/{inspection_id}/manifest", headers=headers)
        _check_http_response(manifest_resp, inspection_id, lease_token)
        manifest = manifest_resp.json()

        left_name = manifest["left_image"]
        right_name = manifest["right_image"]
        calib_name = manifest.get("calibration", "calibration.json")

        # 2. Start heartbeat thread once manifest is claimed
        heartbeat_thread = WorkerHeartbeatThread(
            inspection_id=inspection_id,
            lease_token=lease_token,
            worker_name=worker_name,
            master_url=master_url,
            worker_token=worker_token,
            interval=settings.get_heartbeat_interval(),
            abort_event=abort_event,
            stop_event=stop_heartbeat,
        )
        heartbeat_thread.start()

        # 3. Run pipeline in isolated temporary directory
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)

            if abort_event.is_set():
                raise LeaseRevokedException(f"Lease {lease_token} was revoked before pipeline start.")

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

            if abort_event.is_set():
                raise LeaseRevokedException(f"Lease {lease_token} was revoked during file download.")

            # Read image to obtain actual dimensions for calibration parsing
            left_img = cv2.imread(str(local_left))
            if left_img is None:
                fail_payload = {
                    "message": f"Worker could not decode left image '{left_name}'",
                    "worker": worker_name,
                }
                fail_resp = client.post(f"/api/internal/jobs/{inspection_id}/fail", headers=headers, json=fail_payload)
                _check_http_response(fail_resp, inspection_id, lease_token)
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
                fail_resp = client.post(f"/api/internal/jobs/{inspection_id}/fail", headers=headers, json=fail_payload)
                _check_http_response(fail_resp, inspection_id, lease_token)
                return False

            # Progress tracking callback
            stage_start_times: dict[int, float] = {}

            def on_step_change(step_idx: int, stage_id: str, status: str, msg: Optional[str] = None):
                if abort_event.is_set():
                    raise LeaseRevokedException(f"Lease {lease_token} was revoked during step {stage_id}.")

                duration = None
                now = time.time()
                if status in ("active", "processing"):
                    stage_start_times[step_idx] = now
                elif status in ("completed", "failed") and step_idx in stage_start_times:
                    duration = round(now - stage_start_times[step_idx], 3)

                try:
                    resp = client.post(
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
                    if resp.status_code == 409:
                        abort_event.set()
                        raise LeaseRevokedException(
                            f"Master returned 409 Conflict on step update: lease {lease_token} revoked."
                        )
                except LeaseRevokedException:
                    raise
                except Exception as step_err:
                    logger.warning(f"Worker failed to post step update: {step_err}")

            # 4. Execute pipeline
            try:
                result: PipelineExecutionResult = run_pipeline(
                    job_dir=temp_dir,
                    left_image_path=local_left,
                    right_image_path=local_right,
                    calibration=calibration_model,
                    on_step_change=on_step_change,
                )
            except LeaseRevokedException:
                raise
            except Exception as pipe_err:
                if abort_event.is_set():
                    raise LeaseRevokedException(f"Lease {lease_token} was revoked during pipeline run.")
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

            if abort_event.is_set():
                raise LeaseRevokedException(f"Lease {lease_token} was revoked prior to artifact upload.")

            # 5. Handle pipeline outcome
            if result.success:
                # Upload all generated artifacts
                skip_files = {left_name, right_name, calib_name}
                for f in sorted(temp_dir.iterdir()):
                    if f.is_file() and f.name not in skip_files:
                        if abort_event.is_set():
                            raise LeaseRevokedException(f"Lease {lease_token} was revoked during artifact upload.")
                        with open(f, "rb") as fp:
                            art_resp = client.post(
                                f"/api/internal/jobs/{inspection_id}/artifacts/{f.name}",
                                headers=headers,
                                files={"file": (f.name, fp, "application/octet-stream")},
                            )
                            _check_http_response(art_resp, inspection_id, lease_token)

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
                comp_resp = client.post(
                    f"/api/internal/jobs/{inspection_id}/complete",
                    headers=headers,
                    json=complete_payload,
                )
                _check_http_response(comp_resp, inspection_id, lease_token)
                return True
            else:
                fail_payload = {
                    "message": result.failure_message or "Pipeline execution failed",
                    "failed_stage_index": result.failed_stage_index,
                    "worker": worker_name,
                }
                fail_resp = client.post(
                    f"/api/internal/jobs/{inspection_id}/fail",
                    headers=headers,
                    json=fail_payload,
                )
                _check_http_response(fail_resp, inspection_id, lease_token)
                return False

    finally:
        stop_heartbeat.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2.0)
        if close_client:
            client.close()


@celery_app.task(bind=True, max_retries=2)
def run_inspection_task(self, inspection_id: str, lease_token: int = 1):
    """
    Celery task entry point for distributed inspection processing.
    Passes generation lease_token to the pipeline executor.
    Aborts cleanly without retries or failure reports if the lease is revoked (HTTP 409).
    Retries up to 2 times on transient network errors.
    """
    try:
        return execute_worker_pipeline(inspection_id=inspection_id, lease_token=lease_token)
    except LeaseRevokedException as rev_exc:
        logger.info(
            f"Worker task for inspection {inspection_id} cleanly aborted: {rev_exc}. "
            "No retry or failure report will be executed."
        )
        return False
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409:
            logger.info(
                f"Worker task for inspection {inspection_id} received 409 Conflict. "
                "Lease expired/reassigned. Aborting cleanly."
            )
            return False
        logger.error(f"Network error in run_inspection_task({inspection_id}, lease={lease_token}): {exc}. Retrying...")
        raise self.retry(exc=exc, countdown=5)
    except Exception as exc:
        logger.error(f"Non-recoverable error in run_inspection_task({inspection_id}, lease={lease_token}): {exc}")
        # Report failure if possible and lease is still valid
        try:
            headers = {
                "X-Worker-Token": settings.get_worker_token(),
                "X-Lease-Token": str(lease_token),
                "X-Worker-Name": settings.get_worker_name(),
            }
            with httpx.Client(base_url=settings.get_master_url(), timeout=10.0) as client:
                client.post(
                    f"/api/internal/jobs/{inspection_id}/fail",
                    headers=headers,
                    json={"message": f"Task runner exception: {str(exc)}", "worker": settings.get_worker_name()},
                )
        except Exception:
            pass
        return False
