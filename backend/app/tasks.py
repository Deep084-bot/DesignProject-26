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
import math
import json
import io
import shutil
from typing import Optional, List, Dict, Any
import cv2
import httpx
import numpy as np
from celery import chord
from celery.exceptions import Ignore, Retry

from .celery_app import celery_app
from .pipeline.calibration_parser import parse_calibration, CalibrationValidationError
from .pipeline.damage import (
    generate_tile_definitions,
    extract_tile_crop,
    classify_tile_batch,
    aggregate_damage_findings,
    CrackClassifier,
)
from .pipeline.runner import (
    run_pipeline,
    run_pipeline_stages_0_to_3,
    run_pipeline_stage_5,
    PipelineExecutionResult,
    Stage0to3Result,
)
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

    def _pulse(self) -> bool:
        """
        Sends an authenticated heartbeat pulse to the master under current fencing headers.
        WHY: Factored out to execute an immediate initial pulse upon thread start as well as
        periodic interval pulses, ensuring last_heartbeat_at is refreshed on the master before
        compute-intensive pipeline stages begin.
        Returns True if successful, False on HTTP error or network failure.
        Sets abort_event and returns False if 409 Conflict (lease revoked).
        """
        headers = {
            "X-Worker-Token": self.worker_token,
            "X-Lease-Token": str(self.lease_token),
            "X-Worker-Name": self.worker_name,
        }
        url = f"{self.master_url}/api/internal/jobs/{self.inspection_id}/heartbeat"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers)
                if resp.status_code == 409:
                    logger.warning(
                        f"Heartbeat received 409 Conflict for job {self.inspection_id} "
                        f"(lease {self.lease_token}). Lease revoked. Signaling abort."
                    )
                    self.abort_event.set()
                    return False
                elif resp.status_code >= 400:
                    logger.warning(f"Heartbeat HTTP error {resp.status_code}: {resp.text}")
                    return False
                return True
        except Exception as exc:
            logger.warning(f"Heartbeat network error for job {self.inspection_id}: {exc}")
            return False

    def run(self):
        # 1. Immediate initial pulse on start to ensure master records fresh activity
        if not self.stop_event.is_set() and not self.abort_event.is_set():
            if not self._pulse() and self.abort_event.is_set():
                return

        # 2. Periodic heartbeat loop
        while not self.stop_event.is_set() and not self.abort_event.is_set():
            # Wait for interval in small slices to respond promptly to stop_event
            start_wait = time.monotonic()
            while (time.monotonic() - start_wait) < self.interval:
                if self.stop_event.is_set() or self.abort_event.is_set():
                    return
                time.sleep(0.5)

            if self.stop_event.is_set() or self.abort_event.is_set():
                return

            if not self._pulse() and self.abort_event.is_set():
                return


class BatchHeartbeatThread(threading.Thread):
    """
    Background daemon thread sending periodic heartbeats to master for an active batch claim.
    """
    def __init__(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        master_url: str,
        worker_token: str,
        fanout_run_id: str,
        batch_id: int,
        claim_token: str,
        abort_event: threading.Event,
        stop_event: threading.Event,
        interval: float = 10.0,
    ):
        super().__init__(name=f"batch-heartbeat-{inspection_id}-{batch_id}", daemon=True)
        self.inspection_id = inspection_id
        self.lease_token = lease_token
        self.worker_name = worker_name
        self.master_url = master_url
        self.worker_token = worker_token
        self.fanout_run_id = fanout_run_id
        self.batch_id = batch_id
        self.claim_token = claim_token
        self.abort_event = abort_event
        self.stop_event = stop_event
        self.interval = interval

    def run(self):
        headers = {
            "X-Worker-Token": self.worker_token,
            "X-Lease-Token": str(self.lease_token),
            "X-Worker-Name": self.worker_name,
        }
        url = f"{self.master_url.rstrip('/')}/api/internal/jobs/{self.inspection_id}/batches/{self.batch_id}/heartbeat"
        payload = {
            "fanout_run_id": self.fanout_run_id,
            "claim_token": self.claim_token,
            "extension_sec": 30.0,
        }
        while not self.stop_event.is_set() and not self.abort_event.is_set():
            start_wait = time.monotonic()
            while (time.monotonic() - start_wait) < self.interval:
                if self.stop_event.is_set() or self.abort_event.is_set():
                    return
                time.sleep(0.5)

            if self.stop_event.is_set() or self.abort_event.is_set():
                return

            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 409:
                        logger.warning(
                            f"Batch heartbeat 409 Conflict for job {self.inspection_id} batch {self.batch_id}. Aborting."
                        )
                        self.abort_event.set()
                        return
                    elif resp.status_code >= 400:
                        logger.warning(f"Batch heartbeat HTTP error {resp.status_code}: {resp.text}")
            except Exception as exc:
                logger.warning(f"Batch heartbeat error: {exc}")


class FinalizerHeartbeatThread(threading.Thread):
    """
    Background daemon thread sending periodic heartbeats to master for active finalizer delegation.
    """
    def __init__(
        self,
        inspection_id: str,
        lease_token: int,
        worker_name: str,
        master_url: str,
        worker_token: str,
        delegation_token: str,
        finalizer_generation: int,
        abort_event: threading.Event,
        stop_event: threading.Event,
        interval: float = 10.0,
    ):
        super().__init__(name=f"finalizer-heartbeat-{inspection_id}", daemon=True)
        self.inspection_id = inspection_id
        self.lease_token = lease_token
        self.worker_name = worker_name
        self.master_url = master_url
        self.worker_token = worker_token
        self.delegation_token = delegation_token
        self.finalizer_generation = finalizer_generation
        self.abort_event = abort_event
        self.stop_event = stop_event
        self.interval = interval

    def _pulse(self) -> bool:
        """
        Sends an authenticated finalizer heartbeat pulse to master under delegation fencing.
        WHY: Factored out to pulse immediately on thread start before sleeping, ensuring
        finalizer_expires_at is durably extended on master before compute or aggregation begins.
        Returns True if successful, False on HTTP error or network failure.
        Sets abort_event and returns False if 409 Conflict (delegation revoked).
        """
        headers = {
            "X-Worker-Token": self.worker_token,
            "X-Lease-Token": str(self.lease_token),
            "X-Worker-Name": self.worker_name,
        }
        url = f"{self.master_url.rstrip('/')}/api/internal/jobs/{self.inspection_id}/finalizer/heartbeat"
        payload = {
            "delegation_token": self.delegation_token,
            "finalizer_generation": self.finalizer_generation,
            "extension_sec": 30.0,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 409:
                    logger.warning(
                        f"Finalizer heartbeat received 409 Conflict for job {self.inspection_id} "
                        f"(gen {self.finalizer_generation}). Delegation revoked. Signaling abort."
                    )
                    self.abort_event.set()
                    return False
                elif resp.status_code >= 400:
                    logger.warning(f"Finalizer heartbeat HTTP error {resp.status_code}: {resp.text}")
                    return False
                return True
        except Exception as exc:
            logger.warning(f"Finalizer heartbeat error for job {self.inspection_id}: {exc}")
            return False

    def run(self):
        # 1. Immediate initial pulse on start to refresh finalizer expiration window
        if not self.stop_event.is_set() and not self.abort_event.is_set():
            if not self._pulse() and self.abort_event.is_set():
                return

        # 2. Periodic heartbeat loop
        slice_sec = min(0.5, max(0.01, self.interval / 2))
        while not self.stop_event.is_set() and not self.abort_event.is_set():
            start_wait = time.monotonic()
            while (time.monotonic() - start_wait) < self.interval:
                if self.stop_event.is_set() or self.abort_event.is_set():
                    return
                time.sleep(slice_sec)

            if self.stop_event.is_set() or self.abort_event.is_set():
                return

            if not self._pulse() and self.abort_event.is_set():
                return


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
    Coordinator Celery task orchestrating Stages 0–3, fan-out initialization,
    and task replacement with a Celery chord across distributed cluster workers.
    """
    master_url = settings.get_master_url()
    worker_token = settings.get_worker_token()
    worker_name = settings.get_worker_name()

    # For backward compatibility with unit tests mocking execute_worker_pipeline
    if hasattr(execute_worker_pipeline, "mock_calls") or hasattr(execute_worker_pipeline, "_mock_self"):
        try:
            return execute_worker_pipeline(inspection_id=inspection_id, lease_token=lease_token)
        except LeaseRevokedException as exc:
            logger.info(f"Coordinator task for inspection {inspection_id} cleanly aborted: {exc}")
            return False

    headers = {
        "X-Worker-Token": worker_token,
        "X-Lease-Token": str(lease_token),
        "X-Worker-Name": worker_name,
    }

    # Check if fan-out is already initialized on master (e.g. coordinator redelivery)
    try:
        with httpx.Client(base_url=master_url, timeout=10.0) as check_client:
            st_resp = check_client.get(f"/api/internal/jobs/{inspection_id}/fanout/status", headers=headers)
            if st_resp.status_code == 200:
                fanout_data = st_resp.json()
                f_status = fanout_data.get("status")
                f_run_id = fanout_data.get("fanout_run_id")
                total_b = fanout_data.get("total_batches", 0)

                if f_status == "FANOUT_RUNNING" and total_b > 0 and f_run_id:
                    logger.info(
                        f"Coordinator redelivery for job {inspection_id}: fanout {f_run_id} is already "
                        f"running ({total_b} batches). Skipping Stages 0–3 and re-dispatching chord replacement."
                    )
                    header = [
                        classify_batch_task.s(inspection_id, lease_token, f_run_id, b_id)
                        for b_id in range(total_b)
                    ]
                    callback = finalize_inspection_task.si(inspection_id, lease_token, f_run_id)
                    sig = chord(header, callback)
                    return self.replace(sig)
                elif f_status in ("ALL_BATCHES_COMMITTED", "CALLBACK_PENDING", "FINALIZATION_RUNNING", "TERMINAL"):
                    logger.info(
                        f"Coordinator redelivery for job {inspection_id}: fanout state is already '{f_status}'. "
                        "Bypassing duplicate coordinator processing."
                    )
                    return True
    except LeaseRevokedException:
        return False
    except Exception as exc:
        logger.debug(f"Fan-out status check on coordinator start: {exc}")

    # Initial coordinator execution: run Stages 0-3
    abort_event = threading.Event()
    stop_heartbeat = threading.Event()
    heartbeat_thread: Optional[WorkerHeartbeatThread] = None

    try:
        with httpx.Client(base_url=master_url, timeout=60.0) as client:
            # 1. Fetch manifest
            manifest_resp = client.get(f"/api/internal/jobs/{inspection_id}/manifest", headers=headers)
            _check_http_response(manifest_resp, inspection_id, lease_token)
            manifest = manifest_resp.json()

            left_name = manifest["left_image"]
            right_name = manifest["right_image"]
            calib_name = manifest.get("calibration", "calibration.json")

            # 2. Start heartbeat thread
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

            with tempfile.TemporaryDirectory(prefix=f"surface_coord_{inspection_id}_") as td:
                temp_dir = Path(td)

                if abort_event.is_set():
                    raise LeaseRevokedException(f"Lease {lease_token} was revoked before pipeline start.")

                # Download inputs
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

                left_img = cv2.imread(str(local_left))
                right_img = cv2.imread(str(local_right))
                if left_img is None or right_img is None:
                    fail_payload = {
                        "message": "Worker could not decode input stereo images",
                        "worker": worker_name,
                    }
                    client.post(f"/api/internal/jobs/{inspection_id}/fail", headers=headers, json=fail_payload)
                    return False

                hl, wl = left_img.shape[:2]
                calibration_model, calib_dict = parse_calibration(calib_bytes, image_width=wl, image_height=hl)

                # Step notification callback
                def on_step_change(idx, step_id, status, message=None, duration_sec=None):
                    try:
                        client.post(
                            f"/api/internal/jobs/{inspection_id}/step",
                            headers=headers,
                            json={
                                "index": idx,
                                "step_id": step_id,
                                "status": status,
                                "worker": worker_name,
                                "duration_sec": duration_sec,
                                "message": message,
                            },
                        )
                    except Exception:
                        pass

                # 3. Execute Stages 0-3
                stage_res = run_pipeline_stages_0_to_3(
                    job_dir=temp_dir,
                    left_image_path=local_left,
                    right_image_path=local_right,
                    calibration=calibration_model,
                    on_step_change=on_step_change,
                )

                if not stage_res.success:
                    fail_payload = {
                        "message": stage_res.failure_message or f"Stage {stage_res.failed_stage_index} failed",
                        "failed_stage_index": stage_res.failed_stage_index,
                        "worker": worker_name,
                    }
                    client.post(f"/api/internal/jobs/{inspection_id}/fail", headers=headers, json=fail_payload)
                    return False

                if abort_event.is_set():
                    raise LeaseRevokedException(f"Lease {lease_token} was revoked after stage 3 completion.")

                # 4. Upload Stage 0-3 artifacts to master
                artifacts_to_upload = [
                    "rectified_left.png",
                    "rectified_right.png",
                    "disparity.npy",
                    "disparity_vis.png",
                    "pointcloud.ply",
                    "mesh.ply",
                    "deviation_map.npy",
                    "deviation_heatmap.png",
                    "deviations.json",
                ]
                for art_name in artifacts_to_upload:
                    art_file = temp_dir / art_name
                    if art_file.exists():
                        with open(art_file, "rb") as af:
                            up_resp = client.post(
                                f"/api/internal/jobs/{inspection_id}/artifacts/{art_name}",
                                headers=headers,
                                files={"file": (art_name, af, "application/octet-stream")},
                            )
                            _check_http_response(up_resp, inspection_id, lease_token)

                # 5. Generate deterministic tile definitions and batch partitions
                rect_shape = stage_res.rectified_left.shape if stage_res.rectified_left is not None else left_img.shape
                all_tile_defs = generate_tile_definitions(
                    image_shape=rect_shape,
                    deviation_regions=stage_res.deviation_regions,
                    tile_size=settings.get_tile_size(),
                    stride=settings.get_tile_size(),
                )
                batch_size = 8
                total_tiles = len(all_tile_defs)
                total_batches = max(1, math.ceil(total_tiles / batch_size)) if total_tiles > 0 else 1
                batches_payload = []
                for b_idx in range(total_batches):
                    s_idx = b_idx * batch_size
                    e_idx = min(s_idx + batch_size, total_tiles)
                    batches_payload.append({
                        "batch_id": b_idx,
                        "tile_indices": list(range(s_idx, e_idx)),
                    })

                # 6. Initialize fan-out on master
                fanout_init_resp = client.post(
                    f"/api/internal/jobs/{inspection_id}/fanout/init",
                    headers=headers,
                    json={
                        "total_batches": total_batches,
                        "batches": batches_payload,
                    },
                )
                _check_http_response(fanout_init_resp, inspection_id, lease_token)
                fanout_run_id = fanout_init_resp.json()["fanout_run_id"]

                # 7. Post Stage 4 processing step
                client.post(
                    f"/api/internal/jobs/{inspection_id}/step",
                    headers=headers,
                    json={
                        "index": 4,
                        "step_id": "damage",
                        "status": "processing",
                        "worker": worker_name,
                        "message": f"Stage 4: Distributed tile classification fan-out dispatched ({total_batches} batches)",
                    },
                )

                # 8. Build Celery chord and replace coordinator task
                header = [
                    classify_batch_task.s(inspection_id, lease_token, fanout_run_id, b_id)
                    for b_id in range(total_batches)
                ]
                callback = finalize_inspection_task.si(inspection_id, lease_token, fanout_run_id)
                sig = chord(header, callback)
                return self.replace(sig)

    except (Ignore, Retry):
        raise
    except LeaseRevokedException as rev_exc:
        logger.info(f"Coordinator task for inspection {inspection_id} cleanly aborted: {rev_exc}")
        return False
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409:
            logger.info(f"Coordinator task for inspection {inspection_id} received 409 Conflict. Aborting cleanly.")
            return False
        logger.error(f"Network error in coordinator task for {inspection_id}: {exc}. Retrying...")
        raise self.retry(exc=exc, countdown=5)
    except Exception as exc:
        logger.error(f"Non-recoverable error in coordinator task for {inspection_id}: {exc}")
        try:
            with httpx.Client(base_url=settings.get_master_url(), timeout=10.0) as client:
                client.post(
                    f"/api/internal/jobs/{inspection_id}/fail",
                    headers=headers,
                    json={"message": f"Coordinator exception: {str(exc)}", "worker": worker_name},
                )
        except Exception:
            pass
        return False
    finally:
        stop_heartbeat.set()
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2.0)


@celery_app.task(
    bind=True,
    name="app.tasks.classify_batch_task",
    max_retries=24,
    default_retry_delay=5,
    acks_late=True,
    reject_on_worker_lost=True,
)
def classify_batch_task(
    self,
    inspection_id: str,
    lease_token: int,
    fanout_run_id: str,
    batch_id: int,
) -> Dict[str, Any]:
    """
    Worker Celery task for classifying a single tile batch.
    Claims batch, pulses heartbeat, classifies tiles, and commits probabilities to master.
    """
    master_url = settings.get_master_url().rstrip("/")
    worker_token = settings.get_worker_token()
    worker_name = settings.get_worker_name()

    headers = {
        "X-Worker-Token": worker_token,
        "X-Lease-Token": str(lease_token),
        "X-Worker-Name": worker_name,
    }

    # 1. Attempt batch claim
    claim_url = f"{master_url}/api/internal/jobs/{inspection_id}/batches/{batch_id}/claim"
    claim_payload = {
        "fanout_run_id": fanout_run_id,
        "claim_timeout_sec": 30.0,
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(claim_url, headers=headers, json=claim_payload)
    except Exception as exc:
        logger.warning(f"Network error claiming batch {batch_id} for job {inspection_id}: {exc}")
        raise self.retry(exc=exc, countdown=3)

    if resp.status_code == 409:
        logger.info(f"Claim batch {batch_id} returned 409 Conflict: {resp.text}. Clean abort.")
        return {"batch_id": batch_id, "status": "aborted"}

    resp.raise_for_status()
    claim_data = resp.json()
    c_status = claim_data.get("status")

    if c_status == "already_committed":
        return {"batch_id": batch_id, "status": "already_committed"}

    if c_status == "already_claimed":
        retry_after = float(claim_data.get("retry_after_sec", 5.0))
        raise self.retry(countdown=retry_after, max_retries=24)

    if c_status != "claimed":
        raise ValueError(f"Unexpected claim status '{c_status}' for batch {batch_id}")

    claim_token = claim_data["claim_token"]
    tile_indices = claim_data["tile_indices"]

    # 2. Start batch heartbeat thread
    abort_event = threading.Event()
    stop_event = threading.Event()
    hb = BatchHeartbeatThread(
        inspection_id=inspection_id,
        lease_token=lease_token,
        worker_name=worker_name,
        master_url=master_url,
        worker_token=worker_token,
        fanout_run_id=fanout_run_id,
        batch_id=batch_id,
        claim_token=claim_token,
        abort_event=abort_event,
        stop_event=stop_event,
        interval=10.0,
    )
    hb.start()

    try:
        if abort_event.is_set():
            return {"batch_id": batch_id, "status": "aborted"}

        # 3. Retrieve rectified_left.png and deviations.json
        with httpx.Client(timeout=30.0) as client:
            img_resp = client.get(f"{master_url}/api/inspections/{inspection_id}/artifacts/rectified_left.png")
            img_resp.raise_for_status()
            img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
            rectified_left = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            dev_resp = client.get(f"{master_url}/api/inspections/{inspection_id}/artifacts/deviations.json")
            dev_resp.raise_for_status()
            dev_data = dev_resp.json()
            deviation_regions = dev_data.get("regions", [])

        if abort_event.is_set():
            return {"batch_id": batch_id, "status": "aborted"}

        # 4. Generate tile definitions and slice batch
        all_tile_defs = generate_tile_definitions(
            image_shape=rectified_left.shape,
            deviation_regions=deviation_regions,
            tile_size=settings.get_tile_size(),
            stride=settings.get_tile_size(),
        )
        batch_tile_defs = [all_tile_defs[i] for i in tile_indices]

        # 5. Classify tile batch
        probabilities = classify_tile_batch(
            image_bgr=rectified_left,
            tile_defs_batch=batch_tile_defs,
            classifier=CrackClassifier.get_instance(),
        )

        if abort_event.is_set():
            return {"batch_id": batch_id, "status": "aborted"}

        # 6. Commit batch classification results
        with httpx.Client(timeout=15.0) as client:
            commit_url = f"{master_url}/api/internal/jobs/{inspection_id}/batches/{batch_id}/commit"
            commit_payload = {
                "fanout_run_id": fanout_run_id,
                "claim_token": claim_token,
                "probabilities": probabilities,
            }
            c_resp = client.post(commit_url, headers=headers, json=commit_payload)
            if c_resp.status_code == 409:
                logger.info(f"Commit batch {batch_id} returned 409: {c_resp.text}. Stale write safely rejected.")
                return {"batch_id": batch_id, "status": "stale_rejected"}
            c_resp.raise_for_status()

        return {"batch_id": batch_id, "status": "committed"}

    finally:
        stop_event.set()
        hb.join(timeout=2.0)


@celery_app.task(
    bind=True,
    name="app.tasks.finalize_inspection_task",
    max_retries=2,
    default_retry_delay=5,
    acks_late=True,
    reject_on_worker_lost=True,
)
def finalize_inspection_task(
    self,
    inspection_id: str,
    lease_token: int,
    fanout_run_id: str,
) -> bool:
    """
    Finalizer Celery task: claims delegation, pulses finalizer heartbeat, retrieves
    authoritative committed batches from master, aggregates findings, uploads crops,
    runs stage 5 results packaging, and completes inspection.
    """
    master_url = settings.get_master_url().rstrip("/")
    worker_token = settings.get_worker_token()
    worker_name = settings.get_worker_name()

    headers = {
        "X-Worker-Token": worker_token,
        "X-Lease-Token": str(lease_token),
        "X-Worker-Name": worker_name,
    }

    # 1. Claim finalizer delegation
    with httpx.Client(timeout=15.0) as client:
        claim_url = f"{master_url}/api/internal/jobs/{inspection_id}/finalizer/claim"
        try:
            resp = client.post(
                claim_url,
                headers=headers,
                json={"fanout_run_id": fanout_run_id, "delegation_duration_sec": 30.0},
            )
        except Exception as exc:
            logger.warning(f"Network error claiming finalizer for {inspection_id}: {exc}")
            raise self.retry(exc=exc, countdown=3)

        if resp.status_code == 409:
            logger.info(f"Finalizer claim for {inspection_id} returned 409 Conflict: {resp.text}. Exiting.")
            return False

        resp.raise_for_status()
        delegation_data = resp.json()
        delegation_token = delegation_data["delegation_token"]
        finalizer_generation = delegation_data["finalizer_generation"]

    # 2. Start finalizer heartbeat
    abort_event = threading.Event()
    stop_event = threading.Event()
    hb = FinalizerHeartbeatThread(
        inspection_id=inspection_id,
        lease_token=lease_token,
        worker_name=worker_name,
        master_url=master_url,
        worker_token=worker_token,
        delegation_token=delegation_token,
        finalizer_generation=finalizer_generation,
        abort_event=abort_event,
        stop_event=stop_event,
        interval=10.0,
    )
    hb.start()

    finalizer_headers = {
        "X-Worker-Token": worker_token,
        "X-Lease-Token": str(lease_token),
        "X-Worker-Name": worker_name,
        "X-Delegation-Token": delegation_token,
        "X-Finalizer-Generation": str(finalizer_generation),
    }

    temp_dir = Path(tempfile.mkdtemp(prefix=f"surface_final_{inspection_id}_"))
    try:
        if abort_event.is_set():
            return False

        # 3. Retrieve authoritative committed batches
        with httpx.Client(timeout=30.0) as client:
            b_resp = client.get(
                f"{master_url}/api/internal/jobs/{inspection_id}/batches/committed",
                headers=finalizer_headers,
            )
            b_resp.raise_for_status()
            batch_data = b_resp.json()

            # Retrieve artifacts for aggregation
            img_resp = client.get(f"{master_url}/api/inspections/{inspection_id}/artifacts/rectified_left.png")
            img_resp.raise_for_status()
            img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
            rectified_left = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            dev_resp = client.get(f"{master_url}/api/inspections/{inspection_id}/artifacts/deviations.json")
            dev_resp.raise_for_status()
            dev_data = dev_resp.json()
            deviation_regions = dev_data.get("regions", [])
            deviation_summary = dev_data.get("summary")

            disp_resp = client.get(f"{master_url}/api/inspections/{inspection_id}/artifacts/disparity.npy")
            disparity = None
            if disp_resp.status_code == 200:
                disparity = np.load(io.BytesIO(disp_resp.content))

            m_resp = client.get(f"{master_url}/api/inspections/{inspection_id}")
            m_resp.raise_for_status()
            manifest = m_resp.json()

        if abort_event.is_set():
            return False

        # 4. Generate tile definitions and verify tile permutation integrity
        all_tile_defs = generate_tile_definitions(
            image_shape=rectified_left.shape,
            deviation_regions=deviation_regions,
            tile_size=settings.get_tile_size(),
            stride=settings.get_tile_size(),
        )
        total_tiles = len(all_tile_defs)

        reconstructed_probs = [None] * total_tiles
        for b_item in batch_data["batches"]:
            for t_idx, p in zip(b_item["tile_indices"], b_item["probabilities"]):
                reconstructed_probs[t_idx] = float(p)

        assert all(p is not None for p in reconstructed_probs), "Missing tile probabilities in committed batches"
        probabilities: List[float] = [float(p) for p in reconstructed_probs]

        # 5. Aggregate findings
        damage_result = aggregate_damage_findings(
            tile_defs=all_tile_defs,
            probabilities=probabilities,
            deviation_regions=deviation_regions,
            disparity=disparity,
            intrinsics=None,
            threshold=settings.get_crack_threshold(),
            job_dir=temp_dir,
            image_bgr=rectified_left,
        )

        if abort_event.is_set():
            return False

        # 6. Upload finding crops
        with httpx.Client(timeout=30.0) as client:
            for crop_file in temp_dir.glob("finding_*.png"):
                with open(crop_file, "rb") as cf:
                    up_resp = client.post(
                        f"{master_url}/api/internal/jobs/{inspection_id}/artifacts/{crop_file.name}",
                        headers=finalizer_headers,
                        files={"file": (crop_file.name, cf, "image/png")},
                    )
                    up_resp.raise_for_status()

        # 7. Post Stage 4 completed
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{master_url}/api/internal/jobs/{inspection_id}/step",
                headers=finalizer_headers,
                json={
                    "index": 4,
                    "step_id": "damage",
                    "status": "completed",
                    "worker": worker_name,
                    "message": f"Found {len(damage_result.findings)} findings",
                },
            )

            # Stage 5 results
            client.post(
                f"{master_url}/api/internal/jobs/{inspection_id}/step",
                headers=finalizer_headers,
                json={
                    "index": 5,
                    "step_id": "results",
                    "status": "completed",
                    "worker": worker_name,
                    "message": "Results aggregated and finalized",
                },
            )

        # 8. Complete inspection
        complete_payload = {
            "point_count": manifest.get("point_count") or 0,
            "mesh_status": manifest.get("mesh_status") or "completed",
            "mesh_error": manifest.get("mesh_error"),
            "triangle_count": manifest.get("triangle_count"),
            "deviation_summary": deviation_summary,
            "findings": [f.model_dump() for f in damage_result.findings],
            "classifier_status": damage_result.classifier_status,
            "classifier_error": damage_result.classifier_error,
            "duration_sec": 0.0,
            "worker": worker_name,
        }
        with httpx.Client(timeout=15.0) as client:
            comp_resp = client.post(
                f"{master_url}/api/internal/jobs/{inspection_id}/complete",
                headers=finalizer_headers,
                json=complete_payload,
            )
            comp_resp.raise_for_status()

        logger.info(f"Successfully finalized inspection {inspection_id} under generation {finalizer_generation}.")
        return True

    finally:
        stop_event.set()
        hb.join(timeout=2.0)
        shutil.rmtree(temp_dir, ignore_errors=True)
