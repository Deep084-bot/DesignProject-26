"""
Recovery Supervisor for Distributed Job Execution in SURFACE.

Authoritative specification: Approved M4a Recovery Architecture.
Periodically audits active inspections, detects dead or unresponsive workers,
and safely triggers atomic CAS reassignment and task recovery.
"""

import asyncio
import logging
import time
from typing import Dict, Set

from .job_manager import JobManager, job_manager
from ..celery_app import celery_app
from .. import settings

logger = logging.getLogger(__name__)


class RecoverySupervisor:
    def __init__(self, manager: JobManager = job_manager):
        self.job_manager = manager
        self._running = False
        self._grace_periods: Dict[str, float] = {}

    async def check_recovery_cycle(self):
        """
        Executes one evaluation cycle across active jobs following the approved decision table:
        1. Worker Online + Fresh Heartbeat (<= 35s): Retain assignment, clear grace.
        2. Worker Online + Expired Heartbeat (> 35s): 15s grace period, then recover.
        3. Worker Offline + Fresh Heartbeat (<= 35s): 15s grace period, then recover.
        4. Worker Offline + Expired Heartbeat (> 35s): Immediate recovery.
        5. Redis/broker unreachable: Safely defer cycle without mutation.
        """
        mode = settings.get_execution_mode()
        if mode != "celery":
            return

        # Check broker connectivity and active worker pings
        try:
            loop = asyncio.get_running_loop()
            insp = await loop.run_in_executor(None, lambda: celery_app.control.inspect(timeout=1.5))
            if insp is None:
                logger.warning("Recovery supervisor: inspect returned None. Deferring cycle.")
                return
            pings = await loop.run_in_executor(None, insp.ping)
            if pings is None:
                logger.warning("Recovery supervisor: Celery ping failed. Deferring cycle.")
                return
            online_workers: Set[str] = set(pings.keys())
        except Exception as exc:
            logger.warning(f"Recovery supervisor: Redis broker error ({exc}). Deferring recovery.")
            return

        jobs = self.job_manager.list_jobs()
        lease_timeout = settings.get_lease_timeout()
        grace_threshold = settings.get_offline_grace_period()
        now_mono = time.monotonic()

        for job in jobs:
            if job.status not in ("queued", "processing"):
                self._grace_periods.pop(job.id, None)
                continue

            if not job.assigned_worker:
                # Job has not been claimed by a worker yet
                self._grace_periods.pop(job.id, None)
                continue

            is_online = job.assigned_worker in online_workers
            heartbeat_age = self.job_manager.get_heartbeat_age(job.id)
            heartbeat_fresh = (heartbeat_age is not None and heartbeat_age <= lease_timeout)

            # Decision Table Evaluation
            if is_online and heartbeat_fresh:
                # 1. Healthy assignment
                self._grace_periods.pop(job.id, None)
                continue

            elif is_online and not heartbeat_fresh:
                # 2. Worker process online but task is silent/hung
                if job.id not in self._grace_periods:
                    self._grace_periods[job.id] = now_mono
                    logger.info(
                        f"Job {job.id} heartbeat expired ({heartbeat_age}s > {lease_timeout}s) "
                        f"while worker '{job.assigned_worker}' is online. Grace period started."
                    )
                    continue
                elif (now_mono - self._grace_periods[job.id]) < grace_threshold:
                    continue

            elif not is_online and heartbeat_fresh:
                # 3. Transient ping drop
                if job.id not in self._grace_periods:
                    self._grace_periods[job.id] = now_mono
                    logger.info(
                        f"Worker '{job.assigned_worker}' for job {job.id} is offline "
                        f"with fresh heartbeat ({heartbeat_age}s). Grace period started."
                    )
                    continue
                elif (now_mono - self._grace_periods[job.id]) < grace_threshold:
                    continue

            else:
                # 4. Worker is offline AND heartbeat expired -> Immediate recovery
                logger.info(
                    f"Worker '{job.assigned_worker}' offline and heartbeat expired "
                    f"for job {job.id}. Eligible for immediate recovery."
                )

            # Trigger Atomic CAS Reassignment
            self._grace_periods.pop(job.id, None)
            expected_lease = job.lease_token
            new_lease = self.job_manager.reassign_orphaned_job(
                inspection_id=job.id,
                expected_lease_token=expected_lease,
                reason=f"Worker '{job.assigned_worker}' loss / heartbeat timeout",
            )
            if new_lease is not None:
                logger.info(
                    f"Reassigned orphaned job {job.id}: incremented lease from "
                    f"{expected_lease} to {new_lease}. Enqueuing recovery task..."
                )
                try:
                    from ..tasks import run_inspection_task
                    run_inspection_task.delay(job.id, new_lease)
                    self.job_manager.mark_outbox_published(job.id, new_lease)
                    self.job_manager.cleanup_outbox(job.id, new_lease)
                    logger.info(f"Successfully enqueued recovery task for job {job.id} with lease {new_lease}.")
                except Exception as pub_err:
                    logger.error(f"Failed to publish recovery task for job {job.id} (lease {new_lease}): {pub_err}")

    async def run_loop(self):
        """Autonomous supervisor background loop running every SURFACE_SUPERVISOR_INTERVAL seconds."""
        self._running = True
        logger.info("Recovery supervisor background loop started.")
        while self._running:
            try:
                await self.check_recovery_cycle()
            except Exception as exc:
                logger.error(f"Error in recovery supervisor cycle: {exc}")
            interval = settings.get_supervisor_interval()
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
        logger.info("Recovery supervisor background loop stopped.")

    def stop(self):
        self._running = False


# Global singleton instance
recovery_supervisor = RecoverySupervisor(job_manager)
