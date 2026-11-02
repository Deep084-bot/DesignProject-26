"""
Cluster Monitoring API Routes for SURFACE.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2.
Exposes cluster worker status and task summary:
- GET /api/cluster/workers
- GET /api/cluster/summary
Safely falls back to mode='local' and empty workers list when in local mode or when Redis is unreachable.
"""

from typing import List, Dict, Any, Set
from fastapi import APIRouter

from ..services.job_manager import job_manager
from ..celery_app import celery_app
from .. import settings


router = APIRouter(prefix="/api/cluster", tags=["cluster"])


@router.get("/workers")
async def get_cluster_workers():
    """
    Returns list of active Celery workers via inspect(timeout=1.5).
    In local mode or when Redis is unreachable, returns 200 with mode='local' and empty list.
    """
    mode = settings.get_execution_mode()
    if mode != "celery":
        return {"mode": "local", "workers": []}

    try:
        insp = celery_app.control.inspect(timeout=1.5)
        if insp is None:
            return {"mode": "local", "workers": []}

        pings = insp.ping()
        if not pings:
            # If celery mode but no workers are online
            return {"mode": "celery", "workers": []}

        active_dict = insp.active() or {}
        stats_dict = insp.stats() or {}

        workers = []
        for worker_name in sorted(pings.keys()):
            active_tasks = active_dict.get(worker_name, [])
            active_jobs: List[str] = []
            for task in active_tasks:
                args = task.get("args", [])
                if args and isinstance(args, (list, tuple)):
                    active_jobs.append(str(args[0]))
                elif "id" in task:
                    active_jobs.append(str(task["id"]))

            tasks_done_data = stats_dict.get(worker_name, {}).get("total", {})
            if isinstance(tasks_done_data, dict):
                total_done = sum(tasks_done_data.values())
            elif isinstance(tasks_done_data, (int, float)):
                total_done = int(tasks_done_data)
            else:
                total_done = 0

            workers.append({
                "name": worker_name,
                "online": True,
                "active_jobs": active_jobs,
                "tasks_done": total_done,
            })

        return {"mode": "celery", "workers": workers}
    except Exception:
        # Fall back safely on Redis disconnection or inspect errors
        return {"mode": "local", "workers": []}


@router.get("/summary")
async def get_cluster_summary():
    """
    Returns counts of queued, processing, completed, and failed jobs,
    along with active worker assignments derived from job state.
    Cross-references live worker status from Celery inspect ping.
    """
    mode = settings.get_execution_mode()
    jobs = job_manager.list_jobs()

    queued = sum(1 for j in jobs if j.status == "queued")
    processing = sum(1 for j in jobs if j.status == "processing")
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")

    online_workers: Set[str] = set()
    inspect_succeeded = False
    if mode == "celery":
        try:
            insp = celery_app.control.inspect(timeout=1.0)
            if insp is not None:
                pings = insp.ping()
                if pings is not None:
                    raw_online = set(pings.keys())
                    online_workers = raw_online | {
                        norm for w in raw_online if (norm := settings.normalize_worker_identity(w))
                    }
                    inspect_succeeded = True
        except Exception:
            pass

    active_assignments: Dict[str, str] = {}
    for j in jobs:
        if j.status in ("queued", "processing"):
            assigned_worker = j.assigned_worker
            if not assigned_worker:
                for step in j.steps:
                    if step.worker:
                        assigned_worker = step.worker
            if assigned_worker:
                assigned_norm = settings.normalize_worker_identity(assigned_worker)
                is_online = (assigned_worker in online_workers) or (assigned_norm in online_workers if assigned_norm else False)
                if mode == "celery" and inspect_succeeded and not is_online:
                    active_assignments[j.id] = f"{assigned_worker} (unresponsive)"
                else:
                    active_assignments[j.id] = assigned_worker

    return {
        "mode": mode,
        "total_jobs": len(jobs),
        "queued": queued,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "active_assignments": active_assignments,
    }
