"""
Celery Application Configuration for SURFACE.

Authoritative specification: SURFACE_Master_Prompt.md Section 4.2.
Configures broker and backend from REDIS_URL with late acknowledgments,
pre-fetch tuning, task timeouts, and JSON serialization.
"""

from celery import Celery
from . import settings

# Instantiate Celery app
celery_app = Celery(
    "surface",
    broker=settings.get_redis_url(),
    backend=settings.get_redis_url(),
    include=["app.tasks"],
)

# Apply authoritative production configurations
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 900},
    task_time_limit=600,
    task_soft_time_limit=540,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
