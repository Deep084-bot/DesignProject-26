"""
Central settings and environment configuration for the SURFACE inspection system.

Authoritative specification: SURFACE_Master_Prompt.md Section 2.5.
All settings have safe defaults and can be overridden via environment variables.
Supports dynamic lookup so test monkeypatching is immediately reflected.
"""

import os
import socket
from typing import List


def get_execution_mode() -> str:
    """Returns 'local' (default) or 'celery'."""
    val = os.getenv("SURFACE_EXECUTION", "local").strip().lower()
    return "celery" if val == "celery" else "local"


def get_redis_url() -> str:
    """Returns Celery broker and result backend URL (default: redis://localhost:6379/0)."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()


def get_master_url() -> str:
    """Returns the master API URL reachable by workers (default: http://localhost:8000)."""
    return os.getenv("SURFACE_MASTER_URL", "http://localhost:8000").strip().rstrip("/")


def get_worker_token() -> str:
    """Returns the shared secret token for /api/internal/* routes (default: empty string)."""
    return os.getenv("SURFACE_WORKER_TOKEN", "").strip()


def get_worker_name() -> str:
    """Returns the worker identifier displayed in the UI (default: system hostname)."""
    return os.getenv("SURFACE_WORKER_NAME", socket.gethostname()).strip()


def get_cors_origins() -> List[str]:
    """Returns additional CORS origins parsed from comma-separated SURFACE_CORS_ORIGINS."""
    raw = os.getenv("SURFACE_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_model_path() -> str:
    """Returns crack classifier checkpoint path."""
    return os.getenv("SURFACE_MODEL_PATH", "backend/models_store/best_model_sdnet2018.pth").strip()


def get_crack_threshold() -> float:
    """Returns probability threshold for classifying a tile as cracked (default: 0.5)."""
    try:
        return float(os.getenv("SURFACE_CRACK_THRESHOLD", "0.5"))
    except (ValueError, TypeError):
        return 0.5


def get_tile_size() -> int:
    """Returns tile size in pixels (default: 256)."""
    try:
        return int(os.getenv("SURFACE_TILE_SIZE", "256"))
    except (ValueError, TypeError):
        return 256


def get_dev_k() -> float:
    """Returns noise multiplier for deviation threshold (default: 3.0)."""
    try:
        return float(os.getenv("SURFACE_DEV_K", "3.0"))
    except (ValueError, TypeError):
        return 3.0


def get_dev_min_mm() -> float:
    """Returns minimum deviation threshold in mm (default: 5.0)."""
    try:
        return float(os.getenv("SURFACE_DEV_MIN_MM", "5.0"))
    except (ValueError, TypeError):
        return 5.0


def get_dev_min_area_px() -> int:
    """Returns smallest deviation region kept in pixels (default: 200)."""
    try:
        return int(os.getenv("SURFACE_DEV_MIN_AREA_PX", "200"))
    except (ValueError, TypeError):
        return 200


def get_mesh_depth() -> int:
    """Returns Poisson octree depth (default: 8)."""
    try:
        depth = int(os.getenv("SURFACE_MESH_DEPTH", "8"))
        return depth if depth > 0 else 8
    except (ValueError, TypeError):
        return 8


# Module-level __getattr__ enables direct access like `settings.SURFACE_EXECUTION`
# while dynamically evaluating os.getenv at access time.
def __getattr__(name: str):
    if name == "SURFACE_EXECUTION":
        return get_execution_mode()
    if name == "REDIS_URL":
        return get_redis_url()
    if name == "SURFACE_MASTER_URL":
        return get_master_url()
    if name == "SURFACE_WORKER_TOKEN":
        return get_worker_token()
    if name == "SURFACE_WORKER_NAME":
        return get_worker_name()
    if name == "SURFACE_CORS_ORIGINS":
        return get_cors_origins()
    if name == "SURFACE_MODEL_PATH":
        return get_model_path()
    if name == "SURFACE_CRACK_THRESHOLD":
        return get_crack_threshold()
    if name == "SURFACE_TILE_SIZE":
        return get_tile_size()
    if name == "SURFACE_DEV_K":
        return get_dev_k()
    if name == "SURFACE_DEV_MIN_MM":
        return get_dev_min_mm()
    if name == "SURFACE_DEV_MIN_AREA_PX":
        return get_dev_min_area_px()
    if name == "SURFACE_MESH_DEPTH":
        return get_mesh_depth()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
