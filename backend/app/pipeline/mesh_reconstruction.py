"""
Surface Mesh Reconstruction Adapter.

Integrates Satyam's Poisson Surface Reconstruction pipeline (PointCloudToMesh)
into the Surface 3D Structural Inspection backend.

Converts oriented 3D point cloud coordinates (in millimeters) into a clean,
watertight manifold triangular mesh with surface normals and transferred vertex colors.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import threading

logger = logging.getLogger(__name__)

# Mutex for C++ PoissonRecon library thread-safety across concurrent workers
_POISSON_LOCK = threading.Lock()

# Resolve path to PointCloudToMesh/backend
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PCM_BACKEND = _REPO_ROOT / "PointCloudToMesh" / "backend"


def _ensure_pcm_in_namespace():
    """
    Extends the backend namespace path to include PointCloudToMesh/backend.
    This enables direct imports of backend.geometry.poisson without modifying
    Satyam's code or creating module collision with the main backend package.
    """
    import backend
    pcm_path_str = str(_PCM_BACKEND.resolve())
    if _PCM_BACKEND.exists() and pcm_path_str not in backend.__path__:
        backend.__path__.append(pcm_path_str)


def reconstruct_surface_mesh(
    point_cloud_ply_path: Path,
    output_mesh_ply_path: Path,
    depth: int = 8,
    density_trim_percent: float = 5.0,
    k_neighbors: int = 30,
    scale: float = 1.1,
    linear_fit: bool = False,
    n_threads: int = 1,
    output_display_ply_path: Optional[Path] = None,
    max_display_triangles: int = 3_000_000,
) -> Dict[str, Any]:
    """
    Invokes Poisson Surface Reconstruction on an input PLY point cloud.

    Args:
        point_cloud_ply_path: Path to valid input PLY point cloud file.
        output_mesh_ply_path: Destination path for full-precision output PLY mesh.
        depth: Octree maximum depth for Poisson solver (default 8, valid 6-11).
        density_trim_percent: Percentile of lowest-density hull vertices to remove (default 5.0%).
        k_neighbors: Number of nearest neighbors for normal estimation if normals are missing.
        scale: Bounding box expansion ratio for solver stability.
        linear_fit: Whether to use linear interpolation for iso-surface extraction.
        output_display_ply_path: Optional path for decimated WebGL preview mesh.
        max_display_triangles: Maximum triangles threshold before decimating preview mesh.

    Returns:
        Dict containing reconstruction statistics (num_vertices, num_triangles, processing_time_sec, bounds, etc.).

    Raises:
        FileNotFoundError: If input PLY file does not exist.
        RuntimeError: If Open3D is unavailable or reconstruction fails.
        ValueError: If point cloud has fewer than 10 points.
    """
    _ensure_pcm_in_namespace()

    try:
        from backend.geometry.poisson import run_poisson_reconstruction
    except ImportError as exc:
        logger.error(f"Failed to import Poisson reconstruction module: {exc}")
        raise RuntimeError(
            f"Poisson reconstruction module is unavailable. Verify Open3D and SciPy are installed in backend/venv. Error: {exc}"
        ) from exc

    input_path = Path(point_cloud_ply_path)
    output_path = Path(output_mesh_ply_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {input_path}")

    logger.info(
        f"Starting Poisson surface reconstruction: {input_path.name} -> {output_path.name} "
        f"(octree_depth={depth}, trim={density_trim_percent}%)"
    )

    with _POISSON_LOCK:
        result = run_poisson_reconstruction(
            input_ply_path=input_path,
            output_ply_path=output_path,
            depth=depth,
            density_trim_percent=density_trim_percent,
            k_neighbors=k_neighbors,
            scale=scale,
            linear_fit=linear_fit,
            n_threads=n_threads,
            output_display_ply_path=output_display_ply_path,
            max_display_triangles=max_display_triangles,
        )

    logger.info(
        f"Surface mesh reconstruction complete: {result.get('num_vertices', 0):,} vertices, "
        f"{result.get('num_triangles', 0):,} triangles in {result.get('processing_time_sec', 0)}s"
    )

    return result
