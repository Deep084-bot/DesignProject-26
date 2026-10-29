"""
Poisson Surface Reconstruction Pipeline
Executes Poisson reconstruction using Open3D with density trimming,
color transfer, and topology cleanup.
"""
from pathlib import Path
import time
from typing import Dict, Any, Optional
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False

from backend.geometry.normals import ensure_oriented_normals


def transfer_colors_to_mesh(pcd, mesh) -> None:
    """
    Transfers color information from the original point cloud to the reconstructed mesh vertices
    using high-performance spatial nearest-neighbor lookup.
    Uses multi-threaded SciPy cKDTree for instant processing on large models up to 1GB.
    """
    if not pcd.has_colors() or len(mesh.vertices) == 0:
        return

    try:
        from scipy.spatial import cKDTree
        pcd_pts = np.asarray(pcd.points)
        mesh_verts = np.asarray(mesh.vertices)
        pcd_cols = np.asarray(pcd.colors)

        # Multi-threaded query across all available CPU cores
        tree = cKDTree(pcd_pts)
        _, idxs = tree.query(mesh_verts, k=1, workers=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_cols[idxs])
    except Exception:
        try:
            # Fallback to Open3D KDTreeFlann
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            mesh_vertices = np.asarray(mesh.vertices)
            pcd_colors = np.asarray(pcd.colors)
            new_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float64)

            for i, vert in enumerate(mesh_vertices):
                _, idx, _ = pcd_tree.search_knn_vector_3d(vert, 1)
                if idx:
                    new_colors[i] = pcd_colors[idx[0]]

            mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        except Exception as e:
            print(f"Warning: Color transfer skipped due to error: {e}")


def run_poisson_reconstruction(
    input_ply_path: Path,
    output_ply_path: Path,
    depth: int = 8,
    density_trim_percent: float = 5.0,
    k_neighbors: int = 30,
    scale: float = 1.1,
    linear_fit: bool = False,
    n_threads: int = 1,
    output_display_ply_path: Optional[Path] = None,
    max_display_triangles: int = 3_000_000
) -> Dict[str, Any]:
    """
    Executes the complete Poisson Surface Reconstruction pipeline:
    1. Loads and validates point cloud.
    2. Ensures robust oriented normals.
    3. Runs Octree-based Poisson Surface Reconstruction.
    4. Trims low-density artifact vertices based on percentile threshold.
    5. Cleans non-manifold geometry and computes smooth vertex normals.
    6. Transfers vertex colors if present in input point cloud.
    7. Exports clean 100% full-resolution triangle mesh as PLY.
    8. Generates adaptive WebGL display mesh if polygon count exceeds threshold.
    """
    if not HAS_OPEN3D:
        raise RuntimeError(
            "Open3D is not installed. Please install it using: pip install open3d"
        )

    start_time = time.time()

    # 1. Load point cloud
    if not input_ply_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_ply_path}")

    pcd = o3d.io.read_point_cloud(str(input_ply_path))
    num_input_points = len(pcd.points)

    if num_input_points < 10:
        raise ValueError(
            f"Point cloud must contain at least 10 points for reconstruction (found {num_input_points})."
        )

    # Clean non-finite coordinates
    pcd.remove_non_finite_points()

    # 2. Normal estimation & orientation
    pcd, normals_computed = ensure_oriented_normals(pcd, k_neighbors=k_neighbors)

    # 3. Poisson Surface Reconstruction
    # Clamp octree depth to safe limits (6 to 11)
    octree_depth = max(6, min(11, int(depth)))
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=octree_depth,
        scale=float(scale),
        linear_fit=bool(linear_fit),
        n_threads=int(n_threads)
    )

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Poisson Surface Reconstruction produced an empty mesh.")

    # 4. Density Trimming
    # Poisson reconstruction naturally produces an enclosing bounding hull.
    # The density array indicates point sampling density on the extracted surface.
    densities = np.asarray(densities)
    trimmed = False
    initial_vertices = len(mesh.vertices)

    if density_trim_percent > 0 and len(densities) == initial_vertices:
        trim_quantile = max(0.0, min(0.25, density_trim_percent / 100.0))
        density_threshold = np.quantile(densities, trim_quantile)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        trimmed = True

    # Mesh topology cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Verify mesh still has geometry after cleanup
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(
            "Density trimming or mesh cleanup removed all triangles. "
            "Try reducing the density trim percentage or increasing octree depth."
        )

    # 5. Compute smooth normals for rendering
    mesh.compute_vertex_normals()

    # 6. Transfer colors if available
    has_colors = pcd.has_colors()
    if has_colors:
        transfer_colors_to_mesh(pcd, mesh)

    # 7. Export full-resolution mesh to PLY (100% full precision for download)
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_triangle_mesh(
        str(output_ply_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=has_colors
    )

    if not success:
        raise IOError(f"Failed to write reconstructed mesh to {output_ply_path}")

    # 8. Adaptive Display Optimization for Large Models
    has_display_mesh = False
    display_triangles = len(mesh.triangles)
    if output_display_ply_path and len(mesh.triangles) > max_display_triangles:
        try:
            display_mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=max_display_triangles
            )
            display_mesh.compute_vertex_normals()
            output_display_ply_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(
                str(output_display_ply_path),
                display_mesh,
                write_ascii=False,
                compressed=False,
                write_vertex_normals=True,
                write_vertex_colors=has_colors
            )
            has_display_mesh = True
            display_triangles = len(display_mesh.triangles)
        except Exception as e:
            print(f"Warning: Display mesh decimation skipped: {e}")

    elapsed_time = round(time.time() - start_time, 2)
    min_bound = mesh.get_min_bound().tolist()
    max_bound = mesh.get_max_bound().tolist()
    center = mesh.get_center().tolist()
    dimensions = (mesh.get_max_bound() - mesh.get_min_bound()).tolist()

    return {
        "success": True,
        "num_input_points": num_input_points,
        "num_vertices": int(len(mesh.vertices)),
        "num_triangles": int(len(mesh.triangles)),
        "is_watertight": bool(mesh.is_watertight()),
        "has_vertex_normals": True,
        "has_vertex_colors": bool(has_colors),
        "normals_computed": normals_computed,
        "octree_depth": octree_depth,
        "density_trimmed": trimmed,
        "has_display_mesh": has_display_mesh,
        "display_triangles": display_triangles,
        "processing_time_sec": elapsed_time,
        "bounds": {
            "min": [round(v, 4) for v in min_bound],
            "max": [round(v, 4) for v in max_bound],
            "center": [round(v, 4) for v in center],
            "dimensions": [round(v, 4) for v in dimensions]
        }
    }

