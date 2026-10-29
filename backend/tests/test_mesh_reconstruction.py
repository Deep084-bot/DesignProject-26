"""
Tests for Surface Mesh Reconstruction (Satyam's Poisson Pipeline Integration).
"""

from pathlib import Path
import numpy as np
import pytest

from backend.app.pipeline.ply_writer import write_ply
from backend.app.pipeline.mesh_reconstruction import reconstruct_surface_mesh
from backend.app.pipeline.runner import run_pipeline
from backend.app.models.calibration import StereoCalibration, RectifiedStereoCalibration


def test_reconstruct_surface_mesh_success(tmp_path: Path):
    """Verifies Poisson surface reconstruction generates a valid triangular mesh."""
    # Generate 500 random points on a sphere
    phi = np.random.uniform(0, 2 * np.pi, 500)
    costheta = np.random.uniform(-1, 1, 500)
    theta = np.arccos(costheta)
    r = 50.0

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = np.full((500, 3), 180, dtype=np.uint8)

    pcd_file = tmp_path / "test_points.ply"
    mesh_file = tmp_path / "test_mesh.ply"

    write_ply(pcd_file, points, colors, binary=True)
    assert pcd_file.exists()

    result = reconstruct_surface_mesh(
        point_cloud_ply_path=pcd_file,
        output_mesh_ply_path=mesh_file,
        depth=6,
        density_trim_percent=5.0,
    )

    assert result["success"] is True
    assert mesh_file.exists()
    assert mesh_file.stat().st_size > 0
    assert result["num_vertices"] > 0
    assert result["num_triangles"] > 0
    assert result["has_vertex_normals"] is True
    assert "bounds" in result


def test_reconstruct_surface_mesh_missing_file(tmp_path: Path):
    """Verifies clean FileNotFoundError when input PLY is missing."""
    missing_file = tmp_path / "non_existent.ply"
    out_file = tmp_path / "out.ply"

    with pytest.raises(FileNotFoundError):
        reconstruct_surface_mesh(missing_file, out_file)


def test_reconstruct_surface_mesh_insufficient_points(tmp_path: Path):
    """Verifies ValueError when point cloud has fewer than 10 points."""
    points = np.zeros((5, 3), dtype=np.float32)
    colors = np.zeros((5, 3), dtype=np.uint8)

    pcd_file = tmp_path / "sparse.ply"
    mesh_file = tmp_path / "out.ply"
    write_ply(pcd_file, points, colors, binary=True)

    with pytest.raises(ValueError, match="at least 10 points"):
        reconstruct_surface_mesh(pcd_file, mesh_file)


def test_runner_produces_mesh_artifact(synthetic_stereo_pair, tmp_path: Path):
    """Verifies that full pipeline runner produces both pointcloud.ply and mesh.ply."""
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    with open(left_path, "wb") as f:
        f.write(synthetic_stereo_pair["left_bytes"])
    with open(right_path, "wb") as f:
        f.write(synthetic_stereo_pair["right_bytes"])

    calib = RectifiedStereoCalibration(
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=synthetic_stereo_pair["width"],
        image_height=synthetic_stereo_pair["height"],
    )

    result = run_pipeline(
        job_dir=tmp_path / "job_run",
        left_image_path=left_path,
        right_image_path=right_path,
        calibration=calib,
    )

    assert result.success is True
    assert result.point_count > 0
    assert result.point_cloud_path is not None
    assert result.point_cloud_path.exists()

    # Verify mesh was reconstructed
    assert result.mesh_path is not None
    assert result.mesh_path.exists()
    assert result.mesh_path.stat().st_size > 0
    assert result.triangle_count is not None
    assert result.triangle_count > 0
    assert result.mesh_status == "completed"
    assert result.mesh_error is None


def test_runner_mesh_failure_preserves_pointcloud(monkeypatch, synthetic_stereo_pair, tmp_path: Path):
    """
    Verifies that when surface mesh generation fails, the pipeline does NOT fail
    the entire inspection: pointcloud.ply is preserved, and mesh_status is marked 'failed'.
    """
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    with open(left_path, "wb") as f:
        f.write(synthetic_stereo_pair["left_bytes"])
    with open(right_path, "wb") as f:
        f.write(synthetic_stereo_pair["right_bytes"])

    calib = RectifiedStereoCalibration(
        baseline_distance_mm=120.0,
        focal_length_px=500.0,
        image_width=synthetic_stereo_pair["width"],
        image_height=synthetic_stereo_pair["height"],
    )

    # Simulate an unexpected failure inside mesh reconstruction
    def failing_mesh_reconstruction(*args, **kwargs):
        raise RuntimeError("Simulated Open3D Poisson solver failure")

    import backend.app.pipeline.runner as runner_mod
    monkeypatch.setattr(runner_mod, "reconstruct_surface_mesh", failing_mesh_reconstruction)

    result = run_pipeline(
        job_dir=tmp_path / "job_run_fail",
        left_image_path=left_path,
        right_image_path=right_path,
        calibration=calib,
    )

    # Point cloud generation must succeed
    assert result.success is True
    assert result.point_count > 0
    assert result.point_cloud_path is not None
    assert result.point_cloud_path.exists()

    # Mesh must be cleanly marked as failed without fake data
    assert result.mesh_path is None
    assert result.triangle_count is None
    assert result.mesh_status == "failed"
    assert "Simulated Open3D Poisson solver failure" in result.mesh_error
