"""
Stereo-to-3D Reconstruction Pipeline Runner.

Orchestrates sequential pipeline stages:
0. prepare: Decode images, verify stereo resolution, align epipolar perspectives.
1. depth: Compute StereoSGBM sub-pixel disparity map and visualization.
2. surface: Triangulate 3D physical coordinates and export PLY point cloud.
3. deviations: Real measurement placeholder (milestone 1: zero fabricated deviations).
4. damage: Anomaly classification placeholder (milestone 1: zero fabricated findings).
5. results: Finalize artifact manifest and timing metadata.

Reports authentic progress and explicit errors without simulated timers.
"""

from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
import os
import time
import json
import cv2
import numpy as np

from .calibration_parser import parse_calibration
from .rectification import prepare_and_rectify
from .stereo import compute_stereo_sgbm
from .depth import reconstruct_3d_points
from .ply_writer import write_ply
from .mesh_reconstruction import reconstruct_surface_mesh
from .deviations import analyze_deviations, compute_surface_deviations, DeviationRegion
from .damage import detect_damage
from ..models.calibration import StereoCalibration
from ..models.inspection import Finding


ProgressCallback = Callable[[int, str, str, Optional[str]], None]


class PipelineExecutionResult:
    def __init__(
        self,
        success: bool,
        point_count: int = 0,
        point_cloud_path: Optional[Path] = None,
        mesh_path: Optional[Path] = None,
        triangle_count: Optional[int] = None,
        mesh_status: Optional[str] = None,
        mesh_error: Optional[str] = None,
        disparity_path: Optional[Path] = None,
        disparity_vis_path: Optional[Path] = None,
        rectified_left_path: Optional[Path] = None,
        rectified_right_path: Optional[Path] = None,
        deviation_map_path: Optional[Path] = None,
        deviation_heatmap_path: Optional[Path] = None,
        deviation_regions: Optional[list] = None,
        deviation_summary: Optional[Dict[str, Any]] = None,
        findings: Optional[List[Finding]] = None,
        classifier_status: Optional[str] = None,
        classifier_error: Optional[str] = None,
        duration_sec: float = 0.0,
        failure_message: Optional[str] = None,
        failed_stage_index: Optional[int] = None,
        deviation_stats: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.point_count = point_count
        self.point_cloud_path = point_cloud_path
        self.mesh_path = mesh_path
        self.triangle_count = triangle_count
        self.mesh_status = mesh_status
        self.mesh_error = mesh_error
        self.disparity_path = disparity_path
        self.disparity_vis_path = disparity_vis_path
        self.rectified_left_path = rectified_left_path
        self.rectified_right_path = rectified_right_path
        self.deviation_map_path = deviation_map_path
        self.deviation_heatmap_path = deviation_heatmap_path
        self.deviation_regions = deviation_regions or []
        self.deviation_summary = deviation_summary if deviation_summary is not None else deviation_stats
        self.deviation_stats = self.deviation_summary
        self.findings = findings or []
        self.classifier_status = classifier_status
        self.classifier_error = classifier_error
        self.duration_sec = duration_sec
        self.failure_message = failure_message
        self.failed_stage_index = failed_stage_index


def run_pipeline(
    job_dir: Path,
    left_image_path: Path,
    right_image_path: Path,
    calibration: StereoCalibration,
    on_step_change: Optional[ProgressCallback] = None,
) -> PipelineExecutionResult:
    """
    Executes the stereo processing pipeline synchronously.

    Args:
        job_dir: Working directory for artifacts.
        left_image_path: Path to stored left image file.
        right_image_path: Path to stored right image file.
        calibration: Verified StereoCalibration object.
        on_step_change: Callback invoked as (step_index, step_id, status, error_msg).

    Returns:
        PipelineExecutionResult with execution details and artifact paths.
    """
    start_time = time.perf_counter()
    job_dir.mkdir(parents=True, exist_ok=True)

    def report(idx: int, step_id: str, status: str, msg: Optional[str] = None):
        if on_step_change:
            on_step_change(idx, step_id, status, msg)

    # ---------------------------------------------------------
    # Stage 0: prepare (Decode images, verify resolution, rectify)
    # ---------------------------------------------------------
    report(0, "prepare", "active")
    try:
        left_img = cv2.imread(str(left_image_path))
        if left_img is None:
            raise ValueError(f"Unable to read or decode left image at '{left_image_path.name}'.")

        right_img = cv2.imread(str(right_image_path))
        if right_img is None:
            raise ValueError(f"Unable to read or decode right image at '{right_image_path.name}'.")

        rect_params = prepare_and_rectify(left_img, right_img, calibration)

        rect_left_path = job_dir / "rectified_left.png"
        rect_right_path = job_dir / "rectified_right.png"
        cv2.imwrite(str(rect_left_path), rect_params.rectified_left)
        cv2.imwrite(str(rect_right_path), rect_params.rectified_right)

        report(0, "prepare", "completed")
    except Exception as e:
        err_msg = str(e)
        report(0, "prepare", "failed", err_msg)
        return PipelineExecutionResult(
            success=False,
            duration_sec=round(time.perf_counter() - start_time, 2),
            failure_message=err_msg,
            failed_stage_index=0,
        )

    # ---------------------------------------------------------
    # Stage 1: depth (StereoSGBM disparity calculation)
    # ---------------------------------------------------------
    report(1, "depth", "active")
    try:
        disparity_float, disparity_vis = compute_stereo_sgbm(
            rect_params.rectified_left,
            rect_params.rectified_right,
        )

        disp_npy_path = job_dir / "disparity.npy"
        disp_vis_path = job_dir / "disparity_vis.png"

        np.save(str(disp_npy_path), disparity_float)
        cv2.imwrite(str(disp_vis_path), disparity_vis)

        report(1, "depth", "completed")
    except Exception as e:
        err_msg = str(e)
        report(1, "depth", "failed", err_msg)
        return PipelineExecutionResult(
            success=False,
            duration_sec=round(time.perf_counter() - start_time, 2),
            failure_message=err_msg,
            failed_stage_index=1,
        )

    # ---------------------------------------------------------
    # Stage 2: surface (Depth triangulation & PLY export)
    # ---------------------------------------------------------
    report(2, "surface", "active")
    try:
        # For very large images, subsample slightly to keep browser rendering responsive
        h, w = disparity_float.shape[:2]
        step = 1
        if (w * h) > (1920 * 1080):
            step = 2

        pt_cloud = reconstruct_3d_points(
            disparity=disparity_float,
            baseline_mm=rect_params.baseline_mm,
            fx=rect_params.fx,
            fy=rect_params.fy,
            cx=rect_params.cx,
            cy=rect_params.cy,
            left_bgr_img=rect_params.rectified_left,
            min_disparity=0.5,
            subsample_step=step,
        )

        ply_path = job_dir / "pointcloud.ply"
        write_ply(ply_path, pt_cloud.points, pt_cloud.colors, binary=True)

        # Run Poisson Surface Reconstruction (from PointCloudToMesh)
        mesh_path: Optional[Path] = None
        triangle_count: Optional[int] = None
        mesh_status: str = "unavailable"
        mesh_error: Optional[str] = None

        if pt_cloud.valid_count >= 10:
            mesh_ply_path = job_dir / "mesh.ply"
            try:
                mesh_res = reconstruct_surface_mesh(
                    point_cloud_ply_path=ply_path,
                    output_mesh_ply_path=mesh_ply_path,
                    depth=None,
                    density_trim_percent=5.0,
                )
                if mesh_ply_path.exists() and mesh_ply_path.stat().st_size > 0:
                    mesh_path = mesh_ply_path
                    triangle_count = mesh_res.get("num_triangles")
                    mesh_status = "completed"
                else:
                    mesh_status = "failed"
                    mesh_error = "Poisson reconstruction produced an empty mesh."
            except Exception as mesh_err:
                import logging
                logging.getLogger(__name__).warning(f"Surface mesh reconstruction failed: {mesh_err}")
                mesh_status = "failed"
                mesh_error = str(mesh_err)
        else:
            mesh_status = "unavailable"
            mesh_error = f"Point cloud contains only {pt_cloud.valid_count} points (minimum 10 required for mesh reconstruction)."

        report(2, "surface", "completed")
    except Exception as e:
        err_msg = str(e)
        report(2, "surface", "failed", err_msg)
        return PipelineExecutionResult(
            success=False,
            duration_sec=round(time.perf_counter() - start_time, 2),
            failure_message=err_msg,
            failed_stage_index=2,
        )

    # ---------------------------------------------------------
    # Stage 3: deviations (Dense geometric deviation analysis)
    # ---------------------------------------------------------
    report(3, "deviations", "active")
    deviation_map_path: Optional[Path] = None
    deviation_heatmap_path: Optional[Path] = None
    deviation_regions: list = []
    deviation_summary: Optional[Dict[str, Any]] = None

    dev_k = float(os.environ.get("SURFACE_DEV_K", "3.0"))
    dev_min_mm = float(os.environ.get("SURFACE_DEV_MIN_MM", "5.0"))
    dev_min_area_px = int(os.environ.get("SURFACE_DEV_MIN_AREA_PX", "200"))

    try:
        dev_res = analyze_deviations(
            disparity=disparity_float,
            baseline_mm=rect_params.baseline_mm,
            fx=rect_params.fx,
            fy=rect_params.fy,
            cx=rect_params.cx,
            cy=rect_params.cy,
            left_bgr=rect_params.rectified_left,
            k=dev_k,
            min_mm=dev_min_mm,
            min_area_px=dev_min_area_px,
        )

        dev_map_file = job_dir / "deviation_map.npy"
        dev_heatmap_file = job_dir / "deviation_heatmap.png"
        deviations_json_file = job_dir / "deviations.json"

        np.save(str(dev_map_file), dev_res.deviation_map)
        cv2.imwrite(str(dev_heatmap_file), dev_res.heatmap_bgr)
        with open(deviations_json_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": dev_res.summary,
                "regions": [dict(r) for r in dev_res.regions],
            }, f, indent=2)

        deviation_map_path = dev_map_file
        deviation_heatmap_path = dev_heatmap_file
        deviation_regions = dev_res.regions
        deviation_summary = dev_res.summary

        report(3, "deviations", "completed")
    except Exception as e:
        err_msg = str(e)
        report(3, "deviations", "failed", err_msg)
        return PipelineExecutionResult(
            success=False,
            duration_sec=round(time.perf_counter() - start_time, 2),
            failure_message=err_msg,
            failed_stage_index=3,
        )

    # ---------------------------------------------------------
    # Stage 4: damage (Visual crack classification & finding correlation)
    # ---------------------------------------------------------
    report(4, "damage", "active")
    findings: List[Finding] = []
    classifier_status: Optional[str] = None
    classifier_error: Optional[str] = None

    tile_size = int(os.environ.get("SURFACE_TILE_SIZE", "256"))
    crack_threshold = float(os.environ.get("SURFACE_CRACK_THRESHOLD", "0.5"))

    intrinsics = {
        "fx": rect_params.fx,
        "fy": rect_params.fy,
        "cx": rect_params.cx,
        "cy": rect_params.cy,
        "baseline_mm": rect_params.baseline_mm,
    }

    try:
        damage_res = detect_damage(
            rectified_left_bgr=rect_params.rectified_left,
            deviation_regions=deviation_regions,
            disparity=disparity_float,
            intrinsics=intrinsics,
            tile=tile_size,
            threshold=crack_threshold,
            job_dir=job_dir,
        )
        findings = damage_res.findings
        classifier_status = damage_res.classifier_status
        classifier_error = damage_res.classifier_error
        report(4, "damage", "completed")
    except Exception as e:
        err_msg = str(e)
        report(4, "damage", "failed", err_msg)
        return PipelineExecutionResult(
            success=False,
            duration_sec=round(time.perf_counter() - start_time, 2),
            failure_message=err_msg,
            failed_stage_index=4,
        )

    # ---------------------------------------------------------
    # Stage 5: results (Wrap-up)
    # ---------------------------------------------------------
    report(5, "results", "active")
    duration = round(time.perf_counter() - start_time, 2)
    report(5, "results", "completed")

    return PipelineExecutionResult(
        success=True,
        point_count=pt_cloud.valid_count,
        point_cloud_path=ply_path,
        mesh_path=mesh_path,
        triangle_count=triangle_count,
        mesh_status=mesh_status,
        mesh_error=mesh_error,
        disparity_path=disp_npy_path,
        disparity_vis_path=disp_vis_path,
        rectified_left_path=rect_left_path,
        rectified_right_path=rect_right_path,
        deviation_map_path=deviation_map_path,
        deviation_heatmap_path=deviation_heatmap_path,
        deviation_regions=deviation_regions,
        deviation_summary=deviation_summary,
        findings=findings,
        classifier_status=classifier_status,
        classifier_error=classifier_error,
        duration_sec=duration,
    )
