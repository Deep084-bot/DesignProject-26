"""
Geometric Surface Deviation Analysis (Stage 3).

Fits a robust reference plane to dense 3D triangulated coordinates,
computes signed out-of-plane Euclidean deviations in millimeters:
- Positive deviation = farther than plane = depression / dent
- Negative deviation = closer than plane = bulge / protrusion

Estimates measurement noise using median absolute deviation (MAD * 1.4826),
extracts connected anomaly regions on the dense image grid, and outputs
metrics, deviation map, and diverging heatmap visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import cv2


class DeviationRegion(dict):
    """
    Structured representation of a connected surface deviation anomaly region.
    Inherits from dict for serialization compatibility and exposes properties
    for direct object attribute access.
    """

    def __init__(
        self,
        id: Union[int, str],
        bounding_box: Tuple[int, int, int, int],
        area_px: int,
        max_abs_deviation_mm: float,
        mean_deviation_mm: float,
        centroid_px: Tuple[float, float],
        centroid_3d: Tuple[float, float, float],
        sign: Optional[str] = None,
    ):
        x, y, w, h = bounding_box
        determined_sign = sign if sign is not None else ("depression" if mean_deviation_mm >= 0 else "bulge")
        c_px = [float(centroid_px[0]), float(centroid_px[1])]
        c_3d = [float(centroid_3d[0]), float(centroid_3d[1]), float(centroid_3d[2])]

        super().__init__(
            id=id,
            bbox_px={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            bounding_box=bounding_box,
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
            area_px=int(area_px),
            max_abs_deviation_mm=float(max_abs_deviation_mm),
            mean_deviation_mm=float(mean_deviation_mm),
            sign=determined_sign,
            centroid_px=c_px,
            centroid_3d=c_3d,
            centroid_3d_viewer=c_3d,
        )
        self.id = id
        self.bounding_box = bounding_box
        self.bbox_px = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        self.x = int(x)
        self.y = int(y)
        self.width = int(w)
        self.height = int(h)
        self.area_px = int(area_px)
        self.max_abs_deviation_mm = float(max_abs_deviation_mm)
        self.mean_deviation_mm = float(mean_deviation_mm)
        self.sign = determined_sign
        self.centroid_px = (c_px[0], c_px[1])
        self.centroid_3d = (c_3d[0], c_3d[1], c_3d[2])
        self.centroid_3d_viewer = (c_3d[0], c_3d[1], c_3d[2])


@dataclass
class DeviationAnalysisResult:
    """Complete results of dense surface deviation analysis."""

    deviation_map: np.ndarray  # Shape (H, W), float32, signed deviation in mm (NaN for invalid pixels)
    valid_mask: np.ndarray  # Shape (H, W), bool
    regions: List[DeviationRegion]
    summary: Dict[str, Any]
    plane_params: Tuple[float, float, float, float]  # (nx, ny, nz, d) where ||n|| = 1 and nz > 0
    noise_sigma_mm: float  # MAD * 1.4826
    threshold_mm: float  # max(k * sigma, min_mm)
    mean_deviation_mm: float
    max_abs_deviation_mm: float
    rmse_mm: float
    valid_pixel_count: int
    heatmap_bgr: np.ndarray  # Shape (H, W, 3), uint8 BGR visualization


def fit_reference_plane(
    points_3d: np.ndarray,
    distance_threshold_mm: float = 8.0,
    max_iterations: int = 200,
    random_seed: int = 42,
    max_refine_rounds: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Fits a robust reference plane ax + by + cz + d = 0 (||(a,b,c)|| = 1)
    to a set of 3D points using RANSAC followed by up to 5 rounds of
    SVD least-squares refinement on inliers (|r| < 2.5 * sigma_MAD).

    The normal vector (a, b, c) is oriented such that n_z > 0.
    Under this convention, for a surface located at Z > 0 in front of the camera:
    - Positive signed distance s = n . p + d > 0 represents a point farther than the plane (depression).
    - Negative signed distance s = n . p + d < 0 represents a point closer than the plane (bulge).

    Args:
        points_3d: (N, 3) float array of 3D coordinates in camera frame [X, Y, Z] (mm).
        distance_threshold_mm: Initial inlier distance threshold for RANSAC candidate selection.
        max_iterations: Number of RANSAC sampling iterations (200).
        random_seed: Seed for deterministic sampling.
        max_refine_rounds: Number of iterative least-squares refinement rounds (up to 5).

    Returns:
        (normal, d) where normal is shape (3,) unit vector with n_z > 0, and d is float plane offset.

    Raises:
        ValueError: If fewer than 10 points are provided.
    """
    n_pts = len(points_3d)
    if n_pts < 10:
        raise ValueError(f"At least 10 valid 3D points are required to fit a reference plane, got {n_pts}.")

    rng = np.random.RandomState(random_seed)

    # Subsample at most 50,000 points for RANSAC and refinement if point cloud is large
    if n_pts > 50_000:
        sample_indices = rng.choice(n_pts, 50_000, replace=False)
        eval_points = points_3d[sample_indices]
    else:
        eval_points = points_3d

    best_inlier_count = -1
    best_plane: Optional[Tuple[np.ndarray, float]] = None

    n_eval = len(eval_points)
    # 1. RANSAC Initialization (200 iterations)
    for _ in range(max_iterations):
        idx = rng.choice(n_eval, 3, replace=False)
        p1, p2, p3 = eval_points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        norm_vec = np.cross(v1, v2)
        norm_len = float(np.linalg.norm(norm_vec))
        if norm_len < 1e-6:
            continue

        n_cand = norm_vec / norm_len
        d_cand = -float(np.dot(n_cand, p1))
        if n_cand[2] < 0:
            n_cand = -n_cand
            d_cand = -d_cand

        residuals = np.abs(np.dot(eval_points, n_cand) + d_cand)
        inlier_count = int(np.sum(residuals < distance_threshold_mm))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = (n_cand, d_cand)

    if best_plane is None or best_inlier_count < 3:
        # Fallback to simple PCA plane fit
        centroid = np.mean(eval_points, axis=0)
        centered = eval_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2]
        d = -float(np.dot(normal, centroid))
        if normal[2] < 0:
            normal = -normal
            d = -d
        return normal, d

    normal, d = best_plane

    # 2. Up to 5 rounds of SVD least-squares refinement on inliers (|r| < 2.5 * sigma_MAD)
    for _ in range(max_refine_rounds):
        res = np.abs(np.dot(eval_points, normal) + d)
        med_res = float(np.median(res))
        mad_res = float(np.median(np.abs(res - med_res)))
        sigma_res = float(1.4826 * mad_res)
        robust_thresh = max(2.5 * sigma_res, 0.5)

        inlier_mask = res < robust_thresh
        inlier_pts = eval_points[inlier_mask]
        if len(inlier_pts) < 3:
            break

        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        new_normal = vh[2]
        new_d = -float(np.dot(new_normal, centroid))

        if new_normal[2] < 0:
            new_normal = -new_normal
            new_d = -new_d

        # Convergence check
        norm_diff = float(np.linalg.norm(new_normal - normal))
        offset_diff = abs(new_d - d)
        normal, d = new_normal, new_d
        if norm_diff < 1e-6 and offset_diff < 1e-4:
            break

    # Normalize normal
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
        d = -d

    return normal, d


def create_deviation_heatmap(
    deviation_map: np.ndarray,
    valid_mask: np.ndarray,
    clip_range_mm: float = 15.0,
) -> np.ndarray:
    """
    Renders a colorized diverging heatmap visualization of the surface deviation map:
    - Diverging colormap:
      - Blue: Bulge (negative deviation, closer than reference plane)
      - Green: Nominal reference surface (near 0.0 mm)
      - Red: Depression (positive deviation, farther than reference plane)
    - Symmetric clipping at ±clip_range_mm (±3T)
    - Invalid pixels: black [0, 0, 0]

    Args:
        deviation_map: (H, W) float array with signed deviation values in mm.
        valid_mask: (H, W) boolean mask of valid pixels.
        clip_range_mm: Color scale symmetric clipping limit [-clip, +clip].

    Returns:
        (H, W, 3) uint8 BGR heatmap image.
    """
    h, w = deviation_map.shape[:2]
    heatmap_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    if not np.any(valid_mask):
        return heatmap_bgr

    valid_devs = deviation_map[valid_mask]
    if clip_range_mm <= 0:
        clip_range_mm = 15.0

    # Map [-clip, +clip] to [0, 255] where negative is 0 (blue) and positive is 255 (red)
    scaled = np.clip((valid_devs + clip_range_mm) / (2.0 * clip_range_mm), 0.0, 1.0) * 255.0
    colored_pts = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_bgr[valid_mask] = colored_pts.reshape(-1, 3)

    return heatmap_bgr


def analyze_deviations(
    disparity: Optional[np.ndarray] = None,
    depth_grid_mm: Optional[np.ndarray] = None,
    baseline_mm: float = 120.0,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    left_bgr: Optional[np.ndarray] = None,
    min_disparity: float = 0.5,
    min_depth_mm: float = 50.0,
    max_depth_mm: float = 50000.0,
    k: float = 3.0,
    min_mm: float = 5.0,
    min_area_px: int = 200,
    apply_morphology: bool = True,
    max_regions: int = 50,
) -> DeviationAnalysisResult:
    """
    Performs dense geometric surface deviation analysis (Stage 3).

    Fits a robust reference plane to dense 3D triangulated coordinates, computes
    signed out-of-plane Euclidean deviations in millimeters (positive = depression,
    negative = bulge), estimates measurement noise via median absolute deviation
    (MAD * 1.4826), and extracts connected anomaly regions.

    Args:
        disparity: (H, W) float disparity map, or None if depth_grid_mm is given.
        depth_grid_mm: (H, W) float depth map in millimeters, or None if disparity is given.
        baseline_mm: Stereo camera baseline in millimeters (> 0).
        fx: Focal length horizontal in pixels (> 0).
        fy: Focal length vertical in pixels (> 0).
        cx: Principal point horizontal offset (defaults to image_width / 2).
        cy: Principal point vertical offset (defaults to image_height / 2).
        left_bgr: Optional rectified left image (H, W, 3) uint8 BGR.
        min_disparity: Minimum disparity threshold for valid depth.
        min_depth_mm: Minimum valid physical depth in millimeters (default 50.0 mm).
        max_depth_mm: Maximum valid physical depth in millimeters (default 50000.0 mm).
        k: Noise standard deviation multiplier for anomaly threshold (default 3.0).
        min_mm: Minimum absolute deviation threshold in millimeters (default 5.0 mm).
        min_area_px: Minimum connected component area in pixels to register as a region (default 200 px).
        apply_morphology: Whether to apply 3x3 open and 5x5 close cleanup.
        max_regions: Maximum number of anomaly regions to retain (default 50).

    Returns:
        DeviationAnalysisResult containing deviation map, anomaly regions, and metrics.

    Raises:
        ValueError: If input arrays or camera parameters are invalid, or if
                    fewer than 10% of pixels are valid.
    """
    # 1. Parameter Validation
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Focal length must be strictly positive, got fx={fx}, fy={fy}.")
    if k < 0:
        raise ValueError(f"Noise multiplier k must be non-negative, got {k}.")
    if min_mm < 0:
        raise ValueError(f"Minimum deviation threshold min_mm must be non-negative, got {min_mm}.")

    if depth_grid_mm is not None:
        if depth_grid_mm.ndim != 2:
            raise ValueError(f"depth_grid_mm must be a 2D array, got shape {depth_grid_mm.shape}.")
        h, w = depth_grid_mm.shape
        z_grid = depth_grid_mm.astype(np.float32)
        valid_mask = (
            np.isfinite(z_grid)
            & (z_grid >= min_depth_mm)
            & (z_grid <= max_depth_mm)
        )
    elif disparity is not None:
        if disparity.ndim != 2:
            raise ValueError(f"disparity must be a 2D array, got shape {disparity.shape}.")
        if baseline_mm <= 0:
            raise ValueError(f"Stereo baseline must be strictly positive, got {baseline_mm} mm.")
        h, w = disparity.shape
        valid_disp = np.isfinite(disparity) & (disparity > min_disparity)
        z_grid = np.full((h, w), np.nan, dtype=np.float32)
        z_grid[valid_disp] = (fx * baseline_mm) / disparity[valid_disp]
        valid_mask = (
            valid_disp
            & (z_grid >= min_depth_mm)
            & (z_grid <= max_depth_mm)
        )
    else:
        raise ValueError("Either 'disparity' or 'depth_grid_mm' must be provided.")

    total_pixels = h * w
    valid_pixel_count = int(np.sum(valid_mask))
    valid_fraction = valid_pixel_count / max(1, total_pixels)

    # Master specification gate: raise ValueError if fewer than 10% of pixels are valid
    if valid_fraction < 0.10 or valid_pixel_count < 10:
        raise ValueError(
            f"Insufficient valid 3D depth pixels ({valid_pixel_count}/{total_pixels} = {valid_fraction:.1%}). "
            f"At least 10% valid pixels required to perform surface deviation analysis."
        )

    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # 2. Triangulate Dense 3D Coordinates (Camera Frame: +X Right, +Y Down, +Z Forward in mm)
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]
    z_valid = z_grid[valid_mask]

    x_cam_valid = (u_valid - cx) * z_valid / fx
    y_cam_valid = (v_valid - cy) * z_valid / fy

    points_3d_cam = np.column_stack([x_cam_valid, y_cam_valid, z_valid])

    # 3. Fit Best-Fit Reference Plane to the Dense Surface (normal oriented with n_z > 0)
    normal, d_plane = fit_reference_plane(points_3d_cam)
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])

    # 4. Dense Signed Deviation Map (in mm)
    # Distance = normal . point + d_plane
    # Positive = depression (farther from camera), Negative = bulge (closer to camera)
    valid_deviations = (
        nx * x_cam_valid + ny * y_cam_valid + nz * z_valid + d_plane
    ).astype(np.float32)

    deviation_map = np.full((h, w), np.nan, dtype=np.float32)
    deviation_map[valid_mask] = valid_deviations

    # 5. Robust Noise Estimation via Median Absolute Deviation (MAD * 1.4826)
    med_dev = float(np.median(valid_deviations))
    abs_deviations = np.abs(valid_deviations - med_dev)
    mad = float(np.median(abs_deviations))
    noise_sigma_mm = float(mad * 1.4826)

    # 6. Configurable Anomaly Threshold: max(k * sigma, min_mm)
    threshold_mm = float(max(k * noise_sigma_mm, min_mm))

    # 7. Extract Anomaly Regions on the Dense Grid (|s| > T)
    anomaly_mask = valid_mask & (np.abs(deviation_map) > threshold_mm)

    # Morphological cleaning (open 3x3 then close 5x5)
    if apply_morphology and np.any(anomaly_mask):
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(anomaly_mask.astype(np.uint8), cv2.MORPH_OPEN, k_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
        anomaly_mask = cleaned.astype(bool)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        anomaly_mask.astype(np.uint8), connectivity=8
    )

    extracted_regions: List[DeviationRegion] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx_px = float(centroids[label_id, 0])
        cy_px = float(centroids[label_id, 1])

        comp_mask = labels == label_id
        comp_devs = deviation_map[comp_mask]
        max_abs_dev = float(np.max(np.abs(comp_devs)))
        mean_dev = float(np.mean(comp_devs))
        region_sign = "depression" if mean_dev >= 0 else "bulge"

        # Centroid in Three.js viewer frame:
        # Camera frame: (+X right, +Y down, +Z forward)
        # Viewer frame: (+X right, +Y up, +Z towards viewer) -> (X_cam, -Y_cam, -Z_cam)
        comp_u = u_coords[comp_mask]
        comp_v = v_coords[comp_mask]
        comp_z = z_grid[comp_mask]
        comp_x_cam = (comp_u - cx) * comp_z / fx
        comp_y_cam = (comp_v - cy) * comp_z / fy

        c3d_x_viewer = float(np.mean(comp_x_cam))
        c3d_y_viewer = float(np.mean(-comp_y_cam))
        c3d_z_viewer = float(np.mean(-comp_z))

        region = DeviationRegion(
            id=len(extracted_regions) + 1,
            bounding_box=(x, y, width, height),
            area_px=area,
            max_abs_deviation_mm=max_abs_dev,
            mean_deviation_mm=mean_dev,
            centroid_px=(cx_px, cy_px),
            centroid_3d=(c3d_x_viewer, c3d_y_viewer, c3d_z_viewer),
            sign=region_sign,
        )
        extracted_regions.append(region)

    # Rank regions by max_abs_deviation descending and retain up to max_regions
    extracted_regions.sort(key=lambda r: r.max_abs_deviation_mm, reverse=True)
    regions = extracted_regions[:max_regions]
    # Re-assign 1-based sequential ids
    for idx, r in enumerate(regions):
        r["id"] = idx + 1
        r.id = idx + 1

    # 8. Compute Summary Metrics (10 required keys from master spec)
    mean_dev_mm = float(np.mean(valid_deviations))
    max_abs_dev_mm = float(np.max(np.abs(valid_deviations)))
    rmse_mm = float(np.sqrt(np.mean(valid_deviations ** 2)))
    max_depression_mm = float(np.max(valid_deviations)) if np.any(valid_deviations > 0) else 0.0
    max_bulge_mm = float(np.max(-valid_deviations)) if np.any(valid_deviations < 0) else 0.0
    inliers_count = int(np.sum(np.abs(valid_deviations) <= threshold_mm))
    inlier_fraction = inliers_count / max(1, valid_pixel_count)

    summary = {
        "rmse_mm": round(rmse_mm, 2),
        "max_depression_mm": round(max_depression_mm, 2),
        "max_bulge_mm": round(max_bulge_mm, 2),
        "sigma_mm": round(noise_sigma_mm, 2),
        "threshold_mm": round(threshold_mm, 2),
        "inlier_fraction": round(inlier_fraction, 4),
        "region_count": len(regions),
        "plane_normal": [round(nx, 4), round(ny, 4), round(nz, 4)],
        "plane_offset_mm": round(d_plane, 2),
        "valid_pixel_fraction": round(valid_fraction, 4),
    }

    # 9. Generate Diverging Colormap Heatmap with symmetric clipping at ±3T
    heatmap_bgr = create_deviation_heatmap(
        deviation_map=deviation_map,
        valid_mask=valid_mask,
        clip_range_mm=max(threshold_mm * 3.0, 1e-3),
    )

    return DeviationAnalysisResult(
        deviation_map=deviation_map,
        valid_mask=valid_mask,
        regions=regions,
        summary=summary,
        plane_params=(nx, ny, nz, d_plane),
        noise_sigma_mm=noise_sigma_mm,
        threshold_mm=threshold_mm,
        mean_deviation_mm=mean_dev_mm,
        max_abs_deviation_mm=max_abs_dev_mm,
        rmse_mm=rmse_mm,
        valid_pixel_count=valid_pixel_count,
        heatmap_bgr=heatmap_bgr,
    )


# Alias function name matching both spec and legacy callers
compute_surface_deviations = analyze_deviations
