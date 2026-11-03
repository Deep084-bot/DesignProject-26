"""
Tests for Surface Deviation Analysis (Stage 3).

Verifies all Milestone M1 requirements:
1. Synthetic depression on tilted plane: detected, bbox within ±5 px, max deviation within 15% of 20 mm, sign='depression'.
2. Synthetic bulge on tilted plane: detected, sign='bulge', max deviation within 15% of 20 mm.
3. Flat plane gives zero anomaly regions (region_count == 0).
4. Gaussian disparity noise of std 0.3 px on flat surface gives 0 detected regions.
5. Mostly invalid disparity (< 10% valid pixels) raises descriptive ValueError.
6. Pipeline integration: creates deviation_map.npy, deviation_heatmap.png, deviations.json; API response contains deviation_summary and deviation_heatmap_url; findings remain empty [].
7. Invalid pixels remain NaN in the saved deviation map.
8. Capped at 50 anomaly regions, ranked by maximum absolute deviation descending.
9. Summary metrics contain all 10 required keys and are JSON-serializable with finite values.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.pipeline.deviations import (
    analyze_deviations,
    compute_surface_deviations,
    fit_reference_plane,
    create_deviation_heatmap,
    DeviationRegion,
)
from app.pipeline.runner import run_pipeline
from app.models.calibration import RectifiedStereoCalibration


def test_tilted_plane_with_known_dent_detected():
    """
    Test 1: Synthetic depression.
    Constructs a synthetic tilted plane with a known 20 mm depression.
    Verifies:
    - Region is detected.
    - Bounding box is within ±5 pixels of expected location.
    - Maximum absolute deviation is within 15% of 20 mm (17.0 mm to 23.0 mm).
    - Sign is 'depression'.
    - Centroid_3d is in Three.js viewer frame (Z negative).
    """
    height, width = 200, 200
    fx, fy = 500.0, 500.0
    cx, cy = 100.0, 100.0

    # 1. Base synthetic tilted plane: 0.1 * X - 0.05 * Y - Z + 1000 = 0
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    denom = 1.0 - 0.1 * (u - cx) / fx + 0.05 * (v - cy) / fy
    z_plane = 1000.0 / denom

    # 2. Add a known 20 mm depression (farther from camera, deeper)
    dent_x, dent_y = 75, 80
    dent_w, dent_h = 50, 40
    z_with_dent = z_plane.copy()
    z_with_dent[dent_y : dent_y + dent_h, dent_x : dent_x + dent_w] += 20.0

    # 3. Run deviation analysis
    result = analyze_deviations(
        depth_grid_mm=z_with_dent,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k=3.0,
        min_mm=5.0,
        min_area_px=200,
    )

    # 4. Verify results
    assert len(result.regions) == 1, f"Expected 1 region, found {len(result.regions)}"
    region = result.regions[0]

    # Verify bounding box within ±5 pixels of expected location
    rx, ry, rw, rh = region.bounding_box
    assert abs(rx - dent_x) <= 5
    assert abs(ry - dent_y) <= 5
    assert abs(rw - dent_w) <= 5
    assert abs(rh - dent_h) <= 5

    # Verify dict and attribute access parity
    assert region.bbox_px["x"] == rx
    assert region.bbox_px["width"] == rw

    # Verify max deviation is within 15% of 20 mm (17.0 mm to 23.0 mm)
    assert 17.0 <= region.max_abs_deviation_mm <= 23.0
    assert abs(region.mean_deviation_mm - 20.0) < 3.0

    # Verify sign is 'depression'
    assert region.sign == "depression"
    assert result.summary["max_depression_mm"] >= 17.0

    # Verify 3D centroid in Three.js viewer frame (Z negative)
    c3d = region.centroid_3d_viewer
    assert len(c3d) == 3
    assert c3d[2] < 0, f"Expected negative Z in Three.js frame, got {c3d[2]}"


def test_tilted_plane_with_known_bulge_detected():
    """
    Test 2: Synthetic bulge.
    Constructs a synthetic tilted plane with a known 20 mm outward bulge.
    Verifies:
    - Region is detected.
    - Sign is 'bulge'.
    - Maximum absolute deviation is within 15% of 20 mm.
    """
    height, width = 200, 200
    fx, fy = 500.0, 500.0
    cx, cy = 100.0, 100.0

    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    denom = 1.0 - 0.1 * (u - cx) / fx + 0.05 * (v - cy) / fy
    z_plane = 1000.0 / denom

    # Outward bulge (closer to camera)
    bulge_x, bulge_y, bulge_w, bulge_h = 70, 75, 45, 35
    z_bulge = z_plane.copy()
    z_bulge[bulge_y : bulge_y + bulge_h, bulge_x : bulge_x + bulge_w] -= 20.0

    result = analyze_deviations(
        depth_grid_mm=z_bulge,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k=3.0,
        min_mm=5.0,
        min_area_px=200,
    )

    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.sign == "bulge"
    assert 17.0 <= region.max_abs_deviation_mm <= 23.0
    assert result.summary["max_bulge_mm"] >= 17.0
    assert abs(region.bbox_px["x"] - bulge_x) <= 5
    assert abs(region.bbox_px["y"] - bulge_y) <= 5


def test_flat_plane_gives_no_regions():
    """
    Test 3: Flat surface.
    Verifies that an unperturbed flat plane produces region_count == 0.
    """
    height, width = 150, 150
    fx, fy = 600.0, 600.0
    cx, cy = 75.0, 75.0

    z_flat = np.full((height, width), 1200.0, dtype=np.float32)

    result = analyze_deviations(
        depth_grid_mm=z_flat,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k=3.0,
        min_mm=5.0,
        min_area_px=200,
    )

    assert len(result.regions) == 0
    assert result.summary["region_count"] == 0
    assert result.max_abs_deviation_mm < 1.0


def test_gaussian_disparity_noise_gives_no_regions():
    """
    Test 4: Gaussian disparity noise.
    Flat surface with Gaussian disparity noise (std 0.3 px) must produce
    zero detected regions at default thresholds.
    """
    height, width = 120, 120
    fx, fy = 500.0, 500.0
    baseline_mm = 120.0

    # Nominal disparity = 30.0 px (depth = 2000 mm)
    disp = np.full((height, width), 30.0, dtype=np.float32)
    rng = np.random.RandomState(42)
    disp += rng.normal(0.0, 0.3, (height, width)).astype(np.float32)

    result = analyze_deviations(
        disparity=disp,
        baseline_mm=baseline_mm,
        fx=fx,
        fy=fy,
        k=3.0,
        min_mm=5.0,
        min_area_px=200,
    )

    assert len(result.regions) == 0
    assert result.summary["region_count"] == 0


def test_mostly_invalid_disparity_raises_value_error():
    """
    Test 5: Mostly invalid disparity.
    Disparity map with fewer than 10% valid pixels must raise a clear ValueError.
    """
    height, width = 100, 100
    disp = np.full((height, width), np.nan, dtype=np.float32)
    # Only 50 valid pixels (0.5% of 10,000 total pixels)
    disp[45:50, 45:55] = 25.0

    with pytest.raises(ValueError, match="At least 10% valid pixels required"):
        analyze_deviations(
            disparity=disp,
            baseline_mm=100.0,
            fx=400.0,
            fy=400.0,
        )


def test_disparity_input_mode_and_artifacts(tmp_path: Path):
    """
    Verifies disparity input mode and output artifact creation:
    deviation_map.npy, deviation_heatmap.png, and deviations.json.
    """
    height, width = 100, 100
    fx, fy = 400.0, 400.0
    baseline_mm = 100.0

    # Depth = 1000 mm -> disparity = 40 px
    disp = np.full((height, width), 40.0, dtype=np.float32)
    # Dent: depth = 1025 mm -> disparity = 40000 / 1025 ~ 39.02 px
    disp[30:60, 30:70] = (fx * baseline_mm) / 1025.0

    res = analyze_deviations(
        disparity=disp,
        baseline_mm=baseline_mm,
        fx=fx,
        fy=fy,
        min_area_px=200,
    )

    assert len(res.regions) == 1
    reg = res.regions[0]
    assert reg.sign == "depression"
    assert abs(reg.max_abs_deviation_mm - 25.0) < 1.5

    # Verify saved artifacts
    map_path = tmp_path / "deviation_map.npy"
    heatmap_path = tmp_path / "deviation_heatmap.png"
    json_path = tmp_path / "deviations.json"

    np.save(str(map_path), res.deviation_map)
    cv2.imwrite(str(heatmap_path), res.heatmap_bgr)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": res.summary, "regions": [dict(r) for r in res.regions]}, f)

    assert map_path.exists()
    loaded_map = np.load(str(map_path))
    assert loaded_map.shape == (height, width)
    assert loaded_map.dtype == np.float32

    assert heatmap_path.exists()
    loaded_img = cv2.imread(str(heatmap_path))
    assert loaded_img.shape == (height, width, 3)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert "summary" in loaded_json
    assert "regions" in loaded_json


def test_invalid_pixels_remain_nan_in_saved_map():
    """Verifies that invalid pixels remain float32 NaN in the deviation map."""
    height, width = 100, 100
    disp = np.full((height, width), 30.0, dtype=np.float32)
    # Invalidate a 20x20 block with NaNs and another block with non-positive disparity
    disp[10:30, 10:30] = np.nan
    disp[40:50, 40:50] = 0.1  # below min_disparity=0.5

    res = analyze_deviations(
        disparity=disp,
        baseline_mm=100.0,
        fx=400.0,
        fy=400.0,
        min_disparity=0.5,
    )

    assert np.isnan(res.deviation_map[15, 15])
    assert np.isnan(res.deviation_map[45, 45])
    assert not np.isnan(res.deviation_map[80, 80])
    assert res.deviation_map.dtype == np.float32


def test_region_count_capped_at_max_regions():
    """Verifies that extracted regions are capped at max_regions and sorted descending."""
    height, width = 300, 300
    z_grid = np.full((height, width), 1000.0, dtype=np.float32)

    # Place 60 distinct small indentations separated by background
    rng = np.random.RandomState(42)
    depths = []
    idx = 0
    for row in range(8):
        for col in range(8):
            if idx >= 60:
                break
            y = 15 + row * 35
            x = 15 + col * 35
            dev = float(10.0 + idx * 0.5)
            z_grid[y : y + 15, x : x + 15] += dev
            depths.append(dev)
            idx += 1

    res = analyze_deviations(
        depth_grid_mm=z_grid,
        fx=500.0,
        fy=500.0,
        k=3.0,
        min_mm=5.0,
        min_area_px=10,
        apply_morphology=False,
        max_regions=50,
    )

    assert len(res.regions) == 50
    assert res.summary["region_count"] == 50

    # Must be ranked by max_abs_deviation_mm descending
    for i in range(len(res.regions) - 1):
        assert res.regions[i].max_abs_deviation_mm >= res.regions[i + 1].max_abs_deviation_mm


def test_summary_values_finite_and_serializable():
    """Verifies that deviation summary metrics contain all 10 required keys and are serializable."""
    height, width = 100, 100
    disp = np.full((height, width), 35.0, dtype=np.float32)

    res = analyze_deviations(
        disparity=disp,
        baseline_mm=100.0,
        fx=400.0,
        fy=400.0,
    )

    required_keys = [
        "rmse_mm",
        "max_depression_mm",
        "max_bulge_mm",
        "sigma_mm",
        "threshold_mm",
        "inlier_fraction",
        "region_count",
        "plane_normal",
        "plane_offset_mm",
        "valid_pixel_fraction",
    ]

    for k in required_keys:
        assert k in res.summary, f"Missing required summary key '{k}'"

    serialized = json.dumps(res.summary)
    deserialized = json.loads(serialized)
    for k in required_keys:
        val = deserialized[k]
        if isinstance(val, (int, float)):
            assert np.isfinite(val)
        elif isinstance(val, list):
            assert all(np.isfinite(x) for x in val)


def test_pipeline_runner_executes_deviations_and_keeps_findings_empty(
    synthetic_stereo_pair, tmp_path: Path
):
    """
    Test 6a: Full pipeline runner execution.
    Verifies that stage 3 runs, deviation artifacts exist on disk, and findings remain [].
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

    steps_recorded = []

    def on_step(idx, step_id, status, msg):
        steps_recorded.append((idx, step_id, status))

    job_dir = tmp_path / "job_run"
    result = run_pipeline(
        job_dir=job_dir,
        left_image_path=left_path,
        right_image_path=right_path,
        calibration=calib,
        on_step_change=on_step,
    )

    assert result.success is True
    assert result.point_count > 0
    assert result.point_cloud_path.exists()

    # Stage 3 artifacts must exist
    assert result.deviation_map_path is not None
    assert result.deviation_map_path.exists()
    assert result.deviation_heatmap_path is not None
    assert result.deviation_heatmap_path.exists()
    assert (job_dir / "deviations.json").exists()

    # Check deviation summary on execution result
    assert result.deviation_summary is not None
    assert "rmse_mm" in result.deviation_summary
    assert "region_count" in result.deviation_summary

    # Verify stage progression includes deviations
    step_ids = [s[1] for s in steps_recorded if s[2] == "completed"]
    assert "deviations" in step_ids
    assert "damage" in step_ids
    assert "results" in step_ids


def test_api_response_contains_deviation_contracts(
    test_client: TestClient,
    synthetic_stereo_pair,
    valid_rectified_calib_json,
):
    """
    Test 6b: API response verification.
    Submits a valid inspection and checks that GET /api/inspections/{id}
    exposes deviation_summary and deviation_heatmap_url, while findings == [].
    """
    import time

    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
        "calibration_file": ("calibration.json", valid_rectified_calib_json.encode("utf-8"), "application/json"),
    }
    data = {
        "name": "Deviation Contract API Verification",
        "surface_type": "Retaining wall",
    }

    res = test_client.post("/api/inspections", data=data, files=files)
    assert res.status_code == 201
    inspection_id = res.json()["id"]

    # Poll until completed
    max_wait = 15
    start = time.time()
    final_data = None
    while time.time() - start < max_wait:
        poll = test_client.get(f"/api/inspections/{inspection_id}").json()
        if poll["status"] in ["completed", "failed"]:
            final_data = poll
            break
        time.sleep(0.3)

    assert final_data is not None
    assert final_data["status"] == "completed"

    # Master specification contract: deviation_summary and deviation_heatmap_url
    assert "deviation_summary" in final_data
    assert final_data["deviation_summary"] is not None
    assert "rmse_mm" in final_data["deviation_summary"]
    assert "threshold_mm" in final_data["deviation_summary"]
    assert "region_count" in final_data["deviation_summary"]

    assert "deviation_heatmap_url" in final_data
    assert final_data["deviation_heatmap_url"] == f"/api/inspections/{inspection_id}/artifacts/deviation_heatmap.png"

    # Verify findings remain empty
    assert final_data["findings"] == []

    # Heatmap artifact must be downloadable
    art_res = test_client.get(final_data["deviation_heatmap_url"])
    assert art_res.status_code == 200
    assert art_res.headers["content-type"] == "image/png"
    assert len(art_res.content) > 0


def test_invalid_input_fails_cleanly():
    """Verifies that malformed inputs raise descriptive ValueErrors cleanly."""
    # 1. Neither disparity nor depth provided
    with pytest.raises(ValueError, match="Either 'disparity' or 'depth_grid_mm'"):
        analyze_deviations()

    # 2. Non-positive focal length
    with pytest.raises(ValueError, match="Focal length must be strictly positive"):
        analyze_deviations(depth_grid_mm=np.ones((50, 50)), fx=-100.0)

    # 3. Non-positive baseline
    with pytest.raises(ValueError, match="Stereo baseline must be strictly positive"):
        analyze_deviations(disparity=np.ones((50, 50)), baseline_mm=0.0)

    # 4. All NaNs (zero valid pixels)
    all_nan = np.full((50, 50), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Insufficient valid 3D depth pixels"):
        analyze_deviations(depth_grid_mm=all_nan)

    # 5. Wrong dimensions
    with pytest.raises(ValueError, match="must be a 2D array"):
        analyze_deviations(depth_grid_mm=np.ones((10, 10, 3)))
