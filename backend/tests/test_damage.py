"""
Tests for Damage Detection and Crack Classification (Stage 4).

Covers all 20 Milestone M2 requirements:
1. BGR-to-RGB preprocessing and ImageNet normalization.
2. Correct two-class softmax interpretation.
3. Class 1 probability used as crack confidence.
4. Regular tile generation.
5. Edge reflect-padding.
6. Extra tiles around deviation regions.
7. Positive tile merging.
8. Correct union bounding boxes.
9. Potential crack when classifier group overlaps deviation (Case A).
10. Visual-only potential crack when probability is at least 0.8 (Case B).
11. Minor anomaly when probability is between 0.5 and 0.8 (Case C).
12. Surface irregularity when a deviation region has no positive classifier group (Case D).
13. Correct finding IDs and severity ordering.
14. Correct image crops and artifact paths.
15. Correct 3D center estimation when depth is available.
16. Missing checkpoint fallback.
17. Invalid checkpoint fallback.
18. Classifier inference failure fallback.
19. classifier_status and classifier_error behavior.
20. Full pipeline integration.
21. Real checkpoint validation (optional integration test).
"""

from pathlib import Path
import json
import numpy as np
import cv2
import pytest
import torch
import torch.nn as nn

from app.models.inspection import Finding
from app.pipeline.damage import (
    CrackClassifier,
    detect_damage,
    generate_tiles,
    generate_tile_definitions,
    extract_tile_crop,
    classify_tile_batch,
    aggregate_damage_findings,
    DamageDetectionResult,
    merge_positive_tiles,
    estimate_box_center_3d,
    sort_findings,
    save_finding_crops,
    resolve_model_path,
)
from app.pipeline.runner import (
    run_pipeline,
    run_pipeline_stages_0_to_3,
    run_pipeline_stage_5,
    Stage0to3Result,
    PipelineExecutionResult,
)
from app.models.calibration import RectifiedStereoCalibration


class MockClassifierStub:
    """Deterministic classifier stub for isolated unit testing."""

    def __init__(self, prob_map=None, default_prob=0.0, status="completed", error=None):
        self.prob_map = prob_map or {}
        self.default_prob = default_prob
        self.status = status
        self.error = error
        self.device = torch.device("cpu")
        self.call_count = 0

    def predict_batch(self, tiles_bgr: list, batch_size: int = 16) -> np.ndarray:
        self.call_count += len(tiles_bgr)
        probs = []
        for tile in tiles_bgr:
            # Hash tile or check mean
            mean_val = float(np.mean(tile))
            p = self.prob_map.get(mean_val, self.default_prob)
            probs.append(p)
        return np.array(probs, dtype=np.float32)


# -----------------------------------------------------------------------------
# Test 1, 2, 3: Preprocessing, Softmax, Class 1 Confidence
# -----------------------------------------------------------------------------
def test_bgr_to_rgb_and_imagenet_normalization(monkeypatch):
    """Verifies BGR to RGB conversion, 224x224 resize, and ImageNet normalization."""
    captured_tensors = []

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 2)

        def forward(self, x):
            captured_tensors.append(x.clone())
            return torch.zeros((x.size(0), 2))

    dummy_model = DummyModel()
    clf = CrackClassifier.__new__(CrackClassifier)
    clf.model = dummy_model
    clf.device = torch.device("cpu")
    clf.status = "completed"
    clf.error = None

    # Pure Blue tile in BGR: [255, 0, 0]
    # In RGB: [0, 0, 255]
    # Expected ImageNet norm:
    # R: (0.0 - 0.485) / 0.229 = -2.1179
    # G: (0.0 - 0.456) / 0.224 = -2.0357
    # B: (1.0 - 0.406) / 0.225 = +2.6400
    tile_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tile_bgr[:, :, 0] = 255  # Blue channel

    probs = clf.predict_batch([tile_bgr])
    assert len(captured_tensors) == 1
    tensor = captured_tensors[0]

    assert tensor.shape == (1, 3, 224, 224)
    # Check channel means
    r_val = tensor[0, 0, 0, 0].item()
    g_val = tensor[0, 1, 0, 0].item()
    b_val = tensor[0, 2, 0, 0].item()

    assert abs(r_val - (-2.1179)) < 0.05
    assert abs(g_val - (-2.0357)) < 0.05
    assert abs(b_val - (2.6400)) < 0.05


def test_two_class_softmax_and_class_1_probability():
    """Verifies softmax calculation and that class 1 probability is returned."""
    class LogitModel(nn.Module):
        def forward(self, x):
            # Tile 0: strongly class 0; Tile 1: strongly class 1
            return torch.tensor([
                [10.0, -10.0],
                [-5.0, 5.0],
            ], dtype=torch.float32)

    clf = CrackClassifier.__new__(CrackClassifier)
    clf.model = LogitModel()
    clf.device = torch.device("cpu")
    clf.status = "completed"
    clf.error = None

    tiles = [np.zeros((50, 50, 3), dtype=np.uint8), np.ones((50, 50, 3), dtype=np.uint8)]
    probs = clf.predict_batch(tiles)

    assert len(probs) == 2
    assert probs[0] < 0.001  # Class 1 prob is ~0
    assert probs[1] > 0.999  # Class 1 prob is ~1


# -----------------------------------------------------------------------------
# Test 4, 5, 6: Tile Generation, Edge Padding, Region Extra Tiles
# -----------------------------------------------------------------------------
def test_regular_tile_generation_and_edge_padding():
    """Verifies regular grid generation with stride 256 and edge reflect padding."""
    # 720 x 1280 image:
    # 720: rows 0, 256, 512 (3 rows)
    # 1280: cols 0, 256, 512, 768, 1024 (5 cols)
    # Total grid tiles = 15
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    tiles = generate_tiles(img, deviation_regions=[], tile_size=256, stride=256)

    assert len(tiles) == 15
    for t in tiles:
        assert t["tile_bgr"].shape == (256, 256, 3)
        assert t["source"] == "grid"
        b = t["bbox"]
        assert b["x"] >= 0 and b["y"] >= 0
        assert b["x"] + b["width"] <= 1280
        assert b["y"] + b["height"] <= 720


def test_extra_tiles_around_deviation_regions():
    """Verifies that focused extra tiles are generated for deviation regions."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    regions = [
        {"x": 600, "y": 300, "width": 80, "height": 80},  # small region -> 1 centered tile
    ]
    tiles = generate_tiles(img, deviation_regions=regions, tile_size=256, stride=256)

    # 15 grid tiles + 1 region tile = 16 tiles
    assert len(tiles) == 16
    region_tiles = [t for t in tiles if t["source"] == "region"]
    assert len(region_tiles) == 1
    rt = region_tiles[0]
    assert rt["tile_bgr"].shape == (256, 256, 3)
    # Centered around (640, 340): x0 = 640 - 128 = 512, y0 = 340 - 128 = 212
    assert rt["bbox"]["x"] == 512
    assert rt["bbox"]["y"] == 212


# -----------------------------------------------------------------------------
# Test 7, 8: Positive Tile Merging & Union Bounding Box
# -----------------------------------------------------------------------------
def test_positive_tile_merging_and_union_bounding_box():
    """Verifies that adjacent or overlapping positive tiles merge into union boxes."""
    # Group A: two horizontally touching tiles
    t1 = {"bbox": {"x": 100, "y": 100, "width": 256, "height": 256}, "prob": 0.85, "region": None}
    t2 = {"bbox": {"x": 356, "y": 100, "width": 256, "height": 256}, "prob": 0.92, "region": None}
    # Group B: one isolated tile
    t3 = {"bbox": {"x": 800, "y": 400, "width": 256, "height": 256}, "prob": 0.70, "region": None}

    merged = merge_positive_tiles([t1, t2, t3])
    assert len(merged) == 2

    # Group 1 should be union of t1 and t2: x=100, y=100, width=512, height=256
    g1 = next(g for g in merged if g["bbox"]["x"] == 100)
    assert g1["bbox"]["width"] == 512
    assert g1["bbox"]["height"] == 256
    assert g1["confidence"] == 0.92

    g2 = next(g for g in merged if g["bbox"]["x"] == 800)
    assert g2["bbox"]["width"] == 256
    assert g2["confidence"] == 0.70


# -----------------------------------------------------------------------------
# Test 9, 10, 11, 12: Finding Generation Cases A, B, C, D
# -----------------------------------------------------------------------------
def test_finding_cases_a_b_c_d():
    """
    Verifies all 4 classification and correlation cases:
    - Case A: Classifier positive + deviation overlap -> potential_crack with geom dev
    - Case B: Classifier positive (P >= 0.8), no deviation -> potential_crack without geom dev
    - Case C: Classifier positive (0.5 <= P < 0.8), no deviation -> minor_anomaly
    - Case D: Deviation region without classifier evidence -> surface_irregularity
    """
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Deviation 1: at (100, 100) (will overlap Case A tile)
    # Deviation 2: at (900, 100) (will NOT have classifier match -> Case D)
    regions = [
        {
            "id": 1,
            "x": 100, "y": 100, "width": 80, "height": 80,
            "max_abs_deviation_mm": 22.5,
            "centroid_3d_viewer": [10.0, -20.0, -1500.0],
            "sign": "depression",
        },
        {
            "id": 2,
            "x": 900, "y": 100, "width": 80, "height": 80,
            "max_abs_deviation_mm": 16.0,
            "centroid_3d_viewer": [80.0, -20.0, -1550.0],
            "sign": "bulge",
        },
    ]

    class StubPredictor:
        status = "completed"
        error = None
        device = torch.device("cpu")

        def predict_batch(self, tiles_bgr, batch_size=16):
            probs = []
            for t in tiles_bgr:
                # Custom mock logic:
                probs.append(0.0)
            return np.array(probs, dtype=np.float32)

    # Construct custom mock tiles directly through detect_damage
    # by stubbing predict_batch on CrackClassifier
    class MockPredictor:
        status = "completed"
        error = None
        device = torch.device("cpu")

        def predict_batch(self, tiles_bgr, batch_size=16):
            # Return high prob for tile covering (100, 100) [Case A]
            # Return high prob (0.90) for tile covering (500, 500) [Case B]
            # Return moderate prob (0.65) for tile covering (500, 0) [Case C]
            # Else 0.0
            results = []
            for t in tiles_bgr:
                # Check mean value encoded in tile
                m = float(np.mean(t))
                if m == 10.0:
                    results.append(0.85)  # Case A
                elif m == 20.0:
                    results.append(0.92)  # Case B
                elif m == 30.0:
                    results.append(0.65)  # Case C
                else:
                    results.append(0.10)  # Negatives
            return np.array(results, dtype=np.float32)

    # Encode test markers into img:
    # Tile at (0, 0) -> mean 10.0 (overlaps region at 100, 100)
    img[0:256, 0:256] = 10
    # Tile at (512, 512) -> mean 20.0 (no region, P=0.92)
    img[512:720, 512:768] = 20
    # Tile at (0, 512) -> mean 30.0 (no region, P=0.65)
    img[0:256, 512:768] = 30

    res = detect_damage(
        rectified_left_bgr=img,
        deviation_regions=regions,
        classifier=MockPredictor(),
        threshold=0.5,
    )

    types = [f.type for f in res.findings]
    assert types.count("potential_crack") == 2  # Case A and Case B
    assert types.count("minor_anomaly") == 1    # Case C
    assert types.count("surface_irregularity") == 1  # Case D

    # Verify Case A attributes
    case_a = next(f for f in res.findings if f.geometric_deviation_mm == 22.5)
    assert case_a.type == "potential_crack"
    assert case_a.confidence == 0.85
    assert case_a.region_center_3d == [10.0, -20.0, -1500.0]

    # Verify Case B attributes
    case_b = next(f for f in res.findings if f.type == "potential_crack" and f.geometric_deviation_mm is None)
    assert case_b.confidence == 0.92

    # Verify Case C attributes
    case_c = next(f for f in res.findings if f.type == "minor_anomaly")
    assert case_c.confidence == 0.65

    # Verify Case D attributes
    case_d = next(f for f in res.findings if f.type == "surface_irregularity")
    assert case_d.confidence is None
    assert case_d.geometric_deviation_mm == 16.0


# -----------------------------------------------------------------------------
# Test 13: Finding IDs and Severity Ordering
# -----------------------------------------------------------------------------
def test_finding_ids_and_severity_ordering():
    """Verifies that findings are sorted strictly by severity and assigned sequential Finding IDs."""
    findings = [
        Finding(id="", type="minor_anomaly", confidence=0.7),
        Finding(id="", type="potential_crack", confidence=0.85, geometric_deviation_mm=10.0),
        Finding(id="", type="surface_irregularity", geometric_deviation_mm=25.0),
        Finding(id="", type="potential_crack", confidence=0.95, geometric_deviation_mm=20.0),
    ]

    sorted_f = sort_findings(findings)
    assert sorted_f[0].id == "Finding 01"
    assert sorted_f[0].type == "potential_crack"
    assert sorted_f[0].geometric_deviation_mm == 20.0  # highest dev among potential_cracks

    assert sorted_f[1].id == "Finding 02"
    assert sorted_f[1].type == "potential_crack"

    assert sorted_f[2].id == "Finding 03"
    assert sorted_f[2].type == "surface_irregularity"

    assert sorted_f[3].id == "Finding 04"
    assert sorted_f[3].type == "minor_anomaly"


# -----------------------------------------------------------------------------
# Test 14: Image Crops and Artifact Paths
# -----------------------------------------------------------------------------
def test_image_crops_and_artifact_paths(tmp_path: Path):
    """Verifies saving image crops for each finding and setting image_crop_url."""
    img = np.full((300, 300, 3), 120, dtype=np.uint8)
    # Draw a distinguishable rectangle
    cv2.rectangle(img, (50, 50), (120, 120), (0, 0, 255), -1)

    f = Finding(
        id="Finding 01",
        type="potential_crack",
        bounding_box_2d={"x": 50.0, "y": 50.0, "width": 70.0, "height": 70.0},
    )

    save_finding_crops([f], img, job_dir=tmp_path, inspection_id="INSP-TEST-001", margin_px=10)

    crop_file = tmp_path / "finding_01.png"
    assert crop_file.exists()
    saved_img = cv2.imread(str(crop_file))
    # Box: 50..120 with margin 10 -> (40..130) = 90x90
    assert saved_img.shape == (90, 90, 3)
    assert f.image_crop_url == "/api/inspections/INSP-TEST-001/artifacts/finding_01.png"


# -----------------------------------------------------------------------------
# Test 15: 3D Center Estimation with Depth
# -----------------------------------------------------------------------------
def test_3d_center_estimation_with_disparity():
    """Verifies 3D viewer coordinate estimation from median disparity inside bounding box."""
    H, W = 200, 200
    disp = np.full((H, W), 40.0, dtype=np.float32)  # 40 px disparity
    # Z = (fx * B) / d = (500 * 120) / 40 = 1500 mm
    intrinsics = {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 100.0,
        "cy": 100.0,
        "baseline_mm": 120.0,
    }
    # Box centered at (100, 100) -> u=100, v=100 -> x_cam = 0, y_cam = 0
    bbox = {"x": 80, "y": 80, "width": 40, "height": 40}
    c3d = estimate_box_center_3d(bbox, disp, intrinsics)

    assert c3d is not None
    assert abs(c3d[0] - 0.0) < 1.0
    assert abs(c3d[1] - 0.0) < 1.0
    # Viewer frame: Z negative towards viewer
    assert abs(c3d[2] - (-1500.0)) < 1.0


# -----------------------------------------------------------------------------
# Test 16, 17, 18, 19: Checkpoint Fallbacks and Status Behavior
# -----------------------------------------------------------------------------
def test_missing_checkpoint_fallback():
    """Verifies fallback when model weights checkpoint does not exist."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    regions = [{"x": 20, "y": 20, "width": 40, "height": 40, "max_abs_deviation_mm": 12.0, "sign": "dent"}]

    res = detect_damage(
        rectified_left_bgr=img,
        deviation_regions=regions,
        model_path="/path/does/not/exist/model.pth",
    )

    assert res.classifier_status == "unavailable"
    assert "not found" in (res.classifier_error or "").lower()
    # Deviation region preserved as surface_irregularity
    assert len(res.findings) == 1
    assert res.findings[0].type == "surface_irregularity"
    assert res.findings[0].confidence is None


def test_invalid_checkpoint_fallback(tmp_path: Path):
    """Verifies fallback when model weights checkpoint is corrupt or invalid."""
    corrupt_pth = tmp_path / "corrupt.pth"
    corrupt_pth.write_text("not a pytorch checkpoint")

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    regions = [{"x": 20, "y": 20, "width": 40, "height": 40, "max_abs_deviation_mm": 12.0, "sign": "dent"}]

    res = detect_damage(
        rectified_left_bgr=img,
        deviation_regions=regions,
        model_path=corrupt_pth,
    )

    assert res.classifier_status == "failed"
    assert res.classifier_error is not None
    assert len(res.findings) == 1
    assert res.findings[0].type == "surface_irregularity"


def test_inference_failure_fallback():
    """Verifies fallback when predict_batch raises a runtime exception."""
    class FailingClassifier:
        status = "completed"
        error = None
        device = torch.device("cpu")

        def predict_batch(self, tiles_bgr, batch_size=16):
            raise RuntimeError("Simulated GPU Out Of Memory")

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    regions = [{"x": 10, "y": 10, "width": 30, "height": 30, "max_abs_deviation_mm": 8.0, "sign": "depression"}]

    res = detect_damage(
        rectified_left_bgr=img,
        deviation_regions=regions,
        classifier=FailingClassifier(),
    )

    assert res.classifier_status == "failed"
    assert "Simulated GPU Out Of Memory" in (res.classifier_error or "")
    assert len(res.findings) == 1
    assert res.findings[0].type == "surface_irregularity"


# -----------------------------------------------------------------------------
# Test 20: Full Pipeline Integration with Damage Stage
# -----------------------------------------------------------------------------
def test_pipeline_runner_executes_damage_stage(synthetic_stereo_pair, tmp_path: Path):
    """Verifies end-to-end pipeline execution with Stage 4 damage classification."""
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
    assert result.classifier_status in ["completed", "unavailable"]
    assert isinstance(result.findings, list)

    completed_steps = [s[1] for s in steps_recorded if s[2] == "completed"]
    assert "damage" in completed_steps


# -----------------------------------------------------------------------------
# Test 21: Real Checkpoint Compatibility (Skipped if not present)
# -----------------------------------------------------------------------------
def test_real_checkpoint_efficientnet_b0_compatibility():
    """
    Validates the actual repository checkpoint against EfficientNet-B0 architecture:
    loads weights strictly, runs forward pass, verifies 2-class softmax.
    """
    ckpt_path = resolve_model_path()
    if not ckpt_path or not ckpt_path.exists():
        pytest.skip(f"Real checkpoint not found at '{ckpt_path}'. Skipping integration test.")

    clf = CrackClassifier(model_path=ckpt_path)
    assert clf.status == "completed", f"Failed to load checkpoint: {clf.error}"
    assert clf.classes == ["Uncracked", "Cracked"]

    dummy_tile = np.full((256, 256, 3), 128, dtype=np.uint8)
    probs = clf.predict_batch([dummy_tile])
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


# -----------------------------------------------------------------------------
# Test 22: M4b Phase 1 Modular Tile Definitions and Crop Extraction
# -----------------------------------------------------------------------------
def test_modular_tile_definitions_and_crop_extraction():
    """Validates deterministic tile definitions and extraction with reflect padding."""
    h, w = 300, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.arange(w) % 256

    dev_regions = [
        {"id": "dev_001", "bbox": [50, 60, 150, 180], "area_px": 12000, "deviation_type": "dent"}
    ]

    defs1 = generate_tile_definitions(img.shape, dev_regions, tile_size=256, stride=128)
    defs2 = generate_tile_definitions(img.shape, dev_regions, tile_size=256, stride=128)

    # 1. Determinism
    assert len(defs1) > 0
    assert defs1 == defs2

    # 2. Schema check
    for idx, tdef in enumerate(defs1):
        assert tdef["tile_index"] == idx
        assert "bbox" in tdef and len(tdef["bbox"]) == 4
        assert "crop_coords" in tdef and len(tdef["crop_coords"]) == 4
        assert tdef["source"] in ("grid", "region")
        assert "region" in tdef
        assert "is_small_image" in tdef
        assert tdef["is_small_image"] is False

    # 3. Extraction check
    for tdef in defs1:
        crop = extract_tile_crop(img, tdef, tile_size=256)
        assert crop.shape == (256, 256, 3)
        assert crop.dtype == np.uint8

    # 4. Small image case (< tile_size)
    small_img = np.zeros((100, 120, 3), dtype=np.uint8)
    small_defs = generate_tile_definitions(small_img.shape, [], tile_size=256)
    assert len(small_defs) == 1
    assert small_defs[0]["is_small_image"] is True
    small_crop = extract_tile_crop(small_img, small_defs[0], tile_size=256)
    assert small_crop.shape == (256, 256, 3)


# -----------------------------------------------------------------------------
# Test 23: M4b Phase 1 Modular Batch Classification and Finding Aggregation
# -----------------------------------------------------------------------------
def test_modular_classification_and_aggregation_matches_sequential(tmp_path: Path):
    """
    Verifies that the modular M4b execution path (definitions -> batch classification -> aggregation)
    yields identical findings and statistics to the sequential detect_damage() pipeline.
    """
    h, w = 320, 240
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    disparity = np.full((h, w), 25.0, dtype=np.float32)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 120.0, "cy": 160.0, "baseline_mm": 120.0}

    dev_regions = [
        {"id": "dev_001", "bbox": [30, 40, 100, 120], "area_px": 5600, "deviation_type": "dent"}
    ]

    class MockClassifier:
        def __init__(self):
            self.status = "completed"
            self.error = None
            self.device = "cpu"
            self.classes = ["Uncracked", "Cracked"]

        def predict_batch(self, tiles, batch_size=16):
            return [0.85 if idx % 2 == 0 else 0.15 for idx in range(len(tiles))]

    mock_clf = MockClassifier()

    # 1. Sequential run
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    seq_res = detect_damage(
        rectified_left_bgr=img,
        deviation_regions=dev_regions,
        disparity=disparity,
        intrinsics=intrinsics,
        tile=256,
        threshold=0.5,
        classifier=mock_clf,
        job_dir=seq_dir,
    )

    # 2. Modular run
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    tile_defs = generate_tile_definitions(img.shape, dev_regions, tile_size=256, stride=128)

    batch_size = 2
    all_probs = []
    for i in range(0, len(tile_defs), batch_size):
        batch = tile_defs[i:i + batch_size]
        probs = classify_tile_batch(img, batch, mock_clf, batch_size=batch_size)
        all_probs.extend(probs)

    mod_res = aggregate_damage_findings(
        deviation_regions=dev_regions,
        tile_defs=tile_defs,
        probabilities=all_probs,
        disparity=disparity,
        intrinsics=intrinsics,
        threshold=0.5,
        classifier_status=mock_clf.status,
        classifier_error=mock_clf.error,
        clf_device=mock_clf.device,
        job_dir=mod_dir,
        image_bgr=img,
    )

    # 3. Assert exact match
    assert len(seq_res.findings) == len(mod_res.findings)
    for sf, mf in zip(seq_res.findings, mod_res.findings):
        assert sf.id == mf.id
        assert sf.type == mf.type
        assert sf.bounding_box_2d == mf.bounding_box_2d
        if sf.confidence is not None:
            assert pytest.approx(sf.confidence, rel=1e-4) == mf.confidence
        else:
            assert mf.confidence is None
        assert sf.description == mf.description
        assert sf.geometric_deviation_mm == mf.geometric_deviation_mm
        assert sf.region_center_3d == mf.region_center_3d

    assert seq_res.stats == mod_res.stats
    assert seq_res.classifier_status == mod_res.classifier_status
    assert seq_res.classifier_error == mod_res.classifier_error


# -----------------------------------------------------------------------------
# Test 24: M4b Phase 1 Fallback Modes in Aggregation
# -----------------------------------------------------------------------------
def test_modular_aggregation_fallback_modes(tmp_path: Path):
    """Verifies that aggregate_damage_findings handles unavailable or failed classifier states."""
    h, w = 320, 240
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    disparity = np.full((h, w), 25.0, dtype=np.float32)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 120.0, "cy": 160.0, "baseline_mm": 120.0}

    dev_regions = [
        {"id": "dev_001", "bbox": [30, 40, 100, 120], "area_px": 5600, "deviation_type": "dent"}
    ]

    tile_defs = generate_tile_definitions(img.shape, dev_regions, tile_size=256, stride=128)

    # Case A: classifier unavailable
    res_unavail = aggregate_damage_findings(
        deviation_regions=dev_regions,
        tile_defs=tile_defs,
        probabilities=[],
        disparity=disparity,
        intrinsics=intrinsics,
        threshold=0.5,
        classifier_status="unavailable",
        classifier_error="Checkpoint missing",
        clf_device="cpu",
        job_dir=tmp_path / "unavail",
        image_bgr=img,
    )

    assert res_unavail.classifier_status == "unavailable"
    assert res_unavail.classifier_error == "Checkpoint missing"
    assert len(res_unavail.findings) == 1
    assert res_unavail.findings[0].type == "surface_irregularity"
    assert res_unavail.findings[0].confidence is None
    assert "classifier unavailable" in res_unavail.findings[0].description
    assert res_unavail.stats["tiles_evaluated"] == 0

    # Case B: classifier failed with inference error
    res_failed = aggregate_damage_findings(
        deviation_regions=dev_regions,
        tile_defs=tile_defs,
        probabilities=[],
        disparity=disparity,
        intrinsics=intrinsics,
        threshold=0.5,
        classifier_status="failed",
        classifier_error="Inference error: CUDA out of memory",
        clf_device="cpu",
        tiles_evaluated_count=len(tile_defs),
        job_dir=tmp_path / "failed",
        image_bgr=img,
    )

    assert res_failed.classifier_status == "failed"
    assert "Inference error" in (res_failed.classifier_error or "")
    assert len(res_failed.findings) == 1
    assert res_failed.findings[0].type == "surface_irregularity"
    assert res_failed.findings[0].confidence is None
    assert "inference failed" in res_failed.findings[0].description
    assert res_failed.stats["tiles_evaluated"] == len(tile_defs)


# -----------------------------------------------------------------------------
# Test 25: M4b Phase 1 Pipeline Stages 0-3 and Stage 5 Standalone Execution
# -----------------------------------------------------------------------------
def test_pipeline_stages_0_to_3_and_stage_5_standalone(synthetic_stereo_pair, tmp_path: Path):
    """
    Verifies that run_pipeline_stages_0_to_3 executes Stages 0-3 without invoking Stage 4,
    and run_pipeline_stage_5 completes results packaging independently.
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
    def on_step(idx, step_id, status, msg=None):
        steps_recorded.append((idx, step_id, status))

    job_dir = tmp_path / "job_stages_0_3"
    stage_0_3_res = run_pipeline_stages_0_to_3(
        job_dir=job_dir,
        left_image_path=left_path,
        right_image_path=right_path,
        calibration=calib,
        on_step_change=on_step,
    )

    # 1. Stages 0-3 executed successfully
    assert stage_0_3_res.success is True
    assert stage_0_3_res.point_count > 0
    assert stage_0_3_res.point_cloud_path is not None and stage_0_3_res.point_cloud_path.exists()
    assert stage_0_3_res.disparity_path is not None and stage_0_3_res.disparity_path.exists()
    assert stage_0_3_res.rectified_left_path is not None and stage_0_3_res.rectified_left_path.exists()
    assert stage_0_3_res.deviations_json_path is not None and stage_0_3_res.deviations_json_path.exists()
    assert stage_0_3_res.rectified_left is not None
    assert stage_0_3_res.disparity is not None
    assert stage_0_3_res.intrinsics is not None

    # Verify Stage 4 was NOT executed in steps_recorded
    stage_ids = [s[1] for s in steps_recorded]
    assert "prepare" in stage_ids
    assert "depth" in stage_ids
    assert "surface" in stage_ids
    assert "deviations" in stage_ids
    assert "damage" not in stage_ids
    assert "results" not in stage_ids

    # 2. Stage 5 packaging runs independently
    dummy_damage = DamageDetectionResult(
        findings=[],
        classifier_status="completed",
        classifier_error=None,
        stats={"tiles_total": 0, "tiles_evaluated": 0, "tiles_positive": 0},
    )

    final_res = run_pipeline_stage_5(
        job_dir=job_dir,
        stage_0_3_result=stage_0_3_res,
        damage_result=dummy_damage,
        on_step_change=on_step,
    )

    assert final_res.success is True
    assert final_res.point_count == stage_0_3_res.point_count
    assert final_res.findings == []
    assert final_res.classifier_status == "completed"
    assert "results" in [s[1] for s in steps_recorded]
