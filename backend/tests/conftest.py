"""
Pytest configuration and test fixtures for Surface stereo backend.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2
import tempfile
import shutil

# Ensure backend package is in python search path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.services.job_manager import JobManager, job_manager


@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def temp_job_manager(tmp_path):
    """Provides an isolated JobManager instance backed by a temporary directory."""
    mgr = JobManager(storage_dir=tmp_path / "storage" / "jobs", max_workers=2)
    return mgr


@pytest.fixture
def synthetic_stereo_pair():
    """
    Creates a synthetic stereo pair with rich texture and a known horizontal disparity.
    Left image has background at infinity (d=0) and foreground patch at disparity d=16 px.
    Right image has foreground patch shifted left by 16 px.
    """
    width, height = 320, 240
    np.random.seed(42)

    # Base high-frequency texture (concrete-like speckled noise)
    base_texture = np.random.randint(90, 180, (height, width), dtype=np.uint8)
    base_texture = cv2.GaussianBlur(base_texture, (3, 3), 0)

    left_gray = base_texture.copy()
    right_gray = base_texture.copy()

    # Foreground object: a textured square patch
    patch_w, patch_h = 140, 110
    y1, y2 = 65, 65 + patch_h
    x1, x2 = 120, 120 + patch_w
    disparity_px = 16

    patch_texture = np.random.randint(180, 240, (patch_h, patch_w), dtype=np.uint8)

    # Place in left image at (x1, y1)
    left_gray[y1:y2, x1:x2] = patch_texture

    # Place in right image shifted left by disparity_px: (x1 - 16, y1)
    right_gray[y1:y2, (x1 - disparity_px):(x2 - disparity_px)] = patch_texture

    # Convert to 3-channel BGR images
    left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    # Encode as PNG bytes
    _, left_png = cv2.imencode(".png", left_bgr)
    _, right_png = cv2.imencode(".png", right_bgr)

    return {
        "left_bgr": left_bgr,
        "right_bgr": right_bgr,
        "left_bytes": left_png.tobytes(),
        "right_bytes": right_png.tobytes(),
        "width": width,
        "height": height,
        "disparity_px": disparity_px,
    }


@pytest.fixture
def valid_rectified_calib_json():
    """Standard valid Case A calibration JSON string."""
    return """{
        "camera_model": "pinhole_stereo",
        "baseline_distance_mm": 120.0,
        "focal_length_px": 500.0,
        "image_width": 320,
        "image_height": 240,
        "principal_point": [160.0, 120.0]
    }"""


@pytest.fixture
def valid_full_rig_calib_json():
    """Standard valid Case B calibration JSON string."""
    return """{
        "mode": "full_stereo_rig",
        "left_camera_matrix": [
            [500.0, 0.0, 160.0],
            [0.0, 500.0, 120.0],
            [0.0, 0.0, 1.0]
        ],
        "right_camera_matrix": [
            [500.0, 0.0, 160.0],
            [0.0, 500.0, 120.0],
            [0.0, 0.0, 1.0]
        ],
        "left_distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
        "right_distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation_vector": [-120.0, 0.0, 0.0],
        "image_width": 320,
        "image_height": 240
    }"""
