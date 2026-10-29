"""
Integration tests for FastAPI inspection endpoints and artifact retrieval.
"""

import time
import pytest
from fastapi.testclient import TestClient


def test_create_and_process_inspection_end_to_end(
    test_client: TestClient,
    synthetic_stereo_pair,
    valid_rectified_calib_json,
):
    """
    Submits a valid stereo pair and calibration file.
    Polls status until completed and verifies pointcloud.ply artifact is served.
    """
    # 1. Post new inspection
    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
        "calibration_file": ("calibration.json", valid_rectified_calib_json.encode("utf-8"), "application/json"),
    }
    data = {
        "name": "End-to-End Pipeline Verification",
        "surface_type": "Concrete surface",
        "reconstruction_method": "Automatic surface reconstruction",
    }

    res = test_client.post("/api/inspections", data=data, files=files)
    assert res.status_code == 201
    created = res.json()
    inspection_id = created["id"]
    assert created["name"] == "End-to-End Pipeline Verification"
    assert created["status"] in ["queued", "processing"]
    assert len(created["steps"]) == 6

    # 2. Poll until completed or timeout
    max_wait_seconds = 10
    start = time.time()
    final_inspection = None

    while time.time() - start < max_wait_seconds:
        poll_res = test_client.get(f"/api/inspections/{inspection_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        if poll_data["status"] in ["completed", "failed"]:
            final_inspection = poll_data
            break
        time.sleep(0.3)

    assert final_inspection is not None, "Pipeline execution timed out."
    assert final_inspection["status"] == "completed", f"Failed with: {final_inspection.get('failure_message')}"
    assert final_inspection["point_count"] > 0
    assert final_inspection["point_cloud_url"] is not None
    assert final_inspection["mesh_url"] is not None
    assert final_inspection["triangle_count"] is not None and final_inspection["triangle_count"] > 0
    assert final_inspection["mesh_status"] == "completed"
    assert final_inspection["findings"] == []  # Milestone 1: no fabricated findings

    # 3. Retrieve pointcloud.ply artifact
    artifact_res = test_client.get(f"/api/inspections/{inspection_id}/artifacts/pointcloud.ply")
    assert artifact_res.status_code == 200
    assert b"format binary_little_endian 1.0" in artifact_res.content

    # 4. Retrieve mesh.ply artifact
    mesh_res = test_client.get(f"/api/inspections/{inspection_id}/artifacts/mesh.ply")
    assert mesh_res.status_code == 200
    assert len(mesh_res.content) > 0

    # 5. Retrieve disparity_vis.png artifact
    vis_res = test_client.get(f"/api/inspections/{inspection_id}/artifacts/disparity_vis.png")
    assert vis_res.status_code == 200
    assert vis_res.headers["content-type"] == "image/png"


def test_inspection_with_mesh_failure_preserves_point_cloud(
    monkeypatch,
    test_client: TestClient,
    synthetic_stereo_pair,
    valid_rectified_calib_json,
):
    """
    Verifies that when surface mesh generation encounters an exception,
    the inspection completes with status 'completed', pointcloud.ply is preserved,
    mesh_status is 'failed', and GET /artifacts/mesh.ply yields 404.
    """
    def mock_failing_mesh(*args, **kwargs):
        raise RuntimeError("Injected Poisson failure for API testing")

    import app.pipeline.runner as runner_mod
    monkeypatch.setattr(runner_mod, "reconstruct_surface_mesh", mock_failing_mesh)

    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
        "calibration_file": ("calibration.json", valid_rectified_calib_json.encode("utf-8"), "application/json"),
    }
    data = {
        "name": "Mesh Failure Degradation Test",
        "surface_type": "Concrete surface",
    }

    res = test_client.post("/api/inspections", data=data, files=files)
    assert res.status_code == 201
    inspection_id = res.json()["id"]

    # Poll until completed
    max_wait = 10
    start = time.time()
    final_data = None
    while time.time() - start < max_wait:
        poll_res = test_client.get(f"/api/inspections/{inspection_id}")
        if poll_res.json()["status"] in ["completed", "failed"]:
            final_data = poll_res.json()
            break
        time.sleep(0.3)

    assert final_data is not None
    assert final_data["status"] == "completed"
    assert final_data["point_count"] > 0
    assert final_data["point_cloud_url"] is not None
    assert final_data["mesh_url"] is None
    assert final_data["triangle_count"] is None
    assert final_data["mesh_status"] == "failed"
    assert "Injected Poisson failure" in final_data["mesh_error"]

    # Point cloud must be downloadable
    pc_res = test_client.get(f"/api/inspections/{inspection_id}/artifacts/pointcloud.ply")
    assert pc_res.status_code == 200

    # Mesh artifact must return 404
    mesh_res = test_client.get(f"/api/inspections/{inspection_id}/artifacts/mesh.ply")
    assert mesh_res.status_code == 404


def test_upload_missing_calibration(test_client: TestClient, synthetic_stereo_pair):
    """Rejects submission when calibration is omitted."""
    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
    }
    data = {"name": "No Calibration"}
    res = test_client.post("/api/inspections", data=data, files=files)
    assert res.status_code == 400
    assert "calibration" in res.json()["detail"].lower()


def test_upload_invalid_calibration_json(test_client: TestClient, synthetic_stereo_pair):
    """Rejects malformed JSON calibration file."""
    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
        "calibration_file": ("calibration.json", b"{broken json: true", "application/json"),
    }
    res = test_client.post("/api/inspections", files=files)
    assert res.status_code == 400
    assert "Invalid calibration JSON" in res.json()["detail"]


def test_upload_insufficient_calibration(test_client: TestClient, synthetic_stereo_pair):
    """Rejects calibration that lacks baseline or focal length."""
    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", synthetic_stereo_pair["right_bytes"], "image/png"),
        "calibration_file": ("calibration.json", b'{"reprojection_error": 0.1}', "application/json"),
    }
    res = test_client.post("/api/inspections", files=files)
    assert res.status_code == 400
    assert "Insufficient" in res.json()["detail"]


def test_upload_corrupt_image(test_client: TestClient, valid_rectified_calib_json):
    """Rejects unreadable image bytes."""
    files = {
        "left_image": ("left.png", b"not-a-valid-image", "image/png"),
        "right_image": ("right.png", b"not-a-valid-image", "image/png"),
        "calibration_file": ("calibration.json", valid_rectified_calib_json.encode("utf-8"), "application/json"),
    }
    res = test_client.post("/api/inspections", files=files)
    assert res.status_code == 400
    assert "could not be decoded" in res.json()["detail"].lower()


def test_upload_mismatched_dimensions(
    test_client: TestClient,
    synthetic_stereo_pair,
    valid_rectified_calib_json,
):
    """Rejects left and right images with unequal dimensions."""
    import cv2
    smaller = cv2.resize(synthetic_stereo_pair["left_bgr"], (160, 120))
    _, smaller_png = cv2.imencode(".png", smaller)

    files = {
        "left_image": ("left.png", synthetic_stereo_pair["left_bytes"], "image/png"),
        "right_image": ("right.png", smaller_png.tobytes(), "image/png"),
        "calibration_file": ("calibration.json", valid_rectified_calib_json.encode("utf-8"), "application/json"),
    }
    res = test_client.post("/api/inspections", files=files)
    assert res.status_code == 400
    assert "mismatched image dimensions" in res.json()["detail"].lower()


def test_get_nonexistent_inspection(test_client: TestClient):
    """Returns 404 for unknown ID."""
    res = test_client.get("/api/inspections/NON_EXISTENT_ID")
    assert res.status_code == 404


def test_get_nonexistent_artifact(test_client: TestClient):
    """Returns 404 for unknown artifact."""
    res = test_client.get("/api/inspections/NON_EXISTENT_ID/artifacts/pointcloud.ply")
    assert res.status_code == 404


def test_artifact_path_traversal_prevention(test_client: TestClient):
    """Disallows path traversal attacks like ../../etc/passwd."""
    res = test_client.get("/api/inspections/SOME_ID/artifacts/..%2F..%2Fetc%2Fpasswd")
    assert res.status_code in [404, 400]
