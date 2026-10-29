# Surface — 3D Structural Surface Inspection: Python Processing Backend

This is the Python photogrammetry processing backend for Surface. It accepts calibrated stereo image pairs, performs validation, computes dense epipolar disparity using OpenCV Semi-Global Block Matching (StereoSGBM), triangulates metric 3D point clouds in millimeters, and serves binary PLY artifacts to the Three.js frontend.

---

## Architecture & Technology Stack

- **Framework**: FastAPI (asynchronous ASGI web server)
- **Server**: Uvicorn
- **Image & Stereo Processing**: OpenCV (`opencv-python-headless`)
- **Surface Mesh Reconstruction**: Open3D (`open3d`) & SciPy (`scipy`) — Poisson Surface Reconstruction (`PointCloudToMesh`)
- **Numerical Computations**: NumPy
- **Validation**: Pydantic v2
- **Testing**: Pytest & HTTPX TestClient

---

## Installation & Run Guide (macOS)

### 1. Prerequisites
- macOS (Apple Silicon or Intel)
- Python 3.10, 3.11, or 3.12
- Node.js 18+ (for frontend)

### 2. Set Up Python Virtual Environment

```bash
cd backend

# Create virtual environment using Python 3.12
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Backend Server

```bash
# From workspace root
./backend/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend --reload
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

### 4. Run the Test Suite

```bash
./backend/venv/bin/pytest backend/tests/ -v
```

---

## Calibration Contract & Assumptions

Stereo reconstruction cannot estimate metric 3D depth from arbitrary uncalibrated photos without knowing the physical camera baseline and optical parameters. The backend strictly requires and validates one of two calibration contracts:

### Case A: Already-Rectified Stereo Pairs (`mode: "rectified"`)
Used when images have already been rectified so that epipolar scanlines are strictly horizontal and parallel ($y_L = y_R$).
Required JSON fields:
- `baseline_distance_mm` (float > 0): Distance between camera optical centers in millimeters.
- `focal_length_px` (float > 0): Camera focal length in pixels ($f_x$).
- `principal_point` (optional `[cx, cy]` in pixels): Optical center. Defaults to image center `[W/2, H/2]` if omitted.
- `image_width`, `image_height` (optional integers): Resolution for dimension sanity check.

Example:
```json
{
  "camera_model": "pinhole_stereo",
  "baseline_distance_mm": 120.5,
  "focal_length_px": 900.0,
  "image_width": 1280,
  "image_height": 720,
  "principal_point": [640.0, 360.0]
}
```

### Case B: Raw Stereo Rig Calibration (`mode: "full_stereo_rig"`)
Used when images are raw captures with optical distortion and non-parallel axes.
Required JSON fields:
- `left_camera_matrix` ($3 \times 3$ matrix $K_1$)
- `right_camera_matrix` ($3 \times 3$ matrix $K_2$)
- `left_distortion_coefficients` (vector $D_1$, at least 4 terms: $[k_1, k_2, p_1, p_2]$)
- `right_distortion_coefficients` (vector $D_2$)
- `rotation_matrix` ($3 \times 3$ rotation matrix $R$)
- `translation_vector` ($3 \times 1$ translation vector $T$ in mm)

The backend runs OpenCV `stereoRectify`, `initUndistortRectifyMap`, and `remap` to transform raw images into rectified standard stereo geometry before disparity estimation.

### Input Rejection Policy
Inputs are strictly rejected with HTTP 400 Bad Request if:
- Left or right image is corrupted, empty, or unreadable.
- Left and right image resolutions mismatch.
- Calibration file is missing or contains invalid JSON.
- Calibration lacks baseline or focal length parameters.
- Baseline is non-positive ($B \le 0$).

---

## Coordinate Conventions & Units

- **Physical Units**: Millimeters (`mm`).
- **Disparity**: Sub-pixel horizontal displacement in pixels (`px`).
- **Camera Triangulation (OpenCV Camera Frame)**:
  $$Z = \frac{f_x \times B}{d}$$
  $$X = \frac{(u - c_x) \times Z}{f_x}$$
  $$Y = \frac{(v - c_y) \times Z}{f_y}$$
- **Disparity Filtering**: Disparities $d \le 0.5$, non-finite values, and depths outside physical bounds ($[50\text{ mm}, 50000\text{ mm}]$) are strictly discarded. Never fabricates points.
- **Three.js World Frame Alignment**:
  $$X_{\text{Three}} = X_{\text{camera}}$$
  $$Y_{\text{Three}} = -Y_{\text{camera}} \quad (\text{inverted so image top maps to }+Y\text{ UP})$$
  $$Z_{\text{Three}} = -Z_{\text{camera}} \quad (\text{inverted so surface faces camera looking towards }-Z)$$

---

## 3D Artifact Formats: Point Cloud & Surface Mesh

### 1. Point Cloud Artifact (`pointcloud.ply`)
Saved as standard binary little-endian PLY 1.0 containing unindexed coordinates and RGB colors:
```
ply
format binary_little_endian 1.0
comment Created by Surface Stereo Reconstruction Pipeline
element vertex <N>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
```

### 2. Poisson Surface Mesh Artifact (`mesh.ply`)
Generated via Poisson Surface Reconstruction (Satyam's `PointCloudToMesh` integration). Produces an indexed triangular surface manifold with smooth vertex normals and transferred vertex colors:
```
ply
format binary_little_endian 1.0
element vertex <V>
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face <F>
property list uchar int vertex_indices
end_header
```
Both artifacts are served directly by the backend and rendered in the frontend Three.js viewer with interactive toggling between **Surface Mesh**, **Point Cloud**, and **Wireframe**.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/inspections` | Multipart upload: left image, right image, calibration JSON, metadata. Returns 201 Created with unique inspection ID. |
| `GET` | `/api/inspections/{id}` | Fetches authentic status, 6 pipeline stages, duration, and artifact links. |
| `GET` | `/api/inspections` | Lists all historical inspections. |
| `GET / HEAD` | `/api/inspections/{id}/artifacts/{name}` | Serves pipeline artifacts (`pointcloud.ply`, `disparity_vis.png`, `disparity.npy`, etc.). |
| `GET` | `/api/health` | Service health status. |

---

## Truthful Job Status Guarantee (Milestone 1)

1. Status transitions strictly follow real background execution (`queued` $\to$ `processing` $\to$ `completed` / `failed`).
2. Duration measures authentic CPU/GPU execution time.
3. Errors produce explicit failure messages attached to the failed stage.
4. In this first milestone, findings list is truthful: empty `[]` without fabricated cracks or synthetic deviations.
