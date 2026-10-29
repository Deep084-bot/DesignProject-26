# Stereo Image to 3D Point Cloud

Reconstruct a colored point cloud from a calibrated left/right stereo image
pair using Princeton's RAFT-Stereo model, OpenCV, and Open3D.

The pipeline supports:

- **Rectified stereo images**, with camera intrinsics and baseline.
- **Raw stereo images**, with each camera's intrinsics and distortion plus
  the stereo rotation and translation; OpenCV rectifies these before RAFT
  inference.

It outputs a `.ply` point cloud and NumPy disparity/depth files. One stereo
pair reconstructs surfaces visible to both cameras; it does not recover the
object's hidden sides.

## Requirements

- Python 3.10 or later
- PyTorch (CPU works; CUDA-enabled PyTorch and an NVIDIA GPU are recommended)
- Git
- A synchronized left/right image pair and calibration for those cameras

## Setup

### 1. Get this project

Clone this repository, then open a terminal in its folder:

```powershell
git clone <this-project-url>
cd <this-project-folder>
```

### 2. Install PyTorch and project dependencies

Install PyTorch for your platform using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).
Choose a CUDA-enabled build if you have a compatible NVIDIA GPU. Then install
the other dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 3. Clone the RAFT-Stereo source

From the project folder, clone Princeton's implementation into the expected
`RAFT-Stereo` directory:

```powershell
git clone https://github.com/princeton-vl/RAFT-Stereo.git RAFT-Stereo
```

Cloning RAFT separately is the simplest setup for users of this project and
keeps its source and license in its own repository. The script expects this
folder next to `reconstruct.py` by default. If you put it elsewhere, supply
its location with `--raft-root`.

### 4. Download the default Middlebury checkpoint

RAFT-Stereo needs pretrained weights (a checkpoint) to estimate disparity.
This project uses the official **Middlebury-trained checkpoint by default**.
The checkpoint is a large model file and is intentionally not stored in this
GitHub repository. Download the official RAFT-Stereo models from the
[official model instructions](https://github.com/princeton-vl/RAFT-Stereo#demos)
and [checkpoint download folder](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J).
Select `raftstereo-middlebury.pth` and place it at:

```text
RAFT-Stereo/models/raftstereo-middlebury.pth
```

That is the path the script uses when `--checkpoint` is omitted. You do not
need to train your own model just to run inference.

The Middlebury checkpoint is recommended by the RAFT-Stereo authors for
in-the-wild images, but its accuracy is not guaranteed for every camera or
scene. Test it on representative data. To use another compatible RAFT-Stereo
checkpoint, download that `.pth` file and pass its location using
`--checkpoint`. The checkpoint's architecture must match the RAFT settings
passed to the script.

## Prepare calibration

Create a calibration JSON file for the cameras that captured your images.
Templates are provided:

- [Rectified camera example](calibration.rectified.example.json)
- [Raw stereo camera example](calibration.stereo.example.json)
- [Template for the Center/Focal Length/Skew/RX/RY/RZ/TX/TY/TZ fields](calibration.user.template.json)

Replace example values with your actual camera calibration; the templates'
numbers are not valid calibration for other cameras. Calibration provides the
geometry and real-world scale. The RAFT model estimates disparity; it does not
calibrate cameras.

For `mode: "rectified"`, provide at least the left rectified intrinsic matrix
`K_left` and positive `baseline`. Set `units` to the baseline unit (for example
meters or millimeters). Include `image_size` if the calibration is tied to a
specific resolution. Add `K_right` and `doffs` when required by your
calibration's disparity convention. Corresponding points should already be on
the same image rows.

For `mode: "stereo"`, provide both camera intrinsic matrices (`K_left`,
`K_right`), distortion coefficients (`D_left`, `D_right`), rotation
(`R_left_to_right`) and translation (`T_left_to_right`) from the left camera
to the right camera. The translation unit determines the 3D output unit.
OpenCV uses these values to rectify the input pair.

If your calibration provides Euler angles rather than a rotation matrix, the
[user calibration template](calibration.user.template.json) accepts
`rotation_euler_degrees_xyz`. It assumes the **extrinsic XYZ** convention
`R = Rz @ Ry @ Rx`. Use it only if this matches your calibration's definition;
degrees alone do not determine the rotation order or convention. The template
assumes Camera 0 is left, Camera 1 is right, translations are millimeters,
and Kappa distortion can be ignored. Verify these assumptions for your data.
Replace every `REPLACE_*` string with a numeric value and save the result as a
new `.json` file before running. The template itself is intentionally not a
ready-to-run calibration.

The input image dimensions must match `image_size` when it is present. If you
resize or crop images after calibration, update the intrinsics accordingly.
Do not guess camera parameters. If you do not have a calibration, calibrate
the actual camera pair first.

For the fields shown in your calibration screenshot, use
[calibration.user.template.json](calibration.user.template.json). It assumes
Camera 0 is left, Camera 1 is right, TX/TY/TZ are the direct translation from
left camera to right camera in millimeters, and RX/RY/RZ are extrinsic XYZ
Euler angles in degrees. The corresponding matrix is `Rz @ Ry @ Rx`. It leaves
distortion empty, which means Kappa is ignored. Replace every
`REPLACE_WITH_...` string before running. These pose/angle conventions are
assumptions, not facts established by the screenshot; if your calibration
software uses a different rotation convention or reports camera poses in a
world frame, this template must be adapted before its output can be trusted.

## Run

Provide the **left camera image first** and the **right camera image second**:

```powershell
python reconstruct.py `
  --left C:\data\left.png `
  --right C:\data\right.png `
  --calib C:\data\calibration.json `
  --output C:\data\reconstruction
```

The included Middlebury checkpoint is selected by default. To choose another
checkpoint, add:

```powershell
  --checkpoint C:\models\another-raftstereo-checkpoint.pth
```

The images should be captured at the same time, show the same static scene,
and overlap. Use `--help` to see optional settings:

```powershell
python reconstruct.py --help
```

For a quick test, reduce inference resolution and iterations, for example:

```powershell
python reconstruct.py `
  --left C:\data\left.png `
  --right C:\data\right.png `
  --calib C:\data\calibration.json `
  --max-image-size 480 `
  --iterations 8 `
  --roi 100,100,800,600 `
  --output C:\data\quick-test
```

Increase image size and iteration count for a more detailed result. CPU
inference may be slow. The progress bars show pipeline stages and RAFT update
iterations.

## Outputs

The output directory contains:

- `point_cloud.ply` — colored 3D point cloud; open it in Open3D, MeshLab, or
  CloudCompare.
- `disparity.npy` — predicted disparity in pixels.
- `depth.npy` — depth values in calibration units; invalid pixels are NaN.

Points are expressed in the rectified left-camera coordinate frame. A pose
for the left camera is needed to transform them into a separate world frame.
To build a model of hidden sides, capture additional viewpoints and fuse their
point clouds or depth maps.

## Testing on your own dataset

1. Start with one synchronized pair and the matching camera calibration.
2. Verify which image is left and which is right; check that resolution and
   calibration correspond.
3. Run at smaller resolution first, then inspect the point cloud and depth.
4. Test several representative pairs from your expected distances, lighting,
   textures, and materials.
5. If you have ground-truth depth or disparity, compare against it. Visual
   inspection alone does not quantify accuracy.
6. If disparity is poor, try a different checkpoint or fine-tune/train RAFT
   on rectified stereo examples from your domain. A single stereo pair and
   calibration are enough for inference, not for training a reliable model.

## License and dependencies

RAFT-Stereo is a separate project; see its repository for its source license
and model/checkpoint terms. If you redistribute its code or weights, follow
the applicable licenses and attribution requirements. This project does not
include RAFT source or model weights in its GitHub repository; users clone the
RAFT repository and download a checkpoint separately.
