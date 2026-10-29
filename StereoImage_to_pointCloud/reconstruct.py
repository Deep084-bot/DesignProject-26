"""Reconstruct a colored point cloud from a calibrated stereo image pair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from tqdm.auto import tqdm


def load_calibration(path: Path) -> dict[str, object]:
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(calibration, dict):
        raise ValueError("Calibration JSON must contain a top-level object.")

    mode = calibration.get("mode")
    if mode not in ("rectified", "stereo"):
        raise ValueError("Calibration 'mode' must be 'rectified' or 'stereo'.")
    if "image_size" in calibration:
        try:
            size = np.asarray(calibration["image_size"], dtype=np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "'image_size' must be [width, height] using numeric values."
            ) from error
        if size.shape != (2,) or np.any(size <= 0):
            raise ValueError("'image_size' must be [width, height] with positive values.")
        calibration["image_size"] = (int(size[0]), int(size[1]))

    def matrix(name: str, shape: tuple[int, int]) -> np.ndarray:
        if name not in calibration:
            raise ValueError(f"Calibration is missing '{name}'.")
        try:
            result = np.asarray(calibration[name], dtype=np.float64).reshape(shape)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{name}' must contain {shape[0] * shape[1]} numbers.") from error
        if not np.all(np.isfinite(result)):
            raise ValueError(f"'{name}' contains a non-finite value.")
        return result

    for key in ("K_left", "K_right"):
        if key in calibration or key == "K_left" or mode == "stereo":
            calibration[key] = matrix(key, (3, 3))
    for key in ("K_left", "K_right"):
        if key in calibration:
            intrinsics = calibration[key]
            assert isinstance(intrinsics, np.ndarray)
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"'{key}' focal lengths must be positive.")

    if mode == "rectified":
        if "baseline" not in calibration:
            raise ValueError("Rectified calibration requires a positive 'baseline'.")
        calibration["baseline"] = float(calibration["baseline"])
        if not np.isfinite(calibration["baseline"]) or calibration["baseline"] <= 0:
            raise ValueError("'baseline' must be a positive number.")
        if "doffs" in calibration:
            calibration["doffs"] = float(calibration["doffs"])
            if not np.isfinite(calibration["doffs"]):
                raise ValueError("'doffs' must be finite.")
        elif "K_right" in calibration:
            left_k = calibration["K_left"]
            right_k = calibration["K_right"]
            assert isinstance(left_k, np.ndarray) and isinstance(right_k, np.ndarray)
            calibration["doffs"] = float(right_k[0, 2] - left_k[0, 2])
        else:
            calibration["doffs"] = 0.0
    else:
        for key in ("D_left", "D_right"):
            if key not in calibration:
                raise ValueError(f"Raw stereo calibration requires '{key}'. Use [] for no distortion.")
            distortion = np.asarray(calibration[key], dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(distortion)):
                raise ValueError(f"'{key}' contains a non-finite value.")
            calibration[key] = distortion
        if "R_left_to_right" in calibration:
            calibration["R_left_to_right"] = matrix("R_left_to_right", (3, 3))
        elif "rotation_euler_degrees_xyz" in calibration:
            convention = calibration.get("rotation_convention")
            if convention != "extrinsic_xyz":
                raise ValueError(
                    "Euler angles require 'rotation_convention': 'extrinsic_xyz'. "
                    "Only use this if it matches your calibration's convention; "
                    "otherwise provide 'R_left_to_right' as a matrix."
                )
            try:
                angles = np.asarray(
                    calibration["rotation_euler_degrees_xyz"], dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "'rotation_euler_degrees_xyz' must contain numeric degree values."
                ) from error
            if angles.size != 3 or not np.all(np.isfinite(angles)):
                raise ValueError(
                    "'rotation_euler_degrees_xyz' must contain three finite angles."
                )
            rx, ry, rz = np.deg2rad(angles)
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            rotation_x = np.array(
                [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64
            )
            rotation_y = np.array(
                [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64
            )
            rotation_z = np.array(
                [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64
            )
            calibration["R_left_to_right"] = rotation_z @ rotation_y @ rotation_x
        else:
            raise ValueError(
                "Raw stereo calibration requires 'R_left_to_right' or "
                "'rotation_euler_degrees_xyz'."
            )
        try:
            translation = np.asarray(
                calibration.get("T_left_to_right"), dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "'T_left_to_right' must contain three numeric values."
            ) from error
        if translation.size != 3 or not np.all(np.isfinite(translation)):
            raise ValueError("'T_left_to_right' must contain three finite values.")
        if np.linalg.norm(translation) <= 0:
            raise ValueError("'T_left_to_right' must have nonzero length.")
        calibration["T_left_to_right"] = translation

    if "disparity_range" in calibration:
        disparity_range = np.asarray(calibration["disparity_range"], dtype=np.float64)
        if (
            disparity_range.shape != (2,)
            or not np.all(np.isfinite(disparity_range))
            or disparity_range[0] >= disparity_range[1]
        ):
            raise ValueError("'disparity_range' must be [minimum, maximum] in pixels.")
        calibration["disparity_range"] = disparity_range

    calibration["units"] = str(calibration.get("units", "calibration units"))
    return calibration


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    calibration: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    height, width = left.shape[:2]
    if right.shape[:2] != (height, width):
        raise ValueError("Left and right images must have identical dimensions.")
    image_size = (width, height)
    declared_size = calibration.get("image_size")
    if declared_size is not None and declared_size != image_size:
        raise ValueError(
            f"Images are {width}x{height}, but calibration image_size is "
            f"{declared_size[0]}x{declared_size[1]}. Use matching calibration."
        )

    left_k = calibration["K_left"]
    assert isinstance(left_k, np.ndarray)
    if calibration["mode"] == "rectified":
        baseline = float(calibration["baseline"])
        doffs = float(calibration["doffs"])
        return left, right, left_k, baseline, doffs, np.ones((height, width), dtype=bool)

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required to rectify raw stereo images. Install requirements.txt."
        ) from error

    right_k = calibration["K_right"]
    rotation = calibration["R_left_to_right"]
    left_distortion = calibration["D_left"]
    right_distortion = calibration["D_right"]
    translation = calibration["T_left_to_right"]
    assert isinstance(right_k, np.ndarray)
    assert isinstance(rotation, np.ndarray)
    assert isinstance(left_distortion, np.ndarray)
    assert isinstance(right_distortion, np.ndarray)
    assert isinstance(translation, np.ndarray)

    rotation_left, rotation_right, projection_left, projection_right, _, _, _ = (
        cv2.stereoRectify(
            left_k,
            left_distortion,
            right_k,
            right_distortion,
            image_size,
            rotation,
            translation.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=image_size,
        )
    )
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        left_k,
        left_distortion,
        rotation_left,
        projection_left[:, :3],
        image_size,
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        right_k,
        right_distortion,
        rotation_right,
        projection_right[:, :3],
        image_size,
        cv2.CV_32FC1,
    )
    rectified_left = cv2.remap(
        left, left_map_x, left_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    rectified_right = cv2.remap(
        right, right_map_x, right_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    source_mask = np.ones((height, width), dtype=np.uint8)
    valid_left = cv2.remap(
        source_mask, left_map_x, left_map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ).astype(bool)
    rectified_intrinsics = projection_left[:, :3]
    baseline = float(np.linalg.norm(translation))
    doffs = float(projection_right[0, 2] - projection_left[0, 2])
    return (
        rectified_left,
        rectified_right,
        rectified_intrinsics,
        baseline,
        doffs,
        valid_left,
    )


def load_raft_model(
    raft_root: Path,
    checkpoint: Path,
    device: torch.device,
    context_norm: str,
    n_downsample: int,
    n_gru_layers: int,
    hidden_dims: list[int],
    shared_backbone: bool,
    slow_fast_gru: bool,
    corr_implementation: str,
):
    if not (raft_root / "core" / "raft_stereo.py").is_file():
        raise FileNotFoundError(
            f"RAFT-Stereo source not found under {raft_root}. "
            "Clone the Princeton RAFT-Stereo repository and pass its folder "
            "with --raft-root."
        )
    sys.path.insert(0, str(raft_root.resolve()))
    from core.raft_stereo import RAFTStereo
    from core.utils.utils import InputPadder

    args = SimpleNamespace(
        hidden_dims=hidden_dims,
        context_norm=context_norm,
        n_downsample=n_downsample,
        n_gru_layers=n_gru_layers,
        shared_backbone=shared_backbone,
        corr_implementation=corr_implementation,
        corr_levels=4,
        corr_radius=4,
        slow_fast_gru=slow_fast_gru,
        mixed_precision=device.type == "cuda",
    )
    model = RAFTStereo(args)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("The RAFT checkpoint does not contain a state dictionary.")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, InputPadder


def save_numpy_checkpoint(path: Path, array: np.ndarray) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, array)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def predict_disparity(
    model,
    padder_type,
    left: np.ndarray,
    right: np.ndarray,
    max_image_size: int,
    iterations: int,
    device: torch.device,
) -> np.ndarray:
    height, width = left.shape[:2]
    scale = min(1.0, max_image_size / max(height, width)) if max_image_size else 1.0
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    if (resized_width, resized_height) != (width, height):
        resampling = Image.Resampling.LANCZOS
        left = np.array(
            Image.fromarray(left).resize((resized_width, resized_height), resampling),
            copy=True,
        )
        right = np.array(
            Image.fromarray(right).resize((resized_width, resized_height), resampling),
            copy=True,
        )

    image_tensors = [
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        for image in (left, right)
    ]
    padder = padder_type(image_tensors[0].shape, divis_by=32)
    left_tensor, right_tensor = padder.pad(*image_tensors)

    def update_progress(_module, _inputs, _output) -> None:
        progress.update()

    with tqdm(total=iterations, desc="RAFT updates", unit="iter") as progress:
        hook = model.update_block.register_forward_hook(update_progress)
        try:
            with torch.inference_mode():
                _, raft_flow = model(
                    left_tensor,
                    right_tensor,
                    iters=iterations,
                    test_mode=True,
                )
        finally:
            hook.remove()

    disparity = -padder.unpad(raft_flow)[0, 0].float().cpu().numpy()
    if disparity.shape != (height, width):
        disparity_tensor = torch.from_numpy(disparity)[None, None]
        disparity = functional.interpolate(
            disparity_tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        disparity *= width / resized_width
    return disparity


def make_point_cloud(
    disparity: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    baseline: float,
    doffs: float,
    valid_image_mask: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    min_depth: float | None,
    max_depth: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("Rectified camera focal lengths must be positive.")

    denominator = disparity + doffs
    valid = (
        np.isfinite(disparity)
        & (disparity > 0)
        & (denominator > 0)
        & valid_image_mask
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (fx * baseline / denominator[valid]).astype(np.float32)
    valid &= np.isfinite(depth)
    if min_depth is not None:
        valid &= depth >= min_depth
    if max_depth is not None:
        valid &= depth <= max_depth
    if roi is not None:
        x, y, width, height = roi
        roi_mask = np.zeros(valid.shape, dtype=bool)
        roi_mask[y : y + height, x : x + width] = True
        valid &= roi_mask

    v, u = np.indices(disparity.shape, dtype=np.float32)
    z = depth[valid]
    points = np.column_stack(
        ((u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z)
    ).astype(np.float64)
    point_colors = colors[valid].astype(np.float64) / 255.0
    return points, point_colors, depth


def validate_disparity_range(
    disparity: np.ndarray, calibration: dict[str, object]
) -> None:
    disparity_range = calibration.get("disparity_range")
    if disparity_range is None:
        return
    values = np.asarray(disparity_range, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] >= values[1]:
        raise ValueError("'disparity_range' must be [minimum, maximum] in pixels.")
    valid_fraction = float(
        np.mean(
            np.isfinite(disparity)
            & (disparity >= values[0])
            & (disparity <= values[1])
        )
    )
    if valid_fraction < 0.05:
        raise RuntimeError(
            f"Only {valid_fraction:.1%} of predicted disparities are within "
            f"the calibration's optional range [{values[0]:g}, {values[1]:g}] pixels. "
            "Check image order, rectification, and checkpoint compatibility."
        )


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height.") from error
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            "ROI coordinates must be non-negative and dimensions positive."
        )
    return x, y, width, height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate stereo disparity with RAFT-Stereo and export an Open3D point cloud."
    )
    parser.add_argument("--left", type=Path, required=True, help="Left-camera image.")
    parser.add_argument("--right", type=Path, required=True, help="Right-camera image.")
    parser.add_argument("--calib", type=Path, required=True, help="Calibration JSON file.")
    default_checkpoint = (
        Path(__file__).resolve().parent
        / "RAFT-Stereo"
        / "models"
        / "raftstereo-middlebury.pth"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help=(
            "Compatible RAFT-Stereo checkpoint. Defaults to the included "
            "Middlebury checkpoint; provide another path to override it."
        ),
    )
    default_raft_root = Path(__file__).resolve().parent / "RAFT-Stereo"
    parser.add_argument("--raft-root", type=Path, default=default_raft_root)
    parser.add_argument("--output", type=Path, default=Path("reconstruction"))
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1200,
        help="Maximum inference image dimension; use 0 for original resolution.",
    )
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--roi", type=parse_roi, help="Optional crop: x,y,width,height.")
    parser.add_argument("--min-depth", type=float, help="Minimum depth in calibration units.")
    parser.add_argument("--max-depth", type=float, help="Maximum depth in calibration units.")
    parser.add_argument(
        "--context-norm",
        choices=("batch", "group", "instance", "none"),
        default="batch",
        help="Must match the checkpoint's RAFT-Stereo architecture.",
    )
    parser.add_argument("--n-downsample", type=int, choices=(2, 3), default=2)
    parser.add_argument("--n-gru-layers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        metavar=("LOW", "MID", "HIGH"),
        help="Must match the checkpoint architecture.",
    )
    parser.add_argument("--shared-backbone", action="store_true")
    parser.add_argument("--slow-fast-gru", action="store_true")
    parser.add_argument(
        "--corr-implementation",
        choices=("reg", "alt", "reg_cuda", "alt_cuda"),
        default="alt",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be positive.")
    if any(dimension <= 0 for dimension in args.hidden_dims):
        parser.error("--hidden-dims values must be positive.")
    if args.max_image_size < 0:
        parser.error("--max-image-size cannot be negative.")
    if args.min_depth is not None and args.min_depth < 0:
        parser.error("--min-depth cannot be negative.")
    if args.max_depth is not None and args.max_depth <= 0:
        parser.error("--max-depth must be positive.")
    if (
        args.min_depth is not None
        and args.max_depth is not None
        and args.min_depth >= args.max_depth
    ):
        parser.error("--min-depth must be less than --max-depth.")
    for path in (args.left, args.right, args.calib, args.checkpoint):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    if args.corr_implementation.endswith("_cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA correlation implementations require a CUDA-enabled PyTorch install and GPU."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    stages = tqdm(total=5, desc="Reconstruction", unit="stage")
    stages.set_postfix_str("loading calibration and images")
    calibration = load_calibration(args.calib)
    left, right = load_rgb(args.left), load_rgb(args.right)
    (
        left,
        right,
        intrinsics,
        baseline,
        doffs,
        valid_image_mask,
    ) = rectify_pair(left, right, calibration)
    if args.roi is not None:
        x, y, width, height = args.roi
        if x + width > left.shape[1] or y + height > left.shape[0]:
            parser.error("--roi extends outside the input images.")

    stages.update()
    stages.set_postfix_str("loading RAFT-Stereo checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, padder_type = load_raft_model(
        args.raft_root,
        args.checkpoint,
        device,
        args.context_norm,
        args.n_downsample,
        args.n_gru_layers,
        args.hidden_dims,
        args.shared_backbone,
        args.slow_fast_gru,
        args.corr_implementation,
    )
    stages.update()
    stages.set_postfix_str(f"estimating disparity on {device}")
    disparity = predict_disparity(
        model,
        padder_type,
        left,
        right,
        args.max_image_size,
        args.iterations,
        device,
    )
    validate_disparity_range(disparity, calibration)
    save_numpy_checkpoint(args.output / "disparity.npy", disparity)
    stages.update()
    stages.set_postfix_str("reprojecting depth and saving checkpoint")
    points, point_colors, depth = make_point_cloud(
        disparity,
        left,
        intrinsics,
        baseline,
        doffs,
        valid_image_mask,
        args.roi,
        args.min_depth,
        args.max_depth,
    )
    if points.size == 0:
        raise RuntimeError(
            "No valid 3D points remain. Check disparity, calibration, ROI, and depth limits."
        )
    save_numpy_checkpoint(args.output / "depth.npy", depth)
    stages.update()
    stages.set_postfix_str("writing colored point cloud")

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required to save the point cloud. Install requirements.txt."
        ) from error

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(point_colors)
    cloud_path = args.output / "point_cloud.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud):
        raise OSError(f"Open3D failed to write {cloud_path}.")
    stages.update()
    stages.close()
    print(
        f"Saved {len(points):,} points to {cloud_path}. "
        f"Depth units: {calibration['units']}."
    )


if __name__ == "__main__":
    main()
