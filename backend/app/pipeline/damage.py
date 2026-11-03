"""
Damage Detection and Crack Classification (Stage 4).

Combines deep learning visual surface crack classification (EfficientNet-B0)
with 3D geometric surface deviation anomalies from Stage 3 to produce
categorized, calibrated structural findings.

Distinguishes:
1. potential_crack: Supported by visual crack classification and geometric deviation,
   or strong visual evidence alone (P >= 0.8).
2. minor_anomaly: Visual anomaly with moderate classifier probability (0.5 <= P < 0.8).
3. surface_irregularity: Out-of-plane geometric anomaly without visual crack evidence.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..models.inspection import Finding, FindingType

# Resolve base directories
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _BACKEND_DIR.parent


def resolve_model_path(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolves the crack classifier checkpoint path from argument, env, or defaults."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()
        return p

    env_path = os.environ.get("SURFACE_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p.resolve()
        return p

    candidates = [
        _BACKEND_DIR / "model_store" / "best_model_sdnet2018.pth",
        _BACKEND_DIR / "models_store" / "best_model_sdnet2018.pth",
        _REPO_ROOT / "backend" / "model_store" / "best_model_sdnet2018.pth",
        _REPO_ROOT / "backend" / "models_store" / "best_model_sdnet2018.pth",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


class CrackClassifier:
    """
    Singleton wrapper for the EfficientNet-B0 surface crack classification model.
    Preprocesses tiles (BGR->RGB, 224x224, ImageNet normalized) and performs
    batched evaluation on CUDA, MPS, or CPU.
    """

    _instance: Optional["CrackClassifier"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.model_path = resolve_model_path(model_path)
        self.device = self._select_device(device)
        self.model: Optional[nn.Module] = None
        self.status: str = "unavailable"
        self.error: Optional[str] = None
        self.classes: List[str] = ["Uncracked", "Cracked"]
        self._load()

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> "CrackClassifier":
        with cls._lock:
            target_path = resolve_model_path(model_path)
            if cls._instance is None or cls._instance.model_path != target_path:
                cls._instance = cls(model_path=target_path, device=device)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Resets the singleton for clean isolation in tests."""
        with cls._lock:
            cls._instance = None

    def _select_device(self, preferred: Optional[str] = None) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self):
        if not self.model_path or not self.model_path.exists():
            self.status = "unavailable"
            self.error = f"Model checkpoint not found at '{self.model_path}'"
            self.model = None
            return

        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                self.status = "failed"
                self.error = "Invalid checkpoint format: expected dictionary containing 'state_dict'."
                self.model = None
                return

            if "classes" in ckpt:
                self.classes = ckpt["classes"]

            from torchvision.models import efficientnet_b0

            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(self.classes))
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            self.model = model
            self.status = "completed"
            self.error = None
        except Exception as exc:
            self.status = "failed"
            self.error = f"Failed to load EfficientNet-B0 checkpoint: {exc}"
            self.model = None

    def predict_batch(self, tiles_bgr: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """
        Runs batched inference on a list of BGR tile images.
        Returns 1D float array of predicted crack probabilities (class index 1).
        """
        if self.model is None or self.status != "completed":
            raise RuntimeError(f"Classifier is not available ({self.status}): {self.error}")

        if not tiles_bgr:
            return np.empty((0,), dtype=np.float32)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        probabilities = []
        for i in range(0, len(tiles_bgr), batch_size):
            chunk = tiles_bgr[i : i + batch_size]
            batch_tensors = []
            for img in chunk:
                # 1. BGR to RGB
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 2. Resize to 224x224
                resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                # 3. [0, 1] tensor
                t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                batch_tensors.append(t)

            batch_stack = torch.stack(batch_tensors).to(self.device)
            # 4. Normalize with ImageNet mean and std
            batch_norm = (batch_stack - mean) / std

            with torch.no_grad():
                logits = self.model(batch_norm)
                probs = torch.softmax(logits, dim=1)
                # Class 1 corresponds to Cracked
                cracked_p = probs[:, 1].cpu().numpy()
                probabilities.extend(cracked_p.tolist())

        return np.array(probabilities, dtype=np.float32)


@dataclass
class DamageDetectionResult:
    """Complete results of damage classification and finding correlation."""

    findings: List[Finding] = field(default_factory=list)
    classifier_status: str = "unavailable"
    classifier_error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


def generate_tile_definitions(
    image_shape: Tuple[int, int],
    deviation_regions: List[Any],
    tile_size: int = 256,
    stride: int = 256,
) -> List[Dict[str, Any]]:
    """
    Generates deterministic tile coordinates and bounding boxes across the image
    (regular grid) and focused extra tiles around each deviation anomaly region.

    Returns lightweight, JSON-serializable tile definition metadata without
    storing raw image numpy arrays.
    """
    H, W = image_shape[:2]
    tile_defs: List[Dict[str, Any]] = []

    # If image is smaller than tile_size, mark as single small-image tile
    if H < tile_size or W < tile_size:
        tile_defs.append({
            "tile_index": 0,
            "bbox": {"x": 0, "y": 0, "width": W, "height": H},
            "crop_coords": [0, 0, W, H],
            "source": "grid",
            "region": None,
            "is_small_image": True,
        })
        return tile_defs

    tile_counter = 0

    # 1. Regular Grid Tiles
    ny = int(np.ceil(H / stride))
    nx = int(np.ceil(W / stride))

    for iy in range(ny):
        y0 = iy * stride
        for ix in range(nx):
            x0 = ix * stride
            actual_w = min(tile_size, W - x0)
            actual_h = min(tile_size, H - y0)
            tile_defs.append({
                "tile_index": tile_counter,
                "bbox": {"x": x0, "y": y0, "width": actual_w, "height": actual_h},
                "crop_coords": [y0, x0, tile_size, tile_size],
                "source": "grid",
                "region": None,
                "is_small_image": False,
            })
            tile_counter += 1

    # 2. Extra Tiles Centered Around Deviation Regions
    for reg in deviation_regions:
        rx = reg.get("x", reg.get("bbox_px", {}).get("x", 0)) if isinstance(reg, dict) else reg.x
        ry = reg.get("y", reg.get("bbox_px", {}).get("y", 0)) if isinstance(reg, dict) else reg.y
        rw = reg.get("width", reg.get("bbox_px", {}).get("width", 0)) if isinstance(reg, dict) else reg.width
        rh = reg.get("height", reg.get("bbox_px", {}).get("height", 0)) if isinstance(reg, dict) else reg.height

        cx = rx + rw / 2.0
        cy = ry + rh / 2.0

        if rw <= tile_size and rh <= tile_size:
            # Single centered tile
            x0 = int(round(cx - tile_size / 2.0))
            y0 = int(round(cy - tile_size / 2.0))
            x0 = max(0, min(W - tile_size, x0))
            y0 = max(0, min(H - tile_size, y0))
            tile_defs.append({
                "tile_index": tile_counter,
                "bbox": {"x": x0, "y": y0, "width": tile_size, "height": tile_size},
                "crop_coords": [y0, x0, tile_size, tile_size],
                "source": "region",
                "region": reg,
                "is_small_image": False,
            })
            tile_counter += 1
        else:
            # Multi-tile coverage across large region with 50% overlap
            step = tile_size // 2
            for ty in range(ry, ry + rh, step):
                for tx in range(rx, rx + rw, step):
                    x0 = max(0, min(W - tile_size, tx))
                    y0 = max(0, min(H - tile_size, ty))
                    tile_defs.append({
                        "tile_index": tile_counter,
                        "bbox": {"x": x0, "y": y0, "width": tile_size, "height": tile_size},
                        "crop_coords": [y0, x0, tile_size, tile_size],
                        "source": "region",
                        "region": reg,
                        "is_small_image": False,
                    })
                    tile_counter += 1

    return tile_defs


def extract_tile_crop(
    image_bgr: np.ndarray,
    tile_def: Dict[str, Any],
    tile_size: int = 256,
) -> np.ndarray:
    """
    Extracts a 256x256 BGR tile crop from the full rectified image according to its
    tile definition, applying border reflection padding if required.
    """
    H, W = image_bgr.shape[:2]

    if tile_def.get("is_small_image") or H < tile_size or W < tile_size:
        pad_h = max(0, tile_size - H)
        pad_w = max(0, tile_size - W)
        padded = cv2.copyMakeBorder(image_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        return padded[:tile_size, :tile_size]

    y0 = int(tile_def["crop_coords"][0])
    x0 = int(tile_def["crop_coords"][1])

    pad_bottom = max(0, (y0 + tile_size) - H)
    pad_right = max(0, (x0 + tile_size) - W)
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        padded = cv2.copyMakeBorder(image_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        return padded[y0 + pad_top : y0 + pad_top + tile_size, x0 + pad_left : x0 + pad_left + tile_size]
    else:
        return image_bgr[y0 : y0 + tile_size, x0 : x0 + tile_size]


def generate_tiles(
    image_bgr: np.ndarray,
    deviation_regions: List[Any],
    tile_size: int = 256,
    stride: int = 256,
) -> List[Dict[str, Any]]:
    """
    Generates regular grid tiles across the entire image and focused extra tiles
    around each deviation anomaly region.

    Retained as backward-compatible wrapper that delegates to generate_tile_definitions
    and extract_tile_crop.
    """
    tile_defs = generate_tile_definitions(
        image_shape=image_bgr.shape[:2],
        deviation_regions=deviation_regions,
        tile_size=tile_size,
        stride=stride,
    )
    tiles: List[Dict[str, Any]] = []
    for t_def in tile_defs:
        crop = extract_tile_crop(image_bgr, t_def, tile_size=tile_size)
        tiles.append({
            "tile_bgr": crop,
            "bbox": t_def["bbox"],
            "source": t_def["source"],
            "region": t_def["region"],
        })
    return tiles


def boxes_overlap_or_touch(b1: Dict[str, Any], b2: Dict[str, Any]) -> bool:
    """Determines whether two 2D bounding boxes overlap or touch."""
    x1, y1, w1, h1 = b1["x"], b1["y"], b1["width"], b1["height"]
    x2, y2, w2, h2 = b2["x"], b2["y"], b2["width"], b2["height"]
    return (x1 <= x2 + w2) and (x2 <= x1 + w1) and (y1 <= y2 + h2) and (y2 <= y1 + h1)


def boxes_overlap(b1: Dict[str, Any], b2: Dict[str, Any]) -> bool:
    """Determines whether two 2D bounding boxes have strictly non-zero intersection area."""
    x1, y1, w1, h1 = b1["x"], b1["y"], b1["width"], b1["height"]
    x2, y2, w2, h2 = b2["x"], b2["y"], b2["width"], b2["height"]
    inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return (inter_x * inter_y) > 0


def merge_positive_tiles(positive_tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges touching or overlapping positive tiles into unified groups.
    Computes union bounding boxes and tracks maximum probability in each group.
    """
    if not positive_tiles:
        return []

    n = len(positive_tiles)
    adj: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap_or_touch(positive_tiles[i]["bbox"], positive_tiles[j]["bbox"]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    merged_groups: List[Dict[str, Any]] = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS group collection
        group_indices = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            group_indices.append(curr)
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        group_tiles = [positive_tiles[idx] for idx in group_indices]
        union_x = min(t["bbox"]["x"] for t in group_tiles)
        union_y = min(t["bbox"]["y"] for t in group_tiles)
        union_x2 = max(t["bbox"]["x"] + t["bbox"]["width"] for t in group_tiles)
        union_y2 = max(t["bbox"]["y"] + t["bbox"]["height"] for t in group_tiles)

        max_prob = max(t["prob"] for t in group_tiles)
        associated_regions = [
            t["region"] for t in group_tiles if t.get("region") is not None
        ]

        merged_groups.append({
            "bbox": {
                "x": int(union_x),
                "y": int(union_y),
                "width": int(union_x2 - union_x),
                "height": int(union_y2 - union_y),
            },
            "confidence": float(max_prob),
            "tiles": group_tiles,
            "associated_regions": associated_regions,
        })

    return merged_groups


def estimate_box_center_3d(
    bbox: Dict[str, Any],
    disparity: Optional[np.ndarray],
    intrinsics: Optional[Dict[str, float]],
) -> Optional[List[float]]:
    """
    Estimates 3D center in Three.js viewer frame (x, -y, -z) in mm
    using median valid disparity inside the bounding box.
    """
    if disparity is None or intrinsics is None:
        return None

    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy", fx)
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    baseline = intrinsics.get("baseline_mm") or intrinsics.get("baseline_distance_mm")

    if not fx or not fy or cx is None or cy is None or not baseline:
        return None

    H, W = disparity.shape[:2]
    x = max(0, min(W - 1, int(bbox["x"])))
    y = max(0, min(H - 1, int(bbox["y"])))
    w = max(1, min(W - x, int(bbox["width"])))
    h = max(1, min(H - y, int(bbox["height"])))

    crop = disparity[y : y + h, x : x + w]
    valid = np.isfinite(crop) & (crop > 0.5)
    if not np.any(valid):
        return None

    median_d = float(np.median(crop[valid]))
    if median_d <= 0.5:
        return None

    z_cam = float((fx * baseline) / median_d)
    if z_cam < 50.0 or z_cam > 50000.0:
        return None

    u_center = x + w / 2.0
    v_center = y + h / 2.0
    x_cam = (u_center - cx) * z_cam / fx
    y_cam = (v_center - cy) * z_cam / fy

    # Three.js viewer frame: (+X right, +Y up, +Z towards viewer) -> (x, -y, -z)
    return [round(float(x_cam), 2), round(float(-y_cam), 2), round(float(-z_cam), 2)]


def sort_findings(findings: List[Finding]) -> List[Finding]:
    """
    Sorts findings in strict severity order:
    1. potential_crack
    2. surface_irregularity
    3. minor_anomaly
    Within groups, sorts descending by geometric deviation, then confidence.
    Assigns sequential human-readable IDs ('Finding 01', 'Finding 02', ...).
    """
    type_priority = {
        "potential_crack": 0,
        "surface_irregularity": 1,
        "minor_anomaly": 2,
    }

    def key_fn(f: Finding):
        rank = type_priority.get(f.type, 3)
        dev = f.geometric_deviation_mm if f.geometric_deviation_mm is not None else -1.0
        conf = f.confidence if f.confidence is not None else -1.0
        return (rank, -dev, -conf)

    sorted_list = sorted(findings, key=key_fn)
    for idx, f in enumerate(sorted_list, 1):
        f.id = f"Finding {idx:02d}"
    return sorted_list


def save_finding_crops(
    findings: List[Finding],
    rectified_left_bgr: np.ndarray,
    job_dir: Optional[Path],
    inspection_id: Optional[str] = None,
    margin_px: int = 16,
) -> None:
    """
    Saves an image crop for each finding with padding margin,
    updating each finding's image_crop_url attribute.
    """
    if job_dir is None:
        return

    job_dir.mkdir(parents=True, exist_ok=True)
    H, W = rectified_left_bgr.shape[:2]
    insp_id_str = inspection_id or job_dir.name

    for f in findings:
        if not f.bounding_box_2d:
            continue
        b = f.bounding_box_2d
        x = int(b["x"])
        y = int(b["y"])
        w = int(b["width"])
        h = int(b["height"])

        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(W, x + w + margin_px)
        y2 = min(H, y + h + margin_px)

        crop = rectified_left_bgr[y1:y2, x1:x2]
        num_str = f.id.replace("Finding", "").strip()
        filename = f"finding_{num_str}.png"
        crop_path = job_dir / filename
        cv2.imwrite(str(crop_path), crop)
        f.image_crop_url = f"/api/inspections/{insp_id_str}/artifacts/{filename}"


def normalize_deviation_regions(deviation_regions: List[Any]) -> List[Dict[str, Any]]:
    """Converts a list of DeviationRegion objects or dicts into standard dict representation."""
    regions_list: List[Dict[str, Any]] = []
    for r in deviation_regions:
        if isinstance(r, dict):
            r_norm = dict(r)
            if "bbox_px" in r and isinstance(r["bbox_px"], dict):
                bp = r["bbox_px"]
                r_norm.setdefault("x", int(bp.get("x", 0)))
                r_norm.setdefault("y", int(bp.get("y", 0)))
                r_norm.setdefault("width", int(bp.get("width", 0)))
                r_norm.setdefault("height", int(bp.get("height", 0)))
            elif "bbox" in r:
                bx = r["bbox"]
                if len(bx) == 4:
                    x1, y1, x2, y2 = bx
                    if x2 > x1 and y2 > y1 and "width" not in r:
                        w, h = x2 - x1, y2 - y1
                        x, y = x1, y1
                    else:
                        x, y, w, h = bx[0], bx[1], bx[2], bx[3]
                else:
                    x, y, w, h = 0, 0, 0, 0
                r_norm.setdefault("x", x)
                r_norm.setdefault("y", y)
                r_norm.setdefault("width", w)
                r_norm.setdefault("height", h)
                r_norm.setdefault("bbox_px", {"x": x, "y": y, "width": w, "height": h})
            else:
                x = int(r.get("x", 0))
                y = int(r.get("y", 0))
                w = int(r.get("width", 0))
                h = int(r.get("height", 0))
                r_norm.setdefault("x", x)
                r_norm.setdefault("y", y)
                r_norm.setdefault("width", w)
                r_norm.setdefault("height", h)
                r_norm.setdefault("bbox_px", {"x": x, "y": y, "width": w, "height": h})
            regions_list.append(r_norm)
        else:
            x = getattr(r, "x", 0)
            y = getattr(r, "y", 0)
            w = getattr(r, "width", 0)
            h = getattr(r, "height", 0)
            bbox_px = getattr(r, "bbox_px", {"x": x, "y": y, "width": w, "height": h})
            regions_list.append({
                "id": getattr(r, "id", None),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "bbox_px": bbox_px,
                "max_abs_deviation_mm": getattr(r, "max_abs_deviation_mm", 0.0),
                "mean_deviation_mm": getattr(r, "mean_deviation_mm", 0.0),
                "centroid_3d_viewer": getattr(r, "centroid_3d_viewer", None),
                "sign": getattr(r, "sign", "irregularity"),
            })
    return regions_list


def classify_tile_batch(
    image_bgr: np.ndarray,
    tile_defs_batch: List[Dict[str, Any]],
    classifier: Optional[Any] = None,
    batch_size: int = 16,
) -> List[float]:
    """
    Extracts image crops for a batch of tile definitions and runs batched inference
    using the specified or singleton CrackClassifier.
    Returns a list of float crack probabilities (class index 1).
    """
    if not tile_defs_batch:
        return []

    crops = [extract_tile_crop(image_bgr, td) for td in tile_defs_batch]
    clf = classifier if classifier is not None else CrackClassifier.get_instance()
    try:
        probs = clf.predict_batch(crops, batch_size=batch_size)
    except TypeError:
        probs = clf.predict_batch(crops)
    return [float(p) for p in probs]


def aggregate_damage_findings(
    deviation_regions: List[Any],
    tile_defs: Optional[List[Dict[str, Any]]] = None,
    probabilities: Optional[List[float]] = None,
    disparity: Optional[np.ndarray] = None,
    intrinsics: Optional[Dict[str, float]] = None,
    threshold: float = 0.5,
    classifier_status: str = "completed",
    classifier_error: Optional[str] = None,
    clf_device: Optional[str] = None,
    tiles_evaluated_count: Optional[int] = None,
    job_dir: Optional[Path] = None,
    image_bgr: Optional[np.ndarray] = None,
    inspection_id: Optional[str] = None,
) -> DamageDetectionResult:
    """
    Unified finding aggregation using M2 correlation logic.
    Combines tile classifications and geometric deviation regions into categorized,
    calibrated Findings without duplicate logic between local and distributed execution.
    """
    regions_list = normalize_deviation_regions(deviation_regions)

    # 1. Fallback when classifier is unavailable or failed
    if classifier_status != "completed":
        is_inference_error = bool(classifier_error and "Inference error" in classifier_error)
        description_reason = "classifier inference failed." if is_inference_error else "classifier unavailable."
        tiles_eval = tiles_evaluated_count if tiles_evaluated_count is not None else (len(tile_defs) if (tile_defs and is_inference_error) else 0)

        findings: List[Finding] = []
        for reg in regions_list:
            bbox = reg.get("bbox_px") or {"x": reg.get("x", 0), "y": reg.get("y", 0), "width": reg.get("width", 0), "height": reg.get("height", 0)}
            c3d = reg.get("centroid_3d_viewer")
            f = Finding(
                id="",
                type="surface_irregularity",
                confidence=None,
                geometric_deviation_mm=round(float(reg.get("max_abs_deviation_mm", 0.0)), 2),
                description=f"Surface geometric irregularity ({reg.get('sign', 'anomaly')}) detected from 3D surface deviation; {description_reason}",
                bounding_box_2d={
                    "x": float(bbox["x"]),
                    "y": float(bbox["y"]),
                    "width": float(bbox["width"]),
                    "height": float(bbox["height"]),
                },
                region_center_3d=[round(float(v), 2) for v in c3d] if c3d else None,
            )
            findings.append(f)

        sorted_findings = sort_findings(findings)
        if job_dir is not None and image_bgr is not None:
            save_finding_crops(sorted_findings, image_bgr, job_dir, inspection_id)

        return DamageDetectionResult(
            findings=sorted_findings,
            classifier_status=classifier_status,
            classifier_error=classifier_error,
            stats={
                "tiles_evaluated": tiles_eval,
                "positive_tiles": 0,
                "merged_groups": 0,
                "findings_count": len(sorted_findings),
            },
        )

    # 2. Filter positive tiles
    tile_defs = tile_defs or []
    probabilities = probabilities or []
    positive_tiles: List[Dict[str, Any]] = []
    for t_def, prob in zip(tile_defs, probabilities):
        t_copy = dict(t_def)
        t_copy["prob"] = float(prob)
        if prob >= threshold:
            positive_tiles.append(t_copy)

    # 3. Merge positive tiles
    merged_groups = merge_positive_tiles(positive_tiles)

    # 4. Correlate with M1 deviation regions
    findings: List[Finding] = []
    matched_region_indices = set()

    for grp in merged_groups:
        g_box = grp["bbox"]
        g_conf = grp["confidence"]

        matched_reg = None
        best_reg_dev = -1.0
        best_reg_idx = -1

        for idx, reg in enumerate(regions_list):
            reg_box = reg.get("bbox_px") or {"x": reg.get("x", 0), "y": reg.get("y", 0), "width": reg.get("width", 0), "height": reg.get("height", 0)}
            if boxes_overlap(g_box, reg_box) or reg in grp["associated_regions"]:
                dev_val = float(reg.get("max_abs_deviation_mm", 0.0))
                if dev_val > best_reg_dev:
                    best_reg_dev = dev_val
                    best_reg_idx = idx
                    matched_reg = reg

        if matched_reg is not None:
            # Case A: Positive group overlaps deviation region -> potential_crack
            matched_region_indices.add(best_reg_idx)
            reg_box = matched_reg.get("bbox_px") or {"x": matched_reg.get("x", 0), "y": matched_reg.get("y", 0), "width": matched_reg.get("width", 0), "height": matched_reg.get("height", 0)}
            ux1 = min(g_box["x"], reg_box["x"])
            uy1 = min(g_box["y"], reg_box["y"])
            ux2 = max(g_box["x"] + g_box["width"], reg_box["x"] + reg_box["width"])
            uy2 = max(g_box["y"] + g_box["height"], reg_box["y"] + reg_box["height"])
            union_bbox = {"x": float(ux1), "y": float(uy1), "width": float(ux2 - ux1), "height": float(uy2 - uy1)}

            c3d = matched_reg.get("centroid_3d_viewer")
            f = Finding(
                id="",
                type="potential_crack",
                confidence=round(g_conf, 4),
                geometric_deviation_mm=round(float(matched_reg.get("max_abs_deviation_mm", 0.0)), 2),
                description="Potential crack confirmed by image classification and 3D geometric deviation anomaly.",
                bounding_box_2d=union_bbox,
                region_center_3d=[round(float(v), 2) for v in c3d] if c3d else None,
            )
            findings.append(f)
        else:
            # No deviation overlap
            if g_conf >= 0.8:
                # Case B: High confidence visual evidence alone (P >= 0.8) -> potential_crack
                center_3d = estimate_box_center_3d(g_box, disparity, intrinsics)
                f = Finding(
                    id="",
                    type="potential_crack",
                    confidence=round(g_conf, 4),
                    geometric_deviation_mm=None,
                    description="Potential crack identified from visual surface imagery; no measurable depth change.",
                    bounding_box_2d={
                        "x": float(g_box["x"]),
                        "y": float(g_box["y"]),
                        "width": float(g_box["width"]),
                        "height": float(g_box["height"]),
                    },
                    region_center_3d=center_3d,
                )
                findings.append(f)
            else:
                # Case C: Moderate confidence visual evidence (0.5 <= P < 0.8) -> minor_anomaly
                f = Finding(
                    id="",
                    type="minor_anomaly",
                    confidence=round(g_conf, 4),
                    geometric_deviation_mm=None,
                    description="Minor surface visual anomaly with moderate classifier probability.",
                    bounding_box_2d={
                        "x": float(g_box["x"]),
                        "y": float(g_box["y"]),
                        "width": float(g_box["width"]),
                        "height": float(g_box["height"]),
                    },
                    region_center_3d=None,
                )
                findings.append(f)

    # Case D: Unmatched deviation regions -> surface_irregularity
    for idx, reg in enumerate(regions_list):
        if idx in matched_region_indices:
            continue
        reg_box = reg.get("bbox_px") or {"x": reg.get("x", 0), "y": reg.get("y", 0), "width": reg.get("width", 0), "height": reg.get("height", 0)}
        c3d = reg.get("centroid_3d_viewer")
        f = Finding(
            id="",
            type="surface_irregularity",
            confidence=None,
            geometric_deviation_mm=round(float(reg.get("max_abs_deviation_mm", 0.0)), 2),
            description=f"Surface geometric irregularity ({reg.get('sign', 'anomaly')}) detected from 3D surface deviation; no classifier crack evidence.",
            bounding_box_2d={
                "x": float(reg_box["x"]),
                "y": float(reg_box["y"]),
                "width": float(reg_box["width"]),
                "height": float(reg_box["height"]),
            },
            region_center_3d=[round(float(v), 2) for v in c3d] if c3d else None,
        )
        findings.append(f)

    # 5. Sort findings and assign sequential Finding IDs
    sorted_findings = sort_findings(findings)

    if job_dir is not None and image_bgr is not None:
        save_finding_crops(sorted_findings, image_bgr, job_dir, inspection_id)

    return DamageDetectionResult(
        findings=sorted_findings,
        classifier_status=classifier_status,
        classifier_error=None,
        stats={
            "tiles_evaluated": len(tile_defs) if tile_defs else 0,
            "positive_tiles": len(positive_tiles),
            "merged_groups": len(merged_groups),
            "findings_count": len(sorted_findings),
            "potential_cracks": sum(1 for f in sorted_findings if f.type == "potential_crack"),
            "surface_irregularities": sum(1 for f in sorted_findings if f.type == "surface_irregularity"),
            "minor_anomalies": sum(1 for f in sorted_findings if f.type == "minor_anomaly"),
            "device": str(clf_device or "cpu"),
        },
    )


def detect_damage(
    rectified_left_bgr: np.ndarray,
    deviation_regions: List[Any],
    disparity: Optional[np.ndarray] = None,
    intrinsics: Optional[Dict[str, float]] = None,
    tile: int = 256,
    threshold: float = 0.5,
    classifier: Optional[Any] = None,
    model_path: Optional[Union[str, Path]] = None,
    job_dir: Optional[Path] = None,
    inspection_id: Optional[str] = None,
) -> DamageDetectionResult:
    """
    Executes Stage 4 Damage Detection and Potential Crack Classification.

    Combines EfficientNet-B0 visual crack predictions with Stage 3 geometric
    surface deviation regions. Retained as backward-compatible sequential wrapper
    delegating to modular functions.
    """
    # 1. Initialize or resolve classifier
    clf = classifier if classifier is not None else CrackClassifier.get_instance(model_path=model_path)
    classifier_status = getattr(clf, "status", "unavailable")
    classifier_error = getattr(clf, "error", None)

    # If classifier is not ready/available, produce surface_irregularity findings from deviations only
    if classifier_status != "completed":
        return aggregate_damage_findings(
            deviation_regions=deviation_regions,
            classifier_status=classifier_status,
            classifier_error=classifier_error,
            job_dir=job_dir,
            image_bgr=rectified_left_bgr,
            inspection_id=inspection_id,
        )

    # 2. Tile Generation
    tile_defs = generate_tile_definitions(
        image_shape=rectified_left_bgr.shape[:2],
        deviation_regions=deviation_regions,
        tile_size=tile,
        stride=tile,
    )

    # 3. Batch Inference with fallback
    try:
        probabilities = classify_tile_batch(rectified_left_bgr, tile_defs, clf)
    except Exception as exc:
        return aggregate_damage_findings(
            deviation_regions=deviation_regions,
            tile_defs=tile_defs,
            classifier_status="failed",
            classifier_error=f"Inference error: {exc}",
            tiles_evaluated_count=len(tile_defs),
            job_dir=job_dir,
            image_bgr=rectified_left_bgr,
            inspection_id=inspection_id,
        )

    # 4. Aggregate findings and save crops
    return aggregate_damage_findings(
        deviation_regions=deviation_regions,
        tile_defs=tile_defs,
        probabilities=probabilities,
        disparity=disparity,
        intrinsics=intrinsics,
        threshold=threshold,
        classifier_status=classifier_status,
        classifier_error=None,
        clf_device=getattr(clf, "device", "cpu"),
        job_dir=job_dir,
        image_bgr=rectified_left_bgr,
        inspection_id=inspection_id,
    )
