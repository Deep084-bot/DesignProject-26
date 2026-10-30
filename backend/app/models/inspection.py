"""
Inspection domain models and API response schemas for Surface.
Compatible with frontend TypeScript interfaces.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from .calibration import CalibrationSummary


PipelineStatus = Literal["waiting", "active", "completed", "failed"]
InspectionStatus = Literal["queued", "processing", "completed", "failed"]
FindingType = Literal["potential_crack", "surface_irregularity", "minor_anomaly"]


DEFAULT_STAGES = [
    {
        "id": "prepare",
        "label": "Preparing images",
        "description": "Verifying stereo pair dimensions and aligning camera perspectives.",
    },
    {
        "id": "depth",
        "label": "Reconstructing depth",
        "description": "Calculating spatial displacement between left and right views.",
    },
    {
        "id": "surface",
        "label": "Building the 3D surface",
        "description": "Generating calibrated 3D geometric surface coordinates.",
    },
    {
        "id": "deviations",
        "label": "Analysing surface deviations",
        "description": "Measuring out-of-plane geometric deviations against reference plane.",
    },
    {
        "id": "damage",
        "label": "Identifying potential damage",
        "description": "Correlating visual surface patterns with measured geometric anomalies.",
    },
    {
        "id": "results",
        "label": "Preparing results",
        "description": "Compiling findings, 3D model, and inspection report.",
    },
]


class PipelineStep(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    label: str
    description: str
    status: PipelineStatus = "waiting"


class ImageFileMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    url: str
    dimensions: Optional[str] = None
    size_bytes: Optional[int] = None


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    type: FindingType = "surface_irregularity"
    confidence: Optional[float] = None
    geometric_deviation_mm: Optional[float] = None
    deviation_unit: str = "mm"
    description: str = "Surface observation."
    bounding_box_2d: Optional[Dict[str, float]] = None
    region_center_3d: Optional[List[float]] = None
    image_crop_url: Optional[str] = None


class InspectionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    surface_type: str
    status: InspectionStatus = "queued"
    current_step_index: int = 0
    steps: List[PipelineStep] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reconstruction_method: str = "Automatic surface reconstruction"
    left_image: Optional[ImageFileMeta] = None
    right_image: Optional[ImageFileMeta] = None
    calibration: Optional[CalibrationSummary] = None
    processing_duration_sec: Optional[float] = None
    point_cloud_url: Optional[str] = None
    point_count: Optional[int] = None
    mesh_url: Optional[str] = None
    triangle_count: Optional[int] = None
    mesh_status: Optional[str] = None
    mesh_error: Optional[str] = None
    findings: List[Finding] = Field(default_factory=list)
    failure_message: Optional[str] = None
    is_demo_mode: bool = False

    @classmethod
    def create_initial(
        cls,
        inspection_id: str,
        name: str,
        surface_type: str,
        reconstruction_method: str,
        left_meta: ImageFileMeta,
        right_meta: ImageFileMeta,
        calib_summary: Optional[CalibrationSummary],
    ) -> "InspectionResponse":
        steps = [
            PipelineStep(
                id=stage["id"],
                label=stage["label"],
                description=stage["description"],
                status="active" if i == 0 else "waiting",
            )
            for i, stage in enumerate(DEFAULT_STAGES)
        ]
        return cls(
            id=inspection_id,
            name=name,
            surface_type=surface_type,
            status="queued",
            current_step_index=0,
            steps=steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            reconstruction_method=reconstruction_method,
            left_image=left_meta,
            right_image=right_meta,
            calibration=calib_summary,
            is_demo_mode=False,
        )
