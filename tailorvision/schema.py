"""
tailorvision.schema
===================
Pydantic v2 data models that define the structured output of the pipeline.

All public types are importable directly from this module:

    from tailorvision.schema import MeasurementResult, QualityReport, ...

The ``MeasurementResult`` is the top-level model returned by
``TailorVisionPipeline.run()`` and serialised to JSON output.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ─────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNRELIABLE = "UNRELIABLE"


class ScaleMode(str, Enum):
    KNOWN_HEIGHT = "known_height"
    HEURISTIC = "heuristic"
    NORMALIZED = "normalized"


class WarningCode(str, Enum):
    LOOSE_CLOTHING_WAIST = "POSSIBLE_LOOSE_CLOTHING_AT_WAIST"
    LOOSE_CLOTHING_CHEST = "POSSIBLE_LOOSE_CLOTHING_AT_CHEST"
    BAD_POSTURE = "NON_UPRIGHT_POSTURE"
    LOW_RESOLUTION = "LOW_IMAGE_RESOLUTION"
    SCALE_UNCERTAIN = "SCALE_UNCERTAIN"
    POOR_KEYPOINTS_FRONT = "POOR_KEYPOINTS_FRONT_VIEW"
    POOR_KEYPOINTS_SIDE = "POOR_KEYPOINTS_SIDE_VIEW"
    POOR_FIT_CONVERGENCE = "POOR_FIT_CONVERGENCE"
    IDENTITY_MISMATCH = "POSSIBLE_IDENTITY_MISMATCH"
    RETAKE_RECOMMENDED = "RETAKE_RECOMMENDED"


# ── Sub-models ────────────────────────────────────────────────────────────────

class SMPLXParameters(BaseModel):
    """Fitted SMPL-X shape parameters."""
    betas: List[float] = Field(description="Shape coefficients β (length ≤ 10)")
    pose_neutralized: bool = Field(
        description="Whether T-pose was used for measurement extraction"
    )
    gender: str = Field(description="Body model gender used for fitting")


class ScaleResult(BaseModel):
    """Metric scale recovery result."""
    mode: ScaleMode
    scale_factor: float = Field(description="Multiplier: model_units → centimetres")
    confidence: float = Field(ge=0.0, le=1.0)
    model_height_cm: Optional[float] = Field(
        None, description="Height predicted by body model before scaling"
    )
    anchor_height_cm: Optional[float] = Field(
        None, description="Known height used as anchor (if provided)"
    )


class QualityScores(BaseModel):
    """Per-stage quality scores (0–1, higher is better)."""
    image_quality_front: float = Field(ge=0.0, le=1.0)
    image_quality_side: float = Field(ge=0.0, le=1.0)
    pose_quality_front: float = Field(ge=0.0, le=1.0)
    pose_quality_side: float = Field(ge=0.0, le=1.0)
    fit_quality: float = Field(ge=0.0, le=1.0)
    scale_quality: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)


class TailoringRecommendations(BaseModel):
    """Garment-specific derived measurements with ease allowances."""
    garment_type: str
    collar_size_cm: Optional[float] = None
    chest_with_ease_cm: Optional[float] = None
    waist_with_ease_cm: Optional[float] = None
    hip_with_ease_cm: Optional[float] = None
    rise_cm: Optional[float] = None
    sleeve_length_cm: Optional[float] = None
    inseam_cm: Optional[float] = None
    shoulder_seam_cm: Optional[float] = None
    ease_table: Dict[str, float] = Field(
        default_factory=dict,
        description="Applied ease allowances per measurement (cm)"
    )


class ValidationReport(BaseModel):
    """Result of input quality gate checks."""
    passed: bool
    front_image_ok: bool
    side_image_ok: bool
    reasons: List[str] = Field(default_factory=list)
    front_blur_score: Optional[float] = None
    side_blur_score: Optional[float] = None
    front_resolution: Optional[tuple] = None
    side_resolution: Optional[tuple] = None


# ── Top-level result ──────────────────────────────────────────────────────────

class MeasurementResult(BaseModel):
    """
    Top-level pipeline output.

    This is the object returned by ``TailorVisionPipeline.run()`` and
    serialised when ``--output result.json`` is passed via the CLI.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    body_model_type: str = Field(description="'smplx' or 'smpl'")
    gender: str

    # ── Model parameters ──────────────────────────────────────────────────────
    smplx_parameters: SMPLXParameters

    # ── Core measurements (all in cm) ─────────────────────────────────────────
    measurements_cm: Dict[str, float] = Field(
        description="Named body measurements in centimetres"
    )

    # ── Uncertainty ───────────────────────────────────────────────────────────
    measurement_confidence: Dict[str, ConfidenceLevel] = Field(
        description="Per-measurement confidence level"
    )
    uncertainty_cm: Dict[str, float] = Field(
        description="Per-measurement ±1σ uncertainty in centimetres"
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scale: ScaleResult

    # ── Quality ───────────────────────────────────────────────────────────────
    quality_scores: QualityScores
    validation_report: ValidationReport
    warnings: List[WarningCode] = Field(default_factory=list)

    # ── Tailoring output ──────────────────────────────────────────────────────
    tailoring_recommendations: TailoringRecommendations

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug_artifact_paths: Dict[str, str] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Serialise result to a pretty-printed JSON string."""
        return self.model_dump_json(indent=indent)

    def save_json(self, path: Path) -> None:
        """Write JSON to ``path``, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
