"""
tailorvision.quality.quality_reporter
========================================
Aggregates quality scores from all pipeline stages and emits structured warnings.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from tailorvision.schema import (
    ConfidenceLevel, QualityScores, ScaleResult, ValidationReport, WarningCode,
)
from tailorvision.vision.keypoint_lifter import BiViewPose

if TYPE_CHECKING:
    from tailorvision.fit.pose_fit_engine import FitResult

logger = logging.getLogger(__name__)


class QualityReporter:
    """Aggregates all stage scores into ``QualityScores`` and a warning list."""

    def build(
        self,
        validation: ValidationReport,
        biview: BiViewPose,
        fit: "FitResult",
        scale: ScaleResult,
        measurements_cm: Dict[str, float],
    ) -> tuple:
        """
        Compute quality scores and generate warnings.

        Returns
        -------
        (QualityScores, List[WarningCode])
        """
        warnings: List[WarningCode] = []

        # ── Image quality ─────────────────────────────────────────────────────
        img_q_front = self._blur_to_score(validation.front_blur_score)
        img_q_side  = self._blur_to_score(validation.side_blur_score)

        if not validation.front_image_ok:
            img_q_front = min(img_q_front, 0.3)
        if not validation.side_image_ok:
            img_q_side = min(img_q_side, 0.3)

        # ── Pose quality ──────────────────────────────────────────────────────
        pose_q_front = float(biview.overall_quality)
        pose_q_side  = float(biview.overall_quality)

        if biview.overall_quality < 0.55 or biview.consistency_score < 0.50:
            warnings.append(WarningCode.BAD_POSTURE)

        if biview.overall_quality < 0.55:
            warnings.append(WarningCode.POOR_KEYPOINTS_FRONT)

        # ── Fit quality ───────────────────────────────────────────────────────
        fit_q = float(fit.fit_quality)
        if not fit.converged:
            warnings.append(WarningCode.POOR_FIT_CONVERGENCE)

        # ── Scale quality ─────────────────────────────────────────────────────
        scale_q = float(scale.confidence)
        if scale.confidence < 0.60:
            warnings.append(WarningCode.SCALE_UNCERTAIN)

        # ── Anthropometric warnings ───────────────────────────────────────────
        waist = measurements_cm.get("waist_circumference")
        hip   = measurements_cm.get("hip_circumference")
        chest = measurements_cm.get("chest_circumference")

        if waist and hip and waist / hip < 0.72:
            warnings.append(WarningCode.LOOSE_CLOTHING_WAIST)
        if chest and waist and chest / waist > 1.30:
            warnings.append(WarningCode.LOOSE_CLOTHING_CHEST)

        # ── Overall score ─────────────────────────────────────────────────────
        overall = float(
            0.20 * img_q_front + 0.15 * img_q_side
            + 0.20 * pose_q_front + 0.15 * pose_q_side
            + 0.20 * fit_q + 0.10 * scale_q
        )

        if overall < 0.45 and warnings:
            warnings.append(WarningCode.RETAKE_RECOMMENDED)

        scores = QualityScores(
            image_quality_front=round(img_q_front, 3),
            image_quality_side=round(img_q_side, 3),
            pose_quality_front=round(pose_q_front, 3),
            pose_quality_side=round(pose_q_side, 3),
            fit_quality=round(fit_q, 3),
            scale_quality=round(scale_q, 3),
            overall=round(overall, 3),
        )

        logger.info(
            "Quality report: overall=%.2f, warnings=%s",
            overall, [w.value for w in warnings],
        )
        return scores, warnings

    @staticmethod
    def _blur_to_score(blur: Optional[float]) -> float:
        """Map Laplacian variance to a 0-1 score (saturates at 1000)."""
        if blur is None:
            return 0.5
        return float(min(blur / 1000.0, 1.0))
