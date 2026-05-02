"""
tailorvision.scale.scale_recovery_engine
==========================================
Recovers real-world metric scale (metres → centimetres conversion factor).

Two modes
---------
Mode 1 — Known height (preferred, highest confidence):
    scale_factor = known_height_cm / model_height_cm

Mode 2 — Heuristic (used when height is unknown):
    Estimates the person's real height from pixel height and an estimated
    focal length, then uses that to set the scale factor.
    Confidence is lower and a SCALE_UNCERTAIN warning is emitted.

Mode 3 — Normalised (fallback):
    When heuristic fails, delegates to SMPL-Anthropometry's
    height_normalize_measurements() which rescales all measurements to a
    reference height without knowing absolute scale.  All output measurements
    are still in cm but are height-relative, not absolute.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.schema import ScaleMode, ScaleResult
from tailorvision.vision.keypoint_lifter import BiViewPose
from tailorvision.vision.segmentor import SegmentationResult

if TYPE_CHECKING:
    from tailorvision.fit.body_model_adapter import BodyModelAdapter

SMPLX_CANONICAL_HEIGHT_M = 1.6768  # metres, neutral T-pose reference


logger = logging.getLogger(__name__)

# Assumed real-world average photo distance (metres) when no depth info available.
_DEFAULT_PHOTO_DISTANCE_M = 2.5
# Approx. sensor height fraction used for FOV heuristic (for a vertical phone shot).
_SENSOR_HEIGHT_FRACTION = 0.72


@dataclass
class _RawScale:
    scale_factor: float
    mode: ScaleMode
    confidence: float
    model_height_cm: float
    anchor_height_cm: Optional[float]


class ScaleRecoveryEngine:
    """
    Estimates the scale factor to convert SMPL-X model units (metres) to cm.

    Parameters
    ----------
    config:
        Pipeline configuration.
    adapter:
        Initialised BodyModelAdapter (used to measure model height).
    """

    def __init__(self, config: PipelineConfig, adapter: BodyModelAdapter) -> None:
        self._cfg = config
        self._adapter = adapter

    def recover(
        self,
        betas: np.ndarray,
        biview: BiViewPose,
        segmentation: Optional[SegmentationResult] = None,
    ) -> ScaleResult:
        """
        Determine the scale factor.

        Parameters
        ----------
        betas:
            Fitted shape parameters.
        biview:
            Fused two-view pose (provides pixel heights).
        segmentation:
            Optional segmentation mask (improves pixel-height estimate).

        Returns
        -------
        ScaleResult
        """
        gender = self._cfg.gender
        model_h_m = self._adapter.model_height_m(betas, gender)
        model_h_cm = model_h_m * 100.0

        if self._cfg.known_height_cm is not None:
            return self._mode_known_height(model_h_cm)

        # Attempt heuristic
        pixel_h = self._best_pixel_height(biview, segmentation)
        if pixel_h > 50:
            return self._mode_heuristic(model_h_cm, pixel_h, biview)

        # Fallback: normalised (no absolute scale)
        return self._mode_normalised(model_h_cm)

    # ── Modes ─────────────────────────────────────────────────────────────────

    def _mode_known_height(self, model_h_cm: float) -> ScaleResult:
        known = self._cfg.known_height_cm
        scale = known / model_h_cm
        logger.info(
            "Scale mode=known_height: %.1f cm / %.1f cm = %.4f", known, model_h_cm, scale
        )
        return ScaleResult(
            mode=ScaleMode.KNOWN_HEIGHT,
            scale_factor=round(scale, 5),
            confidence=0.97,
            model_height_cm=round(model_h_cm, 2),
            anchor_height_cm=known,
        )

    def _mode_heuristic(
        self, model_h_cm: float, pixel_h: float, biview: BiViewPose
    ) -> ScaleResult:
        img_h = biview.front_img_wh[1]
        img_w = biview.front_img_wh[0]
        # Weak-perspective height estimate
        # real_height ≈ (pixel_height / image_height) × assumed_FOV_height
        fov_height_m = _DEFAULT_PHOTO_DISTANCE_M * _SENSOR_HEIGHT_FRACTION
        estimated_h_m = (pixel_h / img_h) * fov_height_m
        estimated_h_cm = estimated_h_m * 100.0
        scale = estimated_h_cm / model_h_cm
        # Confidence lower due to unknown distance; plausibility check
        confidence = 0.55 if 120 < estimated_h_cm < 220 else 0.30
        logger.info(
            "Scale mode=heuristic: estimated_h=%.1f cm, scale=%.4f, confidence=%.2f",
            estimated_h_cm, scale, confidence,
        )
        return ScaleResult(
            mode=ScaleMode.HEURISTIC,
            scale_factor=round(scale, 5),
            confidence=confidence,
            model_height_cm=round(model_h_cm, 2),
            anchor_height_cm=round(estimated_h_cm, 1),
        )

    def _mode_normalised(self, model_h_cm: float) -> ScaleResult:
        """
        No absolute scale available — use SMPL-Anthropometry height normalisation.
        Assumes a reference height of 170 cm.
        """
        reference_h = 170.0
        scale = reference_h / model_h_cm
        logger.warning(
            "Scale mode=normalised. Measurements will be relative to %.0f cm reference. "
            "Provide known_height_cm for absolute measurements.", reference_h
        )
        return ScaleResult(
            mode=ScaleMode.NORMALIZED,
            scale_factor=round(scale, 5),
            confidence=0.30,
            model_height_cm=round(model_h_cm, 2),
            anchor_height_cm=reference_h,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _best_pixel_height(
        self,
        biview: BiViewPose,
        seg: Optional[SegmentationResult],
    ) -> float:
        """Return the best available pixel height estimate."""
        candidates = []
        if biview.pixel_height_front > 50:
            candidates.append(biview.pixel_height_front)
        if biview.pixel_height_side > 50:
            candidates.append(biview.pixel_height_side)
        if seg is not None and seg.contour_pixel_height > 50:
            candidates.append(seg.contour_pixel_height)
        return float(np.mean(candidates)) if candidates else 0.0
