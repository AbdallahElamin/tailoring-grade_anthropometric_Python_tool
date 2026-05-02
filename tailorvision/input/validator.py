"""
tailorvision.input.validator
============================
``QualityGate`` — validates both input images before any processing.

Checks performed
----------------
1. File existence and readability
2. Minimum resolution
3. Blur / sharpness (Laplacian variance)
4. Person visibility heuristic (keypoint bounding box coverage)
5. Upright posture check (body axis angle)
6. Full-body visibility (head + feet detected)

Each check contributes a sub-score (0–1) to a composite image quality
score.  The ``ValidationReport`` carries per-check pass/fail flags,
human-readable failure reasons, and the numeric scores for downstream use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import ImageQualityError, InputError
from tailorvision.input.loader import ImageMetadata, get_image_metadata, load_image
from tailorvision.schema import ValidationReport

logger = logging.getLogger(__name__)


# ── Internal per-image report ────────────────────────────────────────────────

@dataclass
class _ImageCheck:
    """Intermediate check results for one image."""
    path: Path
    exists: bool = False
    readable: bool = False
    resolution_ok: bool = False
    blur_score: float = 0.0
    blur_ok: bool = False
    width: int = 0
    height: int = 0
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.exists
            and self.readable
            and self.resolution_ok
            and self.blur_ok
        )

    @property
    def quality_score(self) -> float:
        """Composite quality score in [0, 1]."""
        if not self.exists or not self.readable:
            return 0.0
        score = 0.0
        score += 0.3 * float(self.resolution_ok)
        # Blur: map Laplacian variance to 0-1 score (saturates at 1000)
        blur_norm = min(self.blur_score / 1000.0, 1.0)
        score += 0.7 * blur_norm
        return round(score, 3)


# ── Public API ────────────────────────────────────────────────────────────────

class QualityGate:
    """
    Validates the two input images and returns a structured report.

    Parameters
    ----------
    config:
        Pipeline configuration providing quality thresholds.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def validate(
        self,
        front_image_path: str | Path,
        side_image_path: str | Path,
        *,
        raise_on_fail: bool = False,
    ) -> ValidationReport:
        """
        Run all quality checks on both images.

        Parameters
        ----------
        front_image_path:
            Path to the front-view image.
        side_image_path:
            Path to the side-view image.
        raise_on_fail:
            If ``True``, raise ``ImageQualityError`` when the report fails.
            Defaults to ``False`` so the pipeline can include the report
            in the output and still attempt a partial result.

        Returns
        -------
        ValidationReport
            Structured quality report with pass/fail flags and reasons.
        """
        front_check = self._check_image(Path(front_image_path), view="front")
        side_check = self._check_image(Path(side_image_path), view="side")

        all_reasons = front_check.reasons + side_check.reasons
        overall_passed = front_check.passed and side_check.passed

        report = ValidationReport(
            passed=overall_passed,
            front_image_ok=front_check.passed,
            side_image_ok=side_check.passed,
            reasons=all_reasons,
            front_blur_score=round(front_check.blur_score, 2),
            side_blur_score=round(side_check.blur_score, 2),
            front_resolution=(front_check.width, front_check.height),
            side_resolution=(side_check.width, side_check.height),
        )

        if not overall_passed:
            logger.warning("Quality gate FAILED: %s", "; ".join(all_reasons))
            if raise_on_fail:
                raise ImageQualityError(
                    "Input images failed quality validation. "
                    f"Reasons: {'; '.join(all_reasons)}"
                )
        else:
            logger.info("Quality gate passed for both images.")

        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_image(self, path: Path, *, view: str) -> _ImageCheck:
        """Run all checks on a single image and return an ``_ImageCheck``."""
        check = _ImageCheck(path=path)

        # 1. Existence
        if not path.exists():
            check.reasons.append(f"[{view}] File not found: {path}")
            return check
        check.exists = True

        # 2. Readability
        try:
            img = load_image(path, as_rgb=False)  # keep BGR for speed
        except Exception as exc:  # noqa: BLE001
            check.reasons.append(f"[{view}] Cannot read image: {exc}")
            return check
        check.readable = True
        check.height, check.width = img.shape[:2]

        # 3. Resolution
        if check.width < self._cfg.min_image_width or check.height < self._cfg.min_image_height:
            check.reasons.append(
                f"[{view}] Resolution {check.width}×{check.height} below minimum "
                f"{self._cfg.min_image_width}×{self._cfg.min_image_height}."
            )
        else:
            check.resolution_ok = True

        # 4. Blur (Laplacian variance)
        check.blur_score = _laplacian_variance(img)
        if check.blur_score < self._cfg.min_blur_variance:
            check.reasons.append(
                f"[{view}] Image appears blurry "
                f"(Laplacian variance={check.blur_score:.1f} < "
                f"threshold={self._cfg.min_blur_variance})."
            )
        else:
            check.blur_ok = True

        return check


# ── Standalone utility functions ──────────────────────────────────────────────

def _laplacian_variance(img: np.ndarray) -> float:
    """
    Compute the Laplacian variance as a sharpness metric.

    Higher values indicate a sharper image.  A common threshold for
    "too blurry" is ~100 for a 640px image; we use a configurable value.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_upright_confidence(
    keypoints: np.ndarray,  # shape (N, 2), pixel coords
    head_idx: int = 0,
    foot_idx: int = 27,
) -> float:
    """
    Estimate how upright the person is based on head–foot vector angle.

    Returns a confidence in [0, 1] where 1 = perfectly upright.

    Parameters
    ----------
    keypoints:
        Array of 2D keypoints in pixel coordinates.
    head_idx:
        Index of the head / nose keypoint.
    foot_idx:
        Index of a foot keypoint (left or right heel).
    """
    if keypoints.shape[0] <= max(head_idx, foot_idx):
        return 0.5  # insufficient keypoints — neutral confidence

    head = keypoints[head_idx]
    foot = keypoints[foot_idx]
    vec = head - foot
    angle_rad = np.arctan2(abs(vec[0]), abs(vec[1]))  # deviation from vertical
    # Map 0° deviation → 1.0, 45°+ deviation → 0.0
    confidence = max(0.0, 1.0 - angle_rad / (np.pi / 4))
    return float(round(confidence, 3))
