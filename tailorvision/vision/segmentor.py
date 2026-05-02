"""
tailorvision.vision.segmentor
==============================
Body segmentation / silhouette extraction.

Uses MediaPipe Pose segmentation mask (available at no extra cost when
``enable_segmentation=True`` is set during pose estimation).  The mask
is used downstream by:

  - ``ScaleRecoveryEngine`` — to estimate person height in pixels more
    accurately than keypoints alone.
  - ``PoseFitEngine`` — for future silhouette-based shape refinement (stub
    implementation provided; full optimisation loop in a future release).

Protocol
--------
Any class implementing ``segment(image) -> SegmentationResult`` satisfies
the ``Segmentor`` protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import SegmentationError

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """
    Binary body segmentation mask.

    Attributes
    ----------
    mask:
        ``(H, W)`` boolean array — ``True`` where the person is present.
    confidence:
        Mean segmentation confidence over the person-labeled pixels [0, 1].
    person_area_fraction:
        Fraction of total image area covered by the person mask.
    contour_pixel_height:
        Vertical extent (top-to-bottom) of the person contour in pixels.
        Used as an alternative height estimate for scale recovery.
    """
    mask: np.ndarray        # (H, W) bool
    confidence: float
    person_area_fraction: float
    contour_pixel_height: float


@runtime_checkable
class Segmentor(Protocol):
    def segment(self, image: np.ndarray) -> SegmentationResult: ...


class MediapipeSegmentor:
    """
    MediaPipe-based body segmentation.

    Reuses the segmentation output from a MediaPipe Pose run rather than
    running a separate model, making it essentially free when pose
    estimation has already been performed.

    Parameters
    ----------
    config:
        Pipeline configuration.
    model_complexity:
        MediaPipe model complexity (0, 1, or 2).  Match this to the value
        used in ``MediapipePoseEstimator`` for consistent results.
    """

    def __init__(self, config: PipelineConfig, model_complexity: int = 2) -> None:
        self._cfg = config
        self._model_complexity = model_complexity
        self._pose = None

    def _get_pose(self):
        if self._pose is None:
            try:
                import mediapipe as mp
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=self._model_complexity,
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                )
            except ImportError as exc:
                raise SegmentationError(
                    "mediapipe is not installed. Run: pip install mediapipe"
                ) from exc
        return self._pose

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Segment the person from the background.

        Parameters
        ----------
        image:
            ``(H, W, 3)`` uint8 RGB image.

        Returns
        -------
        SegmentationResult

        Raises
        ------
        SegmentationError
            If MediaPipe cannot detect a person in the image.
        """
        pose = self._get_pose()
        results = pose.process(image)

        if results.segmentation_mask is None:
            raise SegmentationError(
                "MediaPipe segmentation returned no mask. "
                "Ensure the person is clearly visible."
            )

        # raw_mask: (H, W) float32 in [0, 1] — probability of being person
        raw_mask = results.segmentation_mask
        h, w = raw_mask.shape
        binary_mask = raw_mask > 0.5

        # Morphological clean-up to remove noisy speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_mask_uint8 = binary_mask.astype(np.uint8) * 255
        binary_mask_uint8 = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, kernel)
        binary_mask_uint8 = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_OPEN, kernel)
        binary_mask = binary_mask_uint8 > 127

        person_pixels = raw_mask[binary_mask]
        confidence = float(person_pixels.mean()) if person_pixels.size > 0 else 0.0
        area_fraction = float(binary_mask.sum()) / float(h * w)
        contour_height = self._contour_pixel_height(binary_mask)

        logger.debug(
            "Segmentation: area=%.2f%%, confidence=%.3f, contour_height=%.1f px",
            area_fraction * 100, confidence, contour_height,
        )

        return SegmentationResult(
            mask=binary_mask,
            confidence=confidence,
            person_area_fraction=area_fraction,
            contour_pixel_height=contour_height,
        )

    @staticmethod
    def _contour_pixel_height(mask: np.ndarray) -> float:
        """Compute top-to-bottom extent of the person mask in pixels."""
        rows_with_person = np.where(mask.any(axis=1))[0]
        if rows_with_person.size < 2:
            return 0.0
        return float(rows_with_person[-1] - rows_with_person[0])

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None


class StubSegmentor:
    """Synthetic segmentor for unit tests."""

    def segment(self, image: np.ndarray) -> SegmentationResult:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        # Person occupies middle 60% of width, full height
        x0, x1 = int(w * 0.20), int(w * 0.80)
        y0, y1 = int(h * 0.05), int(h * 0.95)
        mask[y0:y1, x0:x1] = True
        return SegmentationResult(
            mask=mask,
            confidence=0.95,
            person_area_fraction=mask.mean(),
            contour_pixel_height=float(y1 - y0),
        )
