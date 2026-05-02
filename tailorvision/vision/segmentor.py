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

    Reuses the segmentation mask produced by ``MediapipePoseEstimator``
    (which already requests ``output_segmentation_masks=True``) rather
    than running a separate model.  If the pose estimator produced a mask
    it is stored in ``pose_estimator._last_segmentation_masks`` and this
    class reads it directly.  When used standalone (e.g. in tests) it
    falls back to running a fresh ``PoseLandmarker`` detection.

    Parameters
    ----------
    config:
        Pipeline configuration.
    pose_estimator:
        The ``MediapipePoseEstimator`` instance to reuse masks from.
        Pass ``None`` to run standalone.
    """

    def __init__(self, config: PipelineConfig, pose_estimator=None) -> None:
        self._cfg = config
        self._pose_est = pose_estimator

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Extract the person segmentation mask.

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
            If no mask can be obtained.
        """
        h, w = image.shape[:2]
        raw_mask_np: Optional[np.ndarray] = None

        # --- Try to reuse mask from pose estimator (most efficient path) -----
        if self._pose_est is not None:
            masks = getattr(self._pose_est, "_last_segmentation_masks", None)
            if masks and len(masks) > 0:
                raw_mask_np = masks[0].numpy_view().copy()  # (H, W) float32

        # --- Fallback: run pose landmarker ourselves --------------------------
        if raw_mask_np is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from pathlib import Path

                model_path = (
                    Path(__file__).parent.parent.parent
                    / "models" / "mediapipe" / "pose_landmarker_heavy.task"
                )
                if not model_path.exists():
                    raise SegmentationError(
                        f"MediaPipe model not found at: {model_path}\n"
                        "Download pose_landmarker_heavy.task first."
                    )
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    output_segmentation_masks=True,
                )
                landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                results = landmarker.detect(mp_image)
                landmarker.close()
                if results.segmentation_masks and len(results.segmentation_masks) > 0:
                    raw_mask_np = results.segmentation_masks[0].numpy_view().copy()
            except ImportError as exc:
                raise SegmentationError(
                    "mediapipe is not installed. Run: pip install mediapipe"
                ) from exc

        if raw_mask_np is None:
            raise SegmentationError(
                "MediaPipe segmentation returned no mask. "
                "Ensure the person is clearly visible."
            )

        binary_mask = raw_mask_np > 0.5

        # Morphological clean-up to remove noisy speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_mask_uint8 = binary_mask.astype(np.uint8) * 255
        binary_mask_uint8 = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_CLOSE, kernel)
        binary_mask_uint8 = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_OPEN, kernel)
        binary_mask = binary_mask_uint8 > 127

        person_pixels = raw_mask_np[binary_mask]
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
        pass  # nothing to release (pose estimator manages its own lifecycle)


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
