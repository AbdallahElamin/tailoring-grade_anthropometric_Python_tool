"""
tailorvision.vision.pose_estimator
===================================
2D human pose estimation with a pluggable backend protocol.

Architecture
------------
``PoseEstimator`` is a Python ``Protocol`` (structural typing).  Any class
that implements ``estimate(image) -> PoseResult`` satisfies it.

Included implementations
------------------------
``MediapipePoseEstimator``
    Uses Google MediaPipe Pose (33 landmarks, no weight download required at
    runtime — weights are bundled inside the ``mediapipe`` package).
    Robust, fast (~30 ms/image on CPU), sufficient for our use case.

``StubPoseEstimator``
    Returns synthetic keypoints for testing without loading any model.

Upgrade path (not yet implemented — marked with TODO)
------------------------------------------------------
``MMPosePoseEstimator``
    High-accuracy ViTPose / RTMPose via the MMPose framework.
    Requires ``mmpose``, ``mmcv``, and a model checkpoint download.
    Provides higher keypoint accuracy, especially for occluded joints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import PoseEstimationError

logger = logging.getLogger(__name__)

# ── MediaPipe landmark indices (subset used in fitting) ───────────────────────
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30


@dataclass
class PoseResult:
    """
    Output of a single-image pose estimation call.

    Attributes
    ----------
    keypoints:
        ``(33, 2)`` array of (x, y) landmark positions in **normalised**
        image coordinates [0, 1].  Row order follows the MediaPipe landmark
        index convention.
    visibility:
        ``(33,)`` float array, per-landmark visibility confidence in [0, 1].
    keypoints_px:
        ``(33, 2)`` array of (x, y) positions in **pixel** coordinates.
    image_width:
        Source image width in pixels (used to back-project normalised → px).
    image_height:
        Source image height in pixels.
    bounding_box:
        ``(x_min, y_min, x_max, y_max)`` in pixel coordinates of the
        detected person's bounding box.
    quality_score:
        Fraction of keypoints with visibility ≥ 0.5 — proxy for detection
        quality in [0, 1].
    """
    keypoints: np.ndarray        # (33, 2) normalised
    visibility: np.ndarray       # (33,)
    keypoints_px: np.ndarray     # (33, 2) pixels
    image_width: int
    image_height: int
    bounding_box: tuple          # (x_min, y_min, x_max, y_max)
    quality_score: float


@runtime_checkable
class PoseEstimator(Protocol):
    """Protocol satisfied by any pose estimation backend."""

    def estimate(self, image: np.ndarray) -> PoseResult:
        """
        Estimate 2D pose from an RGB image.

        Parameters
        ----------
        image:
            ``(H, W, 3)`` uint8 RGB image.

        Returns
        -------
        PoseResult
        """
        ...


# ── MediaPipe implementation ──────────────────────────────────────────────────

# Default model path (relative to project root, downloaded once)
_DEFAULT_TASK_MODEL = (
    Path(__file__).parent.parent.parent / "models" / "mediapipe" / "pose_landmarker_heavy.task"
)


class MediapipePoseEstimator:
    """
    Google MediaPipe Pose estimator.

    Uses the MediaPipe Tasks ``PoseLandmarker`` API (mediapipe ≥ 0.10.30).
    A ``pose_landmarker_heavy.task`` model file must be present at
    ``models/mediapipe/pose_landmarker_heavy.task`` (downloaded automatically
    by the setup step, or manually from the MediaPipe model zoo).

    Parameters
    ----------
    config:
        Pipeline configuration (provides ``min_keypoint_visibility``).
    model_path:
        Path to the ``.task`` bundle.  Defaults to
        ``models/mediapipe/pose_landmarker_heavy.task``.
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_path: Optional[Path] = None,
    ) -> None:
        self._cfg = config
        self._model_path = Path(model_path) if model_path else _DEFAULT_TASK_MODEL
        self._landmarker = None  # lazy-loaded

    def _get_landmarker(self):
        """Lazily initialise the MediaPipe PoseLandmarker."""
        if self._landmarker is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python.core.base_options import BaseOptions

                if not self._model_path.exists():
                    raise PoseEstimationError(
                        f"MediaPipe model not found at: {self._model_path}\n"
                        "Download it with:\n"
                        "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                        "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task "
                        "-OutFile models/mediapipe/pose_landmarker_heavy.task"
                    )

                options = mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self._model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=True,
                )
                self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                logger.debug("MediaPipe PoseLandmarker initialised from %s.", self._model_path)
            except ImportError as exc:
                raise PoseEstimationError(
                    "mediapipe is not installed. Run: pip install mediapipe"
                ) from exc
        return self._landmarker

    def estimate(self, image: np.ndarray) -> PoseResult:
        """
        Run MediaPipe PoseLandmarker on a single RGB image.

        Parameters
        ----------
        image:
            ``(H, W, 3)`` uint8 RGB image.

        Returns
        -------
        PoseResult

        Raises
        ------
        PoseEstimationError
            If no person is detected or fewer than ``min_visible_body_keypoints``
            landmarks have sufficient confidence.
        """
        import mediapipe as mp

        h, w = image.shape[:2]
        landmarker = self._get_landmarker()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = landmarker.detect(mp_image)

        if not results.pose_landmarks:
            raise PoseEstimationError(
                "MediaPipe Pose: no person detected in image. "
                "Ensure the person is fully visible and the image is well-lit."
            )

        lm = results.pose_landmarks[0]  # first (and only) person
        keypoints_norm = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        visibility = np.array(
            [p.visibility if p.visibility is not None else 0.0 for p in lm],
            dtype=np.float32,
        )
        keypoints_px = keypoints_norm * np.array([w, h], dtype=np.float32)

        # Store segmentation mask on self for the segmentor to reuse
        self._last_segmentation_masks = results.segmentation_masks

        # Quality score: fraction of landmarks with visibility >= threshold
        vis_thresh = self._cfg.min_keypoint_visibility
        n_visible = int((visibility >= vis_thresh).sum())
        quality_score = round(n_visible / len(lm), 3)

        if n_visible < self._cfg.min_visible_body_keypoints:
            raise PoseEstimationError(
                f"Only {n_visible}/{len(lm)} keypoints are sufficiently visible "
                f"(threshold={vis_thresh}). "
                "Try a clearer image with the full body in frame."
            )

        # Bounding box from visible keypoints
        visible_pts = keypoints_px[visibility >= vis_thresh]
        x_min, y_min = visible_pts.min(axis=0)
        x_max, y_max = visible_pts.max(axis=0)
        bbox = (float(x_min), float(y_min), float(x_max), float(y_max))

        logger.info(
            "Pose estimated: %d/%d keypoints visible, quality=%.2f",
            n_visible, len(lm), quality_score,
        )

        return PoseResult(
            keypoints=keypoints_norm,
            visibility=visibility,
            keypoints_px=keypoints_px,
            image_width=w,
            image_height=h,
            bounding_box=bbox,
            quality_score=quality_score,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


# ── Stub implementation (for testing) ────────────────────────────────────────

class StubPoseEstimator:
    """
    Synthetic pose estimator that returns plausible T-pose keypoints.

    Useful for unit tests where no real image or model is available.
    The returned keypoints are in normalised coordinates representing
    a person standing with arms at sides, viewed from the front.
    """

    def estimate(self, image: np.ndarray) -> PoseResult:
        h, w = image.shape[:2]

        # Generate a synthetic T-pose skeleton in normalised coordinates.
        # Indices follow MediaPipe 33-landmark convention.
        kp = np.zeros((33, 2), dtype=np.float32)
        kp[0] = [0.50, 0.06]   # nose
        kp[7] = [0.48, 0.04]   # left ear
        kp[8] = [0.52, 0.04]   # right ear
        kp[11] = [0.40, 0.28]  # left shoulder
        kp[12] = [0.60, 0.28]  # right shoulder
        kp[13] = [0.38, 0.44]  # left elbow
        kp[14] = [0.62, 0.44]  # right elbow
        kp[15] = [0.37, 0.58]  # left wrist
        kp[16] = [0.63, 0.58]  # right wrist
        kp[23] = [0.44, 0.52]  # left hip
        kp[24] = [0.56, 0.52]  # right hip
        kp[25] = [0.44, 0.70]  # left knee
        kp[26] = [0.56, 0.70]  # right knee
        kp[27] = [0.44, 0.88]  # left ankle
        kp[28] = [0.56, 0.88]  # right ankle
        kp[29] = [0.43, 0.92]  # left heel
        kp[30] = [0.57, 0.92]  # right heel

        vis = np.ones(33, dtype=np.float32) * 0.95
        kp_px = kp * np.array([w, h], dtype=np.float32)

        return PoseResult(
            keypoints=kp,
            visibility=vis,
            keypoints_px=kp_px,
            image_width=w,
            image_height=h,
            bounding_box=(
                float(kp_px[:, 0].min()),
                float(kp_px[:, 1].min()),
                float(kp_px[:, 0].max()),
                float(kp_px[:, 1].max()),
            ),
            quality_score=1.0,
        )


# ── TODO: MMPose upgrade path ────────────────────────────────────────────────
# class MMPosePoseEstimator:
#     """
#     High-accuracy pose estimation via MMPose ViTPose.
#     TODO: Implement when mmpose and mmcv are available.
#     """
#     def estimate(self, image: np.ndarray) -> PoseResult:
#         raise NotImplementedError("MMPose backend not yet implemented.")
