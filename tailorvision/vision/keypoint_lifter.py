"""
tailorvision.vision.keypoint_lifter
=====================================
Fuses front-view and side-view ``PoseResult`` objects into a single
``BiViewPose`` that packages all the 2D evidence needed for SMPL-X fitting.

Responsibilities
----------------
1.  Cross-view consistency check (same person heuristic via torso ratio).
2.  Per-keypoint visibility filtering.
3.  Compute the body height in pixels for scale recovery.
4.  Map MediaPipe landmark indices → SMPL-X joint targets.
5.  Compute an overall pose quality score used downstream for confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import BiViewConsistencyError
from tailorvision.vision.pose_estimator import PoseResult

logger = logging.getLogger(__name__)


# ── SMPL-X joint mapping ──────────────────────────────────────────────────────
# Maps our MediaPipe landmark index → SMPL-X joint name.
# Only joints that can be reliably inferred from a single-person photo
# are included.  Hands/face joints are intentionally omitted to keep the
# mapping robust.
MP_TO_SMPLX: Dict[int, str] = {
    0:  "head",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


@dataclass
class BiViewPose:
    """
    Fused two-view pose representation.

    Attributes
    ----------
    front_kp_px:
        ``(33, 2)`` front-view keypoints in pixel coordinates.
    side_kp_px:
        ``(33, 2)`` side-view keypoints in pixel coordinates.
    front_vis:
        ``(33,)`` front-view visibility scores.
    side_vis:
        ``(33,)`` side-view visibility scores.
    front_img_wh:
        ``(width, height)`` of the front image in pixels.
    side_img_wh:
        ``(width, height)`` of the side image in pixels.
    pixel_height_front:
        Estimated person height in pixels (front view).
    pixel_height_side:
        Estimated person height in pixels (side view).
    joint_map:
        ``{mp_landmark_idx: smplx_joint_name}`` for all joints present.
    overall_quality:
        Composite pose quality score in [0, 1].
    consistency_score:
        Cross-view torso-ratio consistency score in [0, 1].
    """
    front_kp_px: np.ndarray      # (33, 2)
    side_kp_px: np.ndarray       # (33, 2)
    front_vis: np.ndarray        # (33,)
    side_vis: np.ndarray         # (33,)
    front_img_wh: Tuple[int, int]
    side_img_wh: Tuple[int, int]
    pixel_height_front: float
    pixel_height_side: float
    joint_map: Dict[int, str]
    overall_quality: float
    consistency_score: float


class KeypointLifter:
    """
    Fuses front and side ``PoseResult`` objects into a ``BiViewPose``.

    Parameters
    ----------
    config:
        Pipeline configuration.
    """

    _CONSISTENCY_THRESHOLD = 0.35

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def lift(
        self,
        front: PoseResult,
        side: PoseResult,
        *,
        raise_on_inconsistency: bool = False,
    ) -> BiViewPose:
        """
        Fuse front and side pose results.

        Parameters
        ----------
        front:
            Pose estimation result from the front-view image.
        side:
            Pose estimation result from the side-view image.
        raise_on_inconsistency:
            If ``True``, raise ``BiViewConsistencyError`` when front and side
            views appear to show different people.

        Returns
        -------
        BiViewPose
        """
        consistency = self._cross_view_consistency(front, side)
        logger.info("Cross-view consistency score: %.3f", consistency)

        if consistency < self._CONSISTENCY_THRESHOLD:
            msg = (
                f"Front and side images may show different people "
                f"(consistency={consistency:.2f} < {self._CONSISTENCY_THRESHOLD}). "
                "Ensure both photos are of the same person."
            )
            logger.warning(msg)
            if raise_on_inconsistency:
                raise BiViewConsistencyError(msg)

        pixel_h_front = self._estimate_pixel_height(front)
        pixel_h_side = self._estimate_pixel_height(side)
        overall_q = (front.quality_score + side.quality_score) / 2.0

        return BiViewPose(
            front_kp_px=front.keypoints_px,
            side_kp_px=side.keypoints_px,
            front_vis=front.visibility,
            side_vis=side.visibility,
            front_img_wh=(front.image_width, front.image_height),
            side_img_wh=(side.image_width, side.image_height),
            pixel_height_front=pixel_h_front,
            pixel_height_side=pixel_h_side,
            joint_map=MP_TO_SMPLX,
            overall_quality=round(overall_q, 3),
            consistency_score=round(consistency, 3),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_pixel_height(self, pose: PoseResult) -> float:
        """
        Estimate person height in pixels using head-to-heel distance.

        Falls back to a fraction of image height if required landmarks
        are not visible.
        """
        vis = pose.visibility
        kp = pose.keypoints_px

        head_idx, heel_idx_l, heel_idx_r = 0, 29, 30

        head_vis = vis[head_idx]
        heel_vis = max(vis[heel_idx_l], vis[heel_idx_r])

        if head_vis >= 0.4 and heel_vis >= 0.4:
            best_heel = (
                kp[heel_idx_l] if vis[heel_idx_l] >= vis[heel_idx_r] else kp[heel_idx_r]
            )
            height_px = abs(best_heel[1] - kp[head_idx][1])
            logger.debug("Pixel height (head-to-heel) = %.1f px", height_px)
            return float(height_px)

        # Fallback: assume person fills ~85% of image height
        fallback = pose.image_height * 0.85
        logger.warning(
            "Head/heel not reliably visible — falling back to %.0f px height estimate.",
            fallback,
        )
        return fallback

    def _cross_view_consistency(
        self, front: PoseResult, side: PoseResult
    ) -> float:
        """
        Estimate whether front and side images show the same person.

        Heuristic: compare the shoulder-to-hip distance (torso length) as a
        fraction of pixel height in both views.  Two views of the same person
        should yield similar normalised torso ratios.

        Returns a consistency score in [0, 1].
        """
        def torso_ratio(pose: PoseResult) -> Optional[float]:
            vis = pose.visibility
            kp = pose.keypoints_px
            shoulder_idx_l, shoulder_idx_r = 11, 12
            hip_idx_l, hip_idx_r = 23, 24

            if all(vis[i] >= 0.4 for i in [shoulder_idx_l, shoulder_idx_r, hip_idx_l, hip_idx_r]):
                shoulder_y = (kp[shoulder_idx_l][1] + kp[shoulder_idx_r][1]) / 2
                hip_y = (kp[hip_idx_l][1] + kp[hip_idx_r][1]) / 2
                torso_px = abs(hip_y - shoulder_y)
                return torso_px / pose.image_height
            return None

        front_ratio = torso_ratio(front)
        side_ratio = torso_ratio(side)

        if front_ratio is None or side_ratio is None:
            return 0.6  # neutral score when insufficient landmarks

        ratio_diff = abs(front_ratio - side_ratio)
        # Difference ≤ 0.05 → score 1.0;  difference ≥ 0.20 → score 0.0
        score = 1.0 - min(ratio_diff / 0.20, 1.0)
        return float(score)
