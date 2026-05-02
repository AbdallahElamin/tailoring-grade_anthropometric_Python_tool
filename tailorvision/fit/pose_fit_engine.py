"""
tailorvision.fit.pose_fit_engine
=================================
Fits SMPL-X shape parameters (betas) to two-view 2D keypoint evidence.

Algorithm
---------
1.  Project current SMPL-X joints onto the image planes of the front
    and side views using a weak-perspective camera model.
2.  Compute reprojection loss against detected MediaPipe keypoints
    (visibility-weighted L2).
3.  Add shape regularisation  ‖β‖² and anthropometric prior penalty.
4.  Backpropagate through the SMPL-X PyTorch model and update β with Adam.
5.  Return the converged β, final loss, and fit-quality score.

Camera model
------------
We use a simple scaled-orthographic (weak-perspective) model:
    u = f * (X / Z_mean) + cx
    v = f * (Y / Z_mean) + cy
where f is estimated from image width (f ≈ 1.2 × W for a ~50 mm FOV).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import FitConvergenceError
from tailorvision.fit.body_model_adapter import BodyModelAdapter
from tailorvision.fit.anthropometric_prior import AnthropometricPrior
from tailorvision.vision.keypoint_lifter import BiViewPose, MP_TO_SMPLX

logger = logging.getLogger(__name__)

# MediaPipe index → SMPL-X joint index (body joints 0-21)
# Mapping derived from SMPL-X joint order (see smplx/joint_names.py)
_MP_TO_SMPLX_IDX: Dict[int, int] = {
    0:  15,   # nose   → head
    11:  16,  # l_shoulder → left_shoulder
    12:  17,  # r_shoulder → right_shoulder
    13:  18,  # l_elbow   → left_elbow
    14:  19,  # r_elbow   → right_elbow
    15:  20,  # l_wrist   → left_wrist
    16:  21,  # r_wrist   → right_wrist
    23:   1,  # l_hip     → left_hip
    24:   2,  # r_hip     → right_hip
    25:   4,  # l_knee    → left_knee
    26:   5,  # r_knee    → right_knee
    27:   7,  # l_ankle   → left_ankle
    28:   8,  # r_ankle   → right_ankle
}


@dataclass
class FitResult:
    """Output of the shape-fitting optimisation."""
    betas: np.ndarray          # (10,) optimal shape parameters
    body_pose: np.ndarray      # (63,) body joint rotations (near-zero for natural pose)
    global_orient: np.ndarray  # (3,)  root orientation
    final_loss: float
    converged: bool
    iterations_run: int
    fit_quality: float         # 0-1 score based on final loss magnitude


class PoseFitEngine:
    """
    Fits SMPL-X shape to BiViewPose keypoints via gradient descent.

    Parameters
    ----------
    config:
        Pipeline configuration.
    adapter:
        Pre-initialised ``BodyModelAdapter``.
    """

    def __init__(self, config: PipelineConfig, adapter: BodyModelAdapter) -> None:
        self._cfg = config
        self._adapter = adapter
        self._prior = AnthropometricPrior(device=config.device)
        self._device = torch.device(config.device)
        torch.manual_seed(config.random_seed)

    def fit(
        self,
        biview: BiViewPose,
        gender: Optional[str] = None,
        initial_betas: Optional[np.ndarray] = None,
    ) -> FitResult:
        """
        Run the shape-fitting optimisation loop.

        Parameters
        ----------
        biview:
            Fused two-view pose from ``KeypointLifter``.
        gender:
            Body model gender.
        initial_betas:
            Starting point for optimisation (zeros if ``None``).

        Returns
        -------
        FitResult
        """
        gender = (gender or self._cfg.gender).lower()
        n_beta = self._cfg.fit_n_shape_params

        # Initialise parameters
        if initial_betas is not None:
            b0 = np.asarray(initial_betas, dtype=np.float32)[:n_beta]
        else:
            b0 = np.zeros(n_beta, dtype=np.float32)

        betas = nn.Parameter(
            torch.tensor(b0, dtype=torch.float32, device=self._device).unsqueeze(0)
        )
        # Body pose near-upright; small values keep the person standing
        body_pose = nn.Parameter(
            torch.zeros((1, 63), dtype=torch.float32, device=self._device)
        )
        global_orient = nn.Parameter(
            torch.zeros((1, 3), dtype=torch.float32, device=self._device)
        )

        optimizer = torch.optim.Adam(
            [betas, body_pose, global_orient],
            lr=self._cfg.fit_lr,
        )

        # Pre-compute target keypoints as tensors
        front_kp, front_vis, front_wh = self._prepare_view(biview, view="front")
        side_kp, side_vis, side_wh   = self._prepare_view(biview, view="side")

        best_loss = float("inf")
        best_betas = b0.copy()
        model = self._adapter._load_model(gender)

        for step in range(self._cfg.fit_iterations):
            optimizer.zero_grad()

            # ── Forward pass ─────────────────────────────────────────────────
            output = model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                expression=torch.zeros((1, 10), device=self._device),
                return_verts=False,
            )
            joints_3d = output.joints  # (1, 127, 3)

            # ── Reprojection losses ───────────────────────────────────────────
            loss_front = self._reproj_loss(
                joints_3d, front_kp, front_vis, front_wh, yaw_deg=0.0
            )
            loss_side = self._reproj_loss(
                joints_3d, side_kp, side_vis, side_wh, yaw_deg=90.0
            )

            # ── Regularisation ────────────────────────────────────────────────
            reg_shape = self._cfg.fit_lambda_shape * (betas ** 2).sum()
            reg_pose  = self._cfg.fit_lambda_pose  * (body_pose ** 2).sum()
            reg_anthr = self._cfg.fit_lambda_anthropometric * self._prior.compute_loss(betas)

            loss = loss_front + loss_side + reg_shape + reg_pose + reg_anthr
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_betas = betas.detach().cpu().numpy().squeeze().copy()

            if step % 50 == 0:
                logger.debug("Fit step %d/%d — loss=%.4f", step, self._cfg.fit_iterations, loss_val)

        converged = best_loss < 0.05
        fit_quality = float(np.clip(1.0 - best_loss / 0.5, 0.0, 1.0))

        if not converged:
            logger.warning(
                "Shape fitting did not fully converge (loss=%.4f). "
                "Results may have higher uncertainty.", best_loss
            )

        # Pad betas back to 10
        full_betas = np.zeros(10, dtype=np.float32)
        full_betas[:n_beta] = best_betas[:n_beta]

        return FitResult(
            betas=full_betas,
            body_pose=body_pose.detach().cpu().numpy().squeeze(),
            global_orient=global_orient.detach().cpu().numpy().squeeze(),
            final_loss=best_loss,
            converged=converged,
            iterations_run=self._cfg.fit_iterations,
            fit_quality=round(fit_quality, 3),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare_view(
        self, biview: BiViewPose, view: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Extract keypoint tensors for one view."""
        if view == "front":
            kp_px = biview.front_kp_px
            vis   = biview.front_vis
            wh    = biview.front_img_wh
        else:
            kp_px = biview.side_kp_px
            vis   = biview.side_vis
            wh    = biview.side_img_wh

        kp_t  = torch.tensor(kp_px, dtype=torch.float32, device=self._device)
        vis_t = torch.tensor(vis,   dtype=torch.float32, device=self._device)
        return kp_t, vis_t, wh

    def _reproj_loss(
        self,
        joints_3d: torch.Tensor,   # (1, 127, 3) metres
        target_kp: torch.Tensor,   # (33, 2) pixels
        visibility: torch.Tensor,  # (33,)
        img_wh: Tuple[int, int],
        yaw_deg: float = 0.0,
    ) -> torch.Tensor:
        """Compute visibility-weighted 2D reprojection loss for one view."""
        w, h = img_wh
        # Weak-perspective focal length heuristic
        f = 1.2 * w

        # Rotate joints for side view (90° yaw around Y-axis)
        joints = joints_3d[0]  # (127, 3)
        if abs(yaw_deg) > 1.0:
            theta = torch.tensor(np.radians(yaw_deg), dtype=torch.float32, device=joints.device)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            Ry = torch.stack([
                torch.stack([ cos_t, torch.zeros_like(cos_t), sin_t]),
                torch.stack([ torch.zeros_like(cos_t), torch.ones_like(cos_t), torch.zeros_like(cos_t)]),
                torch.stack([-sin_t, torch.zeros_like(sin_t), cos_t]),
            ])  # (3,3)
            joints = joints @ Ry.T

        z_mean = joints[:, 2].mean().clamp(min=0.5)
        proj_x = f * joints[:, 0] / z_mean + w / 2
        proj_y = -f * joints[:, 1] / z_mean + h / 2  # Y-flip (image vs 3D)

        loss = torch.zeros(1, device=self._device)
        count = 0
        for mp_idx, smplx_idx in _MP_TO_SMPLX_IDX.items():
            if smplx_idx >= joints.shape[0]:
                continue
            vis_w = visibility[mp_idx]
            if vis_w < 0.3:
                continue
            px = target_kp[mp_idx]
            pred = torch.stack([proj_x[smplx_idx], proj_y[smplx_idx]])
            diff = (pred - px) / torch.tensor([w, h], dtype=torch.float32, device=self._device)
            loss = loss + vis_w * (diff ** 2).sum()
            count += 1

        return loss / max(count, 1)
