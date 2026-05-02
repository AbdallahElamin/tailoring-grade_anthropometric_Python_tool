"""
tailorvision.fit.body_model_adapter
=====================================
Thin wrapper around ``smplx.create(...)`` that provides a clean, typed
interface for the rest of the pipeline.

Key responsibilities
--------------------
1.  Lazy-load the SMPL-X model on first use (avoids startup cost when
    model files are not yet available).
2.  Generate body mesh vertices from shape parameters (betas).
3.  Return T-pose (neutral-pose) vertices — required by SMPL-Anthropometry,
    which measures on zero-pose bodies.
4.  Expose joint locations for 2D keypoint reprojection.
5.  Surface clear errors when the model files are missing.

SMPL-X model files
------------------
Download from: https://smpl-x.is.tue.mpg.de (free registration required,
non-commercial license).  Place files in the directory specified by
``config.smplx_model_dir``:

    models/
    └── smplx/
        ├── SMPLX_MALE.npz
        ├── SMPLX_FEMALE.npz
        └── SMPLX_NEUTRAL.npz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import BodyModelNotFoundError, FitError

logger = logging.getLogger(__name__)

# SMPL-X canonical body height (neutral shape, T-pose) in metres.
# Used as a reference for scale recovery when no height is provided.
SMPLX_CANONICAL_HEIGHT_M = 1.6768   # measured from SMPL-Anthropometry reference


class BodyModelAdapter:
    """
    Lazy-loading adapter for the SMPL-X body model.

    Parameters
    ----------
    config:
        Pipeline configuration providing model directory and device.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._models: Dict[str, object] = {}   # cache per gender
        self._device = torch.device(config.device)

    # ── Public API ──────────────────────────────────────────────────────────

    def get_tpose_vertices(
        self,
        betas: np.ndarray,
        gender: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return SMPL-X mesh vertices for the given shape parameters in T-pose.

        T-pose is achieved by setting all pose parameters to zero, which is
        the convention required by SMPL-Anthropometry for measurement.

        Parameters
        ----------
        betas:
            Shape coefficients, array of length ≤ 10.
        gender:
            ``"male"``, ``"female"``, or ``"neutral"``.  Defaults to
            ``config.gender``.

        Returns
        -------
        np.ndarray
            ``(10475, 3)`` vertex positions in metres (SMPL-X native units).
        """
        gender = self._resolve_gender(gender)
        model = self._load_model(gender)

        betas_t = self._to_tensor(betas, shape=(1, 10))
        # All pose parameters zero → T-pose
        body_pose = torch.zeros((1, 21 * 3), device=self._device, dtype=torch.float32)
        global_orient = torch.zeros((1, 3), device=self._device, dtype=torch.float32)
        expression = torch.zeros((1, 10), device=self._device, dtype=torch.float32)

        with torch.no_grad():
            output = model(
                betas=betas_t,
                body_pose=body_pose,
                global_orient=global_orient,
                expression=expression,
                return_verts=True,
            )

        verts = output.vertices.detach().cpu().numpy().squeeze()  # (10475, 3)
        logger.debug("T-pose vertices generated: shape=%s", verts.shape)
        return verts

    def get_posed_vertices_and_joints(
        self,
        betas: np.ndarray,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        gender: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return posed vertices and joint positions.

        Parameters
        ----------
        betas:
            Shape coefficients ``(10,)``.
        body_pose:
            Axis-angle body pose ``(63,)`` — 21 joints × 3.
        global_orient:
            Root orientation axis-angle ``(3,)``.
        gender:
            Body model gender.

        Returns
        -------
        vertices:
            ``(10475, 3)`` posed mesh vertices in metres.
        joints:
            ``(127, 3)`` joint positions in metres.
        """
        gender = self._resolve_gender(gender)
        model = self._load_model(gender)

        betas_t = self._to_tensor(betas, shape=(1, 10))
        pose_t = self._to_tensor(body_pose, shape=(1, 63))
        orient_t = self._to_tensor(global_orient, shape=(1, 3))
        expression = torch.zeros((1, 10), device=self._device, dtype=torch.float32)

        with torch.no_grad():
            output = model(
                betas=betas_t,
                body_pose=pose_t,
                global_orient=orient_t,
                expression=expression,
                return_verts=True,
            )

        verts = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        return verts, joints

    def get_joints_from_betas(
        self,
        betas: np.ndarray,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        gender: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convenience method: return only the joint positions.

        Returns
        -------
        np.ndarray
            ``(127, 3)`` joint positions in metres.
        """
        _, joints = self.get_posed_vertices_and_joints(
            betas, body_pose, global_orient, gender
        )
        return joints

    def model_height_m(
        self, betas: np.ndarray, gender: Optional[str] = None
    ) -> float:
        """
        Return the height of the body model (T-pose) in metres.

        Height is computed as the vertical extent of the mesh from the
        lowest foot vertex to the highest head vertex.
        """
        verts = self.get_tpose_vertices(betas, gender)
        return float(verts[:, 1].max() - verts[:, 1].min())

    # ── Private helpers ─────────────────────────────────────────────────────

    def _resolve_gender(self, gender: Optional[str]) -> str:
        return (gender or self._cfg.gender).lower()

    def _load_model(self, gender: str):
        """Load and cache the SMPL-X model for ``gender``."""
        if gender in self._models:
            return self._models[gender]

        model_dir = self._cfg.smplx_model_dir
        if not model_dir.exists():
            raise BodyModelNotFoundError(
                f"SMPL-X model directory not found: {model_dir}\n"
                "Please download model files from https://smpl-x.is.tue.mpg.de\n"
                "and place SMPLX_MALE.npz / SMPLX_FEMALE.npz / SMPLX_NEUTRAL.npz\n"
                f"into: {model_dir}"
            )

        try:
            import smplx  # type: ignore
        except ImportError as exc:
            raise FitError(
                "smplx is not installed. Run: pip install smplx[all]"
            ) from exc

        gender_map = {"male": "male", "female": "female", "neutral": "neutral"}
        resolved = gender_map.get(gender, "neutral")

        logger.info("Loading SMPL-X model (gender=%s) from %s ...", resolved, model_dir)
        try:
            model = smplx.create(
                str(model_dir),
                model_type="smplx",
                gender=resolved,
                num_betas=10,
                use_face_contour=False,
                ext="npz",
            ).to(self._device)
        except Exception as exc:
            raise BodyModelNotFoundError(
                f"Failed to load SMPL-X model for gender='{resolved}': {exc}"
            ) from exc

        model.eval()
        self._models[gender] = model
        logger.info("SMPL-X model loaded successfully.")
        return model

    def _to_tensor(self, arr: np.ndarray, shape: Tuple) -> torch.Tensor:
        """Pad or truncate ``arr`` to ``shape`` and convert to device tensor."""
        arr = np.asarray(arr, dtype=np.float32).ravel()
        target_len = shape[-1]
        if arr.size < target_len:
            arr = np.pad(arr, (0, target_len - arr.size))
        else:
            arr = arr[:target_len]
        return torch.tensor(arr, dtype=torch.float32, device=self._device).unsqueeze(0)
