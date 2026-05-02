"""
tailorvision.config
===================
Central configuration object for the TailorVision pipeline.

All tuneable parameters live here.  Pass a modified ``PipelineConfig``
instance to ``TailorVisionPipeline`` to override defaults without
touching source code.

Example
-------
>>> from tailorvision.config import PipelineConfig
>>> cfg = PipelineConfig(known_height_cm=175.0, gender="male")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


# ── Paths ────────────────────────────────────────────────────────────────────

# Default model directory: project_root/models/
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


@dataclass
class PipelineConfig:
    """
    Master configuration for the TailorVision pipeline.

    Attributes
    ----------
    model_dir:
        Directory containing SMPL-X model files (``smplx/SMPLX_*.npz``).
    smpl_anthropometry_dir:
        Directory of the cloned SMPL-Anthropometry repository.
    gender:
        Body model gender — ``"male"``, ``"female"``, or ``"neutral"``.
    known_height_cm:
        Client's actual height in centimetres.  When provided, used as the
        primary scale anchor.  Set to ``None`` to fall back to heuristic
        estimation.
    garment_type:
        Controls which ease-allowance table is applied.
        Supported: ``"traditional"``, ``"suit"``, ``"shirt"``, ``"trousers"``.
    output_dir:
        Directory where debug artefacts and JSON results are written.
    device:
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Fitting hyper-parameters
    -------------------------
    fit_iterations:
        Number of Adam optimizer steps during shape fitting.
    fit_lr:
        Learning rate for the Adam optimizer.
    fit_lambda_shape:
        Weight of the shape regularization term ``‖β‖²``.
    fit_lambda_pose:
        Weight of the pose regularization term.
    fit_lambda_anthropometric:
        Weight of anthropometric prior soft constraints.
    fit_n_shape_params:
        Number of SMPL-X beta parameters to optimize (max 10, fewer is faster
        but less expressive).

    Uncertainty estimation
    ----------------------
    uncertainty_n_samples:
        Number of perturbed fitting runs for Monte-Carlo uncertainty.
    uncertainty_keypoint_noise_px:
        Standard deviation of Gaussian noise added to keypoints (pixels).

    Quality gates
    -------------
    min_image_width:
        Minimum acceptable image width in pixels.
    min_image_height:
        Minimum acceptable image height in pixels.
    min_blur_variance:
        Minimum Laplacian variance before image is flagged as blurry.
    min_keypoint_visibility:
        MediaPipe visibility threshold below which a keypoint is discarded.
    min_visible_body_keypoints:
        Minimum number of high-confidence keypoints required per view.

    Miscellaneous
    -------------
    remove_background:
        Whether to run background removal (requires ``rembg``).
    save_debug_artifacts:
        Whether to write pose overlays, mesh OBJ, etc. to ``output_dir``.
    random_seed:
        Seed for all random number generators (reproducibility).
    log_level:
        Python logging level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    """

    # ── Paths ────────────────────────────────────────────────────────────────
    model_dir: Path = field(default_factory=lambda: _DEFAULT_MODEL_DIR)
    smpl_anthropometry_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "third_party" / "SMPL-Anthropometry"
    )
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # ── Body model ────────────────────────────────────────────────────────────
    gender: Literal["male", "female", "neutral"] = "neutral"
    model_type: Literal["smplx", "smpl"] = "smplx"

    # ── Client info ───────────────────────────────────────────────────────────
    known_height_cm: Optional[float] = None
    garment_type: Literal["traditional", "suit", "shirt", "trousers"] = "traditional"

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = "cpu"  # override to "cuda" if GPU is available

    # ── Fitting ──────────────────────────────────────────────────────────────
    fit_iterations: int = 300
    fit_lr: float = 5e-3
    fit_lambda_shape: float = 0.01
    fit_lambda_pose: float = 0.001
    fit_lambda_anthropometric: float = 0.005
    fit_n_shape_params: int = 10

    # ── Uncertainty ──────────────────────────────────────────────────────────
    uncertainty_n_samples: int = 5
    uncertainty_keypoint_noise_px: float = 3.0

    # ── Quality gates ─────────────────────────────────────────────────────────
    min_image_width: int = 480
    min_image_height: int = 640
    min_blur_variance: float = 80.0
    min_keypoint_visibility: float = 0.5
    min_visible_body_keypoints: int = 20

    # ── Misc ─────────────────────────────────────────────────────────────────
    remove_background: bool = False
    save_debug_artifacts: bool = True
    random_seed: int = 42
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Coerce path types and validate critical values."""
        self.model_dir = Path(self.model_dir)
        self.smpl_anthropometry_dir = Path(self.smpl_anthropometry_dir)
        self.output_dir = Path(self.output_dir)

        if self.fit_n_shape_params < 1 or self.fit_n_shape_params > 10:
            raise ValueError("fit_n_shape_params must be in [1, 10].")
        if self.uncertainty_n_samples < 1:
            raise ValueError("uncertainty_n_samples must be ≥ 1.")

    @property
    def smplx_model_dir(self) -> Path:
        """Convenience shortcut to the smplx sub-folder of ``model_dir``."""
        return self.model_dir / "smplx"

    def output_path(self, filename: str) -> Path:
        """Return an absolute path inside ``output_dir`` and create it."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / filename
