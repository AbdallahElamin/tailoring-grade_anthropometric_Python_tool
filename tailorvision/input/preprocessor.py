"""
tailorvision.input.preprocessor
================================
Preprocessing stage: normalize, crop, optionally remove background.

Returns a ``ProcessedImagePair`` containing the cleaned front and side
images ready for pose estimation.

Design notes
------------
- Background removal via ``rembg`` is opt-in (``config.remove_background``).
  When enabled it produces a transparent PNG background; we replace that
  with a neutral grey for downstream models.
- Person crop uses a 15% padding margin to ensure full-body context.
- All output images are resized to 640 px on the longer side while
  preserving aspect ratio — a safe input size for MediaPipe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.input.loader import load_image

logger = logging.getLogger(__name__)

# Target size for the longer dimension of the output image
_TARGET_LONG_SIDE = 640


@dataclass
class ProcessedImagePair:
    """Cleaned image pair ready for pose estimation."""

    front: np.ndarray    # (H, W, 3) uint8 RGB
    side: np.ndarray     # (H, W, 3) uint8 RGB
    front_original: np.ndarray  # unmodified front image (for traceability)
    side_original: np.ndarray   # unmodified side image


def preprocess_image_pair(
    front_path: str | Path,
    side_path: str | Path,
    config: PipelineConfig,
) -> ProcessedImagePair:
    """
    Load, orient, crop, resize, and optionally background-remove both images.

    Parameters
    ----------
    front_path:
        Path to the front-view image.
    side_path:
        Path to the side-view image.
    config:
        Pipeline configuration.

    Returns
    -------
    ProcessedImagePair
    """
    logger.info("Preprocessing front image: %s", Path(front_path).name)
    front_orig = load_image(front_path, as_rgb=True)
    front_proc = _preprocess_single(front_orig, config, view="front")

    logger.info("Preprocessing side image: %s", Path(side_path).name)
    side_orig = load_image(side_path, as_rgb=True)
    side_proc = _preprocess_single(side_orig, config, view="side")

    return ProcessedImagePair(
        front=front_proc,
        side=side_proc,
        front_original=front_orig,
        side_original=side_orig,
    )


# ── Internal ──────────────────────────────────────────────────────────────────

def _preprocess_single(
    img: np.ndarray,
    config: PipelineConfig,
    view: str,
) -> np.ndarray:
    """
    Apply the full preprocessing chain to one image.

    Steps:
    1. Optional background removal
    2. Resize to target long-side
    3. Colour standardisation (ensure uint8 RGB)
    """
    if config.remove_background:
        img = _remove_background(img)

    img = _resize_to_long_side(img, _TARGET_LONG_SIDE)
    img = _ensure_uint8_rgb(img)

    logger.debug("[%s] Preprocessed to shape=%s", view, img.shape)
    return img


def _resize_to_long_side(img: np.ndarray, target: int) -> np.ndarray:
    """Resize image so that its longer side equals ``target`` pixels."""
    h, w = img.shape[:2]
    if max(h, w) == target:
        return img
    scale = target / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure the image is uint8 RGB (3 channels)."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # RGBA → RGB by compositing on grey background
        alpha = img[:, :, 3:4] / 255.0
        bg = np.full_like(img[:, :, :3], 128, dtype=np.uint8)
        img = (img[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
    return img


def _remove_background(img: np.ndarray) -> np.ndarray:
    """
    Remove background using ``rembg`` (optional dependency).

    If ``rembg`` is not installed, logs a warning and returns the original.
    The output is RGBA; downstream ``_ensure_uint8_rgb`` handles compositing.
    """
    try:
        from rembg import remove as rembg_remove  # type: ignore
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_bgra = rembg_remove(img_bgr)
        # rembg returns BGRA — convert to RGBA
        return cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA)
    except ImportError:
        logger.warning(
            "rembg is not installed. Background removal skipped. "
            "Install with: pip install rembg"
        )
        return img
