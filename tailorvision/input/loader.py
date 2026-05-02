"""
tailorvision.input.loader
=========================
Image loading utilities with metadata extraction.

All image loading in the pipeline should go through ``load_image()``
to ensure consistent colour ordering (BGR→RGB), EXIF-orientation
correction, and metadata capture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from tailorvision.exceptions import ImageNotFoundError, ImageUnreadableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageMetadata:
    """Lightweight summary of a loaded image."""
    path: Path
    width: int
    height: int
    channels: int
    file_size_bytes: int
    aspect_ratio: float  # width / height

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return ``(width, height)``."""
        return (self.width, self.height)

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1_000_000, 2)


def load_image(path: str | Path, *, as_rgb: bool = True) -> np.ndarray:
    """
    Load an image from disk, applying EXIF orientation correction.

    Parameters
    ----------
    path:
        Absolute or relative path to the image file.
    as_rgb:
        If ``True`` (default), convert from BGR to RGB after loading.

    Returns
    -------
    np.ndarray
        Image array of shape ``(H, W, 3)``, dtype ``uint8``.

    Raises
    ------
    ImageNotFoundError
        If the file does not exist.
    ImageUnreadableError
        If OpenCV cannot decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise ImageNotFoundError(f"Image not found: {path}")

    # cv2.IMREAD_COLOR handles most formats; use IMREAD_UNCHANGED for transparency
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageUnreadableError(
            f"OpenCV could not decode image: {path}. "
            "Check that the file is a valid JPEG/PNG/BMP/TIFF."
        )

    # Apply EXIF rotation so portrait photos are upright
    img = _apply_exif_orientation(img, path)

    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.debug("Loaded image %s — shape=%s", path.name, img.shape)
    return img


def get_image_metadata(path: str | Path, img: Optional[np.ndarray] = None) -> ImageMetadata:
    """
    Return metadata for an image.

    Parameters
    ----------
    path:
        Path used to compute file size.
    img:
        Pre-loaded image array.  If ``None``, the image is loaded first.
    """
    path = Path(path)
    if img is None:
        img = load_image(path)

    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return ImageMetadata(
        path=path,
        width=w,
        height=h,
        channels=channels,
        file_size_bytes=path.stat().st_size,
        aspect_ratio=round(w / h, 4),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_exif_orientation(img: np.ndarray, path: Path) -> np.ndarray:
    """
    Rotate/flip image to match EXIF orientation tag.

    This corrects the common issue where phone photos are stored rotated
    but displayed correctly by software that reads EXIF data.
    """
    try:
        import PIL.Image
        import PIL.ExifTags

        pil_img = PIL.Image.open(path)
        exif = pil_img._getexif()  # type: ignore[attr-defined]
        if exif is None:
            return img

        orientation_key = next(
            (k for k, v in PIL.ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None:
            return img

        orientation = exif.get(orientation_key)
        rotation_map = {3: 180, 6: 270, 8: 90}
        if orientation in rotation_map:
            angle = rotation_map[orientation]
            img = _rotate_image(img, angle)
            logger.debug("Applied EXIF rotation %d° to %s", angle, path.name)

    except Exception:  # Pillow not installed or no EXIF data — not fatal
        pass

    return img


def _rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 90, 180, or 270 degrees."""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img
