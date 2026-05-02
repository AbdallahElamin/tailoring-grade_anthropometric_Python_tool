"""
tailorvision.exceptions
=======================
Typed exception hierarchy for the TailorVision pipeline.
Every stage raises a specific subclass so callers can catch
exactly what they need without inspecting error messages.
"""


class TailorVisionError(Exception):
    """Base class for all TailorVision exceptions."""


# ── Input / Validation ──────────────────────────────────────────────────────

class InputError(TailorVisionError):
    """Raised when an input image cannot be loaded or is fundamentally unusable."""


class ImageNotFoundError(InputError):
    """Raised when the image file does not exist."""


class ImageUnreadableError(InputError):
    """Raised when the image file exists but cannot be decoded."""


class ImageQualityError(InputError):
    """Raised when the image fails quality validation (resolution, blur, crop)."""


# ── Vision / Pose Estimation ────────────────────────────────────────────────

class VisionError(TailorVisionError):
    """Raised when a computer-vision stage fails."""


class PoseEstimationError(VisionError):
    """Raised when no person or insufficient keypoints are detected."""


class SegmentationError(VisionError):
    """Raised when body segmentation fails."""


class BiViewConsistencyError(VisionError):
    """Raised when front and side poses are incompatible (different people, etc.)."""


# ── Body Model Fitting ──────────────────────────────────────────────────────

class FitError(TailorVisionError):
    """Raised when SMPL-X fitting fails or diverges."""


class BodyModelNotFoundError(FitError):
    """Raised when the SMPL-X model files are missing from the expected path."""


class FitConvergenceError(FitError):
    """Raised when the optimizer fails to converge within the iteration budget."""


# ── Scale Recovery ──────────────────────────────────────────────────────────

class ScaleError(TailorVisionError):
    """Raised when metric scale cannot be reliably determined."""


class ScaleAnchorMissingError(ScaleError):
    """Raised when known_height_cm is required but was not provided."""


# ── Measurement Extraction ──────────────────────────────────────────────────

class MeasurementError(TailorVisionError):
    """Raised when anthropometric measurement extraction fails."""


class MeasurementLibraryError(MeasurementError):
    """Raised when SMPL-Anthropometry is not importable or misconfigured."""


# ── Configuration ───────────────────────────────────────────────────────────

class ConfigurationError(TailorVisionError):
    """Raised when the pipeline configuration is invalid."""
