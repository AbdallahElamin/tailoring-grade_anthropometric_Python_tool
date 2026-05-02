"""
TailorVision
============
A professional-grade anthropometric measurement pipeline for garment tailoring.

Quick start
-----------
>>> from tailorvision import TailorVisionPipeline, PipelineConfig
>>> config = PipelineConfig(known_height_cm=175.0, gender="male")
>>> pipeline = TailorVisionPipeline(config)
>>> result = pipeline.run("front.jpg", "side.jpg")
>>> print(result.measurements_cm)

Imports are intentionally lazy so that sub-modules (schema, config, tailor,
input, vision, quality) can be used in environments where torch / smplx are
not yet installed.  TailorVisionPipeline is only imported when accessed.
"""

__version__ = "0.1.0"
__author__ = "TailorVision Team"

# Always-safe imports (no torch dependency)
from tailorvision.config import PipelineConfig  # noqa: E402
from tailorvision.schema import MeasurementResult  # noqa: E402

__all__ = [
    "TailorVisionPipeline",
    "PipelineConfig",
    "MeasurementResult",
]


def __getattr__(name: str):
    """Lazy-load heavy components only when explicitly requested."""
    if name == "TailorVisionPipeline":
        from tailorvision.pipeline import TailorVisionPipeline
        return TailorVisionPipeline
    raise AttributeError(f"module 'tailorvision' has no attribute {name!r}")
