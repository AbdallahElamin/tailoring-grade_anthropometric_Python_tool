"""tests/test_quality_reporter.py — QualityReporter unit tests (torch-free)."""
from __future__ import annotations
import dataclasses
import pytest
import numpy as np

from tailorvision.quality.quality_reporter import QualityReporter
from tailorvision.schema import WarningCode, ValidationReport, ScaleResult, ScaleMode


# ── Stub FitResult dataclass (avoids importing torch) ─────────────────────────
@dataclasses.dataclass
class _StubFitResult:
    betas: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(10))
    body_pose: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(63))
    global_orient: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    final_loss: float = 0.02
    converged: bool = True
    iterations_run: int = 300
    fit_quality: float = 0.80


def _make_biview(quality=0.85, consistency=0.90):
    from tailorvision.vision.keypoint_lifter import BiViewPose
    kp = np.zeros((33, 2), dtype=np.float32)
    return BiViewPose(
        front_kp_px=kp, side_kp_px=kp,
        front_vis=np.ones(33, dtype=np.float32),
        side_vis=np.ones(33, dtype=np.float32),
        front_img_wh=(640, 480), side_img_wh=(640, 480),
        pixel_height_front=400.0, pixel_height_side=395.0,
        joint_map={}, overall_quality=quality,
        consistency_score=consistency,
    )


def _make_fit(converged=True, quality=0.80):
    return _StubFitResult(converged=converged, fit_quality=quality,
                          final_loss=0.02 if converged else 0.30)


def _make_scale(confidence=0.97):
    return ScaleResult(
        mode=ScaleMode.KNOWN_HEIGHT, scale_factor=1.042, confidence=confidence,
        model_height_cm=168.0, anchor_height_cm=175.0,
    )


def _make_validation(passed=True):
    return ValidationReport(
        passed=passed, front_image_ok=True, side_image_ok=True,
        front_blur_score=500.0, side_blur_score=480.0,
    )


def test_good_inputs_high_overall():
    scores, warnings = QualityReporter().build(
        _make_validation(), _make_biview(), _make_fit(), _make_scale(),
        {"chest_circumference": 96.0, "waist_circumference": 82.0, "hip_circumference": 98.0},
    )
    assert scores.overall > 0.60
    assert WarningCode.RETAKE_RECOMMENDED not in warnings


def test_low_scale_confidence_triggers_warning():
    _, warnings = QualityReporter().build(
        _make_validation(), _make_biview(), _make_fit(), _make_scale(confidence=0.40), {},
    )
    assert WarningCode.SCALE_UNCERTAIN in warnings


def test_poor_fit_triggers_warning():
    _, warnings = QualityReporter().build(
        _make_validation(), _make_biview(),
        _make_fit(converged=False, quality=0.20), _make_scale(), {},
    )
    assert WarningCode.POOR_FIT_CONVERGENCE in warnings


def test_loose_clothing_waist_warning():
    _, warnings = QualityReporter().build(
        _make_validation(), _make_biview(), _make_fit(), _make_scale(),
        {"waist_circumference": 70.0, "hip_circumference": 100.0, "chest_circumference": 95.0},
    )
    assert WarningCode.LOOSE_CLOTHING_WAIST in warnings


def test_loose_clothing_chest_warning():
    _, warnings = QualityReporter().build(
        _make_validation(), _make_biview(), _make_fit(), _make_scale(),
        {"chest_circumference": 130.0, "waist_circumference": 80.0, "hip_circumference": 100.0},
    )
    assert WarningCode.LOOSE_CLOTHING_CHEST in warnings


def test_bad_posture_warning():
    _, warnings = QualityReporter().build(
        _make_validation(), _make_biview(quality=0.40, consistency=0.30),
        _make_fit(), _make_scale(), {},
    )
    assert WarningCode.BAD_POSTURE in warnings


def test_quality_scores_in_range():
    scores, _ = QualityReporter().build(
        _make_validation(), _make_biview(), _make_fit(), _make_scale(), {}
    )
    for field, val in scores.model_dump().items():
        assert 0.0 <= val <= 1.0, f"{field}={val} out of range"
