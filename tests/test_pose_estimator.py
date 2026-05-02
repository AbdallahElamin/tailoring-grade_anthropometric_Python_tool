"""tests/test_pose_estimator.py — Stub pose estimator smoke tests."""
from __future__ import annotations
import numpy as np
import pytest
from tailorvision.vision.pose_estimator import StubPoseEstimator, PoseResult


def _synthetic_image(h=640, w=480):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_stub_returns_pose_result():
    est = StubPoseEstimator()
    img = _synthetic_image()
    result = est.estimate(img)
    assert isinstance(result, PoseResult)
    assert result.keypoints.shape == (33, 2)
    assert result.visibility.shape == (33,)
    assert result.keypoints_px.shape == (33, 2)
    assert 0.0 <= result.quality_score <= 1.0


def test_stub_keypoints_in_image_bounds():
    est = StubPoseEstimator()
    img = _synthetic_image(h=1000, w=800)
    result = est.estimate(img)
    kp_px = result.keypoints_px
    assert kp_px[:, 0].min() >= 0
    assert kp_px[:, 0].max() <= 800
    assert kp_px[:, 1].min() >= 0
    assert kp_px[:, 1].max() <= 1000


def test_stub_normalised_keypoints_in_unit_range():
    est = StubPoseEstimator()
    img = _synthetic_image()
    result = est.estimate(img)
    assert result.keypoints.min() >= 0.0
    assert result.keypoints.max() <= 1.0
