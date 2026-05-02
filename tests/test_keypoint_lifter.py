"""tests/test_keypoint_lifter.py — BiView fusion and consistency tests."""
from __future__ import annotations
import numpy as np
import pytest
from tailorvision.config import PipelineConfig
from tailorvision.vision.pose_estimator import StubPoseEstimator
from tailorvision.vision.keypoint_lifter import KeypointLifter, BiViewPose


def _make_pose(h=640, w=480):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return StubPoseEstimator().estimate(img)


def test_lift_returns_biview():
    cfg = PipelineConfig()
    front = _make_pose(h=640, w=480)
    side  = _make_pose(h=640, w=480)
    biview = KeypointLifter(cfg).lift(front, side)
    assert isinstance(biview, BiViewPose)
    assert biview.front_kp_px.shape == (33, 2)
    assert biview.side_kp_px.shape  == (33, 2)


def test_pixel_height_positive():
    cfg = PipelineConfig()
    front = _make_pose(h=1280, w=720)
    side  = _make_pose(h=1280, w=720)
    biview = KeypointLifter(cfg).lift(front, side)
    assert biview.pixel_height_front > 0
    assert biview.pixel_height_side  > 0


def test_overall_quality_in_range():
    cfg = PipelineConfig()
    front = _make_pose()
    side  = _make_pose()
    biview = KeypointLifter(cfg).lift(front, side)
    assert 0.0 <= biview.overall_quality <= 1.0


def test_consistency_score_in_range():
    cfg = PipelineConfig()
    front = _make_pose()
    side  = _make_pose()
    biview = KeypointLifter(cfg).lift(front, side)
    assert 0.0 <= biview.consistency_score <= 1.0


def test_same_image_high_consistency():
    """Using identical poses should yield near-perfect consistency."""
    cfg = PipelineConfig()
    img = np.zeros((640, 480, 3), dtype=np.uint8)
    pose = StubPoseEstimator().estimate(img)
    biview = KeypointLifter(cfg).lift(pose, pose)
    assert biview.consistency_score > 0.7
