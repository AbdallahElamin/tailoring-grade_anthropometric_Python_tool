"""
tests/conftest.py
=================
Shared pytest fixtures available to all test modules.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


# ── Image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_front_image() -> np.ndarray:
    """640×480 random RGB image simulating a front-view photo."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (640, 480, 3), dtype=np.uint8)


@pytest.fixture()
def synthetic_side_image() -> np.ndarray:
    """640×480 random RGB image simulating a side-view photo."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 255, (640, 480, 3), dtype=np.uint8)


@pytest.fixture()
def front_image_path(tmp_path, synthetic_front_image) -> Path:
    """Write synthetic front image to a temp file and return its path."""
    import cv2
    p = tmp_path / "front.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(synthetic_front_image, cv2.COLOR_RGB2BGR))
    return p


@pytest.fixture()
def side_image_path(tmp_path, synthetic_side_image) -> Path:
    """Write synthetic side image to a temp file and return its path."""
    import cv2
    p = tmp_path / "side.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(synthetic_side_image, cv2.COLOR_RGB2BGR))
    return p


# ── Config fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def default_config(tmp_path):
    """PipelineConfig with output_dir set to tmp_path."""
    from tailorvision.config import PipelineConfig
    return PipelineConfig(
        output_dir=tmp_path / "output",
        save_debug_artifacts=False,
        uncertainty_n_samples=1,   # speed up tests
        fit_iterations=10,          # speed up tests
        known_height_cm=175.0,
    )


# ── Measurement fixtures ──────────────────────────────────────────────────────

@pytest.fixture()
def sample_measurements() -> dict:
    """A realistic set of body measurements in cm for testing."""
    return {
        "head_circumference":           56.0,
        "neck_circumference":           38.0,
        "shoulder_to_crotch_height":    74.0,
        "chest_circumference":          96.0,
        "waist_circumference":          82.0,
        "hip_circumference":            98.0,
        "wrist_right_circumference":    16.5,
        "bicep_right_circumference":    30.0,
        "forearm_right_circumference":  27.0,
        "arm_right_length":             62.0,
        "inside_leg_height":            79.0,
        "thigh_left_circumference":     54.0,
        "calf_left_circumference":      37.0,
        "ankle_left_circumference":     23.0,
        "shoulder_breadth":             44.0,
        "height":                      175.0,
    }


# ── Scale fixture ─────────────────────────────────────────────────────────────

@pytest.fixture()
def known_height_scale() -> "tailorvision.schema.ScaleResult":
    from tailorvision.schema import ScaleResult, ScaleMode
    return ScaleResult(
        mode=ScaleMode.KNOWN_HEIGHT,
        scale_factor=1.042,
        confidence=0.97,
        model_height_cm=168.0,
        anchor_height_cm=175.0,
    )
