"""tests/test_validator.py — Unit tests for the QualityGate."""
from __future__ import annotations
import numpy as np
import pytest
import cv2
import tempfile
from pathlib import Path

from tailorvision.config import PipelineConfig
from tailorvision.input.validator import QualityGate, _laplacian_variance


def _write_synthetic_image(path: Path, width=800, height=1200, blur=False) -> None:
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (51, 51), 30)
    cv2.imwrite(str(path), img)


def test_quality_gate_passes_good_images(tmp_path):
    front = tmp_path / "front.jpg"
    side  = tmp_path / "side.jpg"
    _write_synthetic_image(front)
    _write_synthetic_image(side)

    cfg = PipelineConfig(min_image_width=480, min_image_height=640, min_blur_variance=50.0)
    report = QualityGate(cfg).validate(front, side)
    assert report.front_image_ok
    assert report.side_image_ok


def test_quality_gate_fails_missing_file(tmp_path):
    front = tmp_path / "front.jpg"
    side  = tmp_path / "nonexistent.jpg"
    _write_synthetic_image(front)

    cfg = PipelineConfig()
    report = QualityGate(cfg).validate(front, side)
    assert not report.side_image_ok
    assert not report.passed


def test_quality_gate_fails_low_resolution(tmp_path):
    front = tmp_path / "front.jpg"
    side  = tmp_path / "side.jpg"
    _write_synthetic_image(front, width=100, height=100)
    _write_synthetic_image(side,  width=100, height=100)

    cfg = PipelineConfig(min_image_width=480, min_image_height=640)
    report = QualityGate(cfg).validate(front, side)
    assert not report.front_image_ok
    assert any("Resolution" in r for r in report.reasons)


def test_quality_gate_fails_blurry(tmp_path):
    front = tmp_path / "front.jpg"
    side  = tmp_path / "side.jpg"
    _write_synthetic_image(front, blur=True)
    _write_synthetic_image(side,  blur=True)

    cfg = PipelineConfig(min_blur_variance=500.0)
    report = QualityGate(cfg).validate(front, side)
    assert not report.front_image_ok
    assert any("blurry" in r.lower() for r in report.reasons)


def test_laplacian_variance_sharp():
    sharp = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    assert _laplacian_variance(sharp) > 100


def test_laplacian_variance_blurry():
    blurry = np.zeros((640, 480, 3), dtype=np.uint8)
    assert _laplacian_variance(blurry) < 1.0
