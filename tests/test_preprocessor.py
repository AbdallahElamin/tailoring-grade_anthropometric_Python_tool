"""
tests/test_preprocessor.py — Preprocessing pipeline unit tests.
"""
from __future__ import annotations
import numpy as np
import pytest
import cv2
from pathlib import Path

from tailorvision.config import PipelineConfig
from tailorvision.input.preprocessor import (
    preprocess_image_pair,
    _resize_to_long_side,
    _ensure_uint8_rgb,
)


def _write_image(path: Path, h=800, w=600) -> None:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_resize_to_long_side_portrait():
    img = np.zeros((1200, 600, 3), dtype=np.uint8)
    out = _resize_to_long_side(img, 640)
    assert max(out.shape[:2]) == 640


def test_resize_to_long_side_landscape():
    img = np.zeros((400, 1600, 3), dtype=np.uint8)
    out = _resize_to_long_side(img, 640)
    assert max(out.shape[:2]) == 640


def test_resize_preserves_aspect_ratio():
    img = np.zeros((1000, 500, 3), dtype=np.uint8)
    out = _resize_to_long_side(img, 640)
    orig_ratio = 1000 / 500
    new_ratio  = out.shape[0] / out.shape[1]
    assert abs(orig_ratio - new_ratio) < 0.05


def test_ensure_uint8_rgb_from_grayscale():
    gray = np.zeros((100, 100), dtype=np.uint8)
    rgb = _ensure_uint8_rgb(gray)
    assert rgb.shape == (100, 100, 3)
    assert rgb.dtype == np.uint8


def test_ensure_uint8_rgb_from_rgba():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, 3] = 200  # mostly opaque
    rgb = _ensure_uint8_rgb(rgba)
    assert rgb.shape == (100, 100, 3)


def test_preprocess_image_pair_returns_correct_shapes(tmp_path):
    front = tmp_path / "front.jpg"
    side  = tmp_path / "side.jpg"
    _write_image(front, h=1200, w=900)
    _write_image(side,  h=1100, w=850)

    cfg = PipelineConfig(remove_background=False)
    pair = preprocess_image_pair(front, side, cfg)

    assert pair.front.ndim == 3
    assert pair.side.ndim  == 3
    assert pair.front.dtype == np.uint8
    assert pair.side.dtype  == np.uint8
    assert max(pair.front.shape[:2]) == 640
    assert max(pair.side.shape[:2])  == 640


def test_preprocess_preserves_originals(tmp_path):
    front = tmp_path / "front.jpg"
    side  = tmp_path / "side.jpg"
    _write_image(front, h=800, w=600)
    _write_image(side,  h=800, w=600)

    cfg = PipelineConfig()
    pair = preprocess_image_pair(front, side, cfg)

    # Originals should be full resolution (not resized)
    assert pair.front_original.shape[0] == 800
    assert pair.side_original.shape[0]  == 800
