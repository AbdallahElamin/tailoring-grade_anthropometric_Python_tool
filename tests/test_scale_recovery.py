"""tests/test_scale_recovery.py — Scale recovery engine unit tests."""
from __future__ import annotations
import numpy as np
import pytest

from tailorvision.config import PipelineConfig
from tailorvision.schema import ScaleMode


# ── Helpers ───────────────────────────────────────────────────────────────────

class _MockAdapter:
    """Minimal adapter stub that returns a fixed model height."""
    def model_height_m(self, betas, gender=None):
        return 1.68  # canonical SMPL-X neutral height

class _MockBiView:
    pixel_height_front = 900.0
    pixel_height_side  = 880.0
    front_img_wh = (640, 1280)
    side_img_wh  = (640, 1280)


def _make_engine(known_height_cm=None):
    from tailorvision.scale.scale_recovery_engine import ScaleRecoveryEngine
    cfg = PipelineConfig(known_height_cm=known_height_cm)
    adapter = _MockAdapter()
    return ScaleRecoveryEngine(cfg, adapter)


def test_known_height_mode():
    engine = _make_engine(known_height_cm=175.0)
    betas = np.zeros(10, dtype=np.float32)
    scale = engine.recover(betas, _MockBiView())
    assert scale.mode == ScaleMode.KNOWN_HEIGHT
    assert scale.confidence >= 0.90
    expected_sf = 175.0 / 168.0   # 175 / (1.68 * 100)
    assert abs(scale.scale_factor - expected_sf) < 0.01


def test_heuristic_mode():
    engine = _make_engine(known_height_cm=None)
    betas = np.zeros(10, dtype=np.float32)
    scale = engine.recover(betas, _MockBiView())
    assert scale.mode in (ScaleMode.HEURISTIC, ScaleMode.NORMALIZED)
    assert 0.0 < scale.scale_factor < 5.0


def test_normalised_mode_fallback():
    from tailorvision.scale.scale_recovery_engine import ScaleRecoveryEngine
    cfg = PipelineConfig(known_height_cm=None)
    adapter = _MockAdapter()
    engine = ScaleRecoveryEngine(cfg, adapter)

    class _NoBiView:
        pixel_height_front = 0.0
        pixel_height_side  = 0.0
        front_img_wh = (640, 1280)
        side_img_wh  = (640, 1280)

    scale = engine.recover(np.zeros(10), _NoBiView())
    assert scale.mode == ScaleMode.NORMALIZED
    assert scale.confidence < 0.50
