"""tests/test_schema.py — Pydantic schema serialisation round-trip tests."""
from __future__ import annotations
import json
import pytest
from pathlib import Path

from tailorvision.schema import (
    MeasurementResult, SMPLXParameters, ScaleResult, ScaleMode,
    QualityScores, TailoringRecommendations, ValidationReport,
    ConfidenceLevel, WarningCode,
)


def _make_result() -> MeasurementResult:
    return MeasurementResult(
        body_model_type="smplx",
        gender="neutral",
        smplx_parameters=SMPLXParameters(
            betas=[0.0] * 10,
            pose_neutralized=True,
            gender="neutral",
        ),
        measurements_cm={
            "height": 175.0,
            "chest_circumference": 96.0,
            "waist_circumference": 82.0,
            "hip_circumference": 98.0,
        },
        measurement_confidence={
            "height": ConfidenceLevel.HIGH,
            "chest_circumference": ConfidenceLevel.MEDIUM,
        },
        uncertainty_cm={"chest_circumference": 1.5},
        scale=ScaleResult(
            mode=ScaleMode.KNOWN_HEIGHT,
            scale_factor=1.042,
            confidence=0.97,
            model_height_cm=168.0,
            anchor_height_cm=175.0,
        ),
        quality_scores=QualityScores(
            image_quality_front=0.85,
            image_quality_side=0.80,
            pose_quality_front=0.78,
            pose_quality_side=0.75,
            fit_quality=0.72,
            scale_quality=0.97,
            overall=0.80,
        ),
        validation_report=ValidationReport(
            passed=True,
            front_image_ok=True,
            side_image_ok=True,
        ),
        warnings=[WarningCode.SCALE_UNCERTAIN],
        tailoring_recommendations=TailoringRecommendations(
            garment_type="traditional",
            collar_size_cm=40.5,
            chest_with_ease_cm=108.0,
        ),
    )


def test_serialise_to_json():
    result = _make_result()
    json_str = result.to_json()
    data = json.loads(json_str)
    assert data["body_model_type"] == "smplx"
    assert data["measurements_cm"]["height"] == 175.0


def test_round_trip():
    result = _make_result()
    json_str = result.to_json()
    restored = MeasurementResult.model_validate_json(json_str)
    assert restored.measurements_cm["chest_circumference"] == pytest.approx(96.0)


def test_save_and_load_json(tmp_path):
    result = _make_result()
    out = tmp_path / "result.json"
    result.save_json(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "measurements_cm" in data


def test_warning_codes_serialised():
    result = _make_result()
    data = json.loads(result.to_json())
    assert "SCALE_UNCERTAIN" in data["warnings"]
