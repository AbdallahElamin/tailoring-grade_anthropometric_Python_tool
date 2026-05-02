"""tests/test_tailoring_mapper.py — TailoringMapper unit tests."""
from __future__ import annotations
import pytest
from tailorvision.tailor.tailoring_mapper import TailoringMapper
from tailorvision.tailor.ease_tables import get_ease_table


SAMPLE_MEASUREMENTS = {
    "head_circumference":         56.0,
    "neck_circumference":         38.0,
    "shoulder_to_crotch_height":  74.0,
    "chest_circumference":        96.0,
    "waist_circumference":        82.0,
    "hip_circumference":          98.0,
    "wrist_right_circumference":  16.5,
    "bicep_right_circumference":  30.0,
    "forearm_right_circumference": 27.0,
    "arm_right_length":           62.0,
    "inside_leg_height":          79.0,
    "thigh_left_circumference":   54.0,
    "calf_left_circumference":    37.0,
    "ankle_left_circumference":   23.0,
    "shoulder_breadth":           44.0,
    "height":                    175.0,
}


def test_traditional_garment_ease():
    mapper = TailoringMapper(garment_type="traditional")
    rec = mapper.map(SAMPLE_MEASUREMENTS)

    ease = get_ease_table("traditional")
    assert rec.chest_with_ease_cm == pytest.approx(
        SAMPLE_MEASUREMENTS["chest_circumference"] + ease["chest_circumference"], abs=0.1
    )
    assert rec.waist_with_ease_cm == pytest.approx(
        SAMPLE_MEASUREMENTS["waist_circumference"] + ease["waist_circumference"], abs=0.1
    )
    assert rec.hip_with_ease_cm == pytest.approx(
        SAMPLE_MEASUREMENTS["hip_circumference"] + ease["hip_circumference"], abs=0.1
    )


def test_rise_computation():
    mapper = TailoringMapper(garment_type="traditional")
    rec = mapper.map(SAMPLE_MEASUREMENTS)
    # rise = shoulder_to_crotch - inside_leg + ease on inside_leg
    expected_rise = (
        SAMPLE_MEASUREMENTS["shoulder_to_crotch_height"]
        - SAMPLE_MEASUREMENTS["inside_leg_height"]
    )
    assert rec.rise_cm == pytest.approx(expected_rise, abs=0.5)


def test_collar_size():
    mapper = TailoringMapper(garment_type="traditional")
    rec = mapper.map(SAMPLE_MEASUREMENTS)
    ease = get_ease_table("traditional")
    expected = SAMPLE_MEASUREMENTS["neck_circumference"] + ease["neck_circumference"]
    assert rec.collar_size_cm == pytest.approx(expected, abs=0.1)


def test_shoulder_seam():
    mapper = TailoringMapper(garment_type="suit")
    rec = mapper.map(SAMPLE_MEASUREMENTS)
    assert rec.shoulder_seam_cm == pytest.approx(
        SAMPLE_MEASUREMENTS["shoulder_breadth"] / 2.0, abs=0.1
    )


def test_suit_less_ease_than_traditional():
    trad = TailoringMapper("traditional").map(SAMPLE_MEASUREMENTS)
    suit = TailoringMapper("suit").map(SAMPLE_MEASUREMENTS)
    assert trad.chest_with_ease_cm > suit.chest_with_ease_cm


def test_unknown_garment_falls_back_to_traditional():
    ease_trad = get_ease_table("traditional")
    ease_unk  = get_ease_table("unknown_garment_type")
    assert ease_unk == ease_trad


def test_missing_measurement_returns_none():
    mapper = TailoringMapper("trousers")
    # Trousers without a chest key should not crash; collar_size_cm should be None
    rec = mapper.map({"waist_circumference": 82.0, "hip_circumference": 98.0,
                       "inside_leg_height": 79.0})
    assert rec.collar_size_cm is None
