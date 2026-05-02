"""
tailorvision.tailor.ease_tables
=================================
Ease allowance tables by garment type.

Ease = the additional centimetres added to a body measurement to allow
freedom of movement and drape.  Values are based on standard tailoring
references for each garment category.

Extending: add a new key to _EASE_TABLES to define a new garment type.
"""
from __future__ import annotations
from typing import Dict

# Keys match the snake_case measurement names from MeasurementEngine
EaseTable = Dict[str, float]

_EASE_TABLES: Dict[str, EaseTable] = {
    "traditional": {
        # Traditional African wear (agbada, dashiki, kaftan, boubou)
        # Typically loose-fitting with generous ease.
        "chest_circumference":   12.0,
        "waist_circumference":   14.0,
        "hip_circumference":     16.0,
        "shoulder_breadth":       1.0,
        "neck_circumference":     2.5,
        "arm_right_length":       2.0,
        "inside_leg_height":      3.0,
        "thigh_left_circumference": 8.0,
    },
    "suit": {
        # Tailored suit / formal jacket
        "chest_circumference":    5.0,
        "waist_circumference":    3.0,
        "hip_circumference":      6.0,
        "shoulder_breadth":       0.5,
        "neck_circumference":     2.0,
        "arm_right_length":       1.5,
        "inside_leg_height":      1.5,
        "thigh_left_circumference": 4.0,
    },
    "shirt": {
        "chest_circumference":    8.0,
        "waist_circumference":    8.0,
        "hip_circumference":      8.0,
        "shoulder_breadth":       0.5,
        "neck_circumference":     2.0,
        "arm_right_length":       2.0,
    },
    "trousers": {
        "waist_circumference":    2.5,
        "hip_circumference":      4.0,
        "thigh_left_circumference": 5.0,
        "inside_leg_height":      1.5,
    },
}


def get_ease_table(garment_type: str) -> EaseTable:
    """
    Return the ease allowance table for ``garment_type``.

    Falls back to ``"traditional"`` if the type is not recognised.
    """
    if garment_type not in _EASE_TABLES:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown garment type '%s'; using 'traditional' ease table.", garment_type
        )
        garment_type = "traditional"
    return _EASE_TABLES[garment_type]
