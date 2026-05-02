"""
tailorvision.tailor.tailoring_mapper
======================================
Converts body measurements into garment-ready tailoring outputs.

Derived measurements
--------------------
- Collar size   = neck_circumference + collar ease (usually 2–2.5 cm)
- Rise          = shoulder_to_crotch_height − inside_leg_height
- Sleeve length = arm_right_length (with ease)
- Chest / waist / hip with ease = body measurement + garment ease

All derived values are in centimetres and clearly labelled.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional

from tailorvision.schema import TailoringRecommendations
from tailorvision.tailor.ease_tables import get_ease_table

logger = logging.getLogger(__name__)


class TailoringMapper:
    """
    Maps body measurements to garment-ready tailoring values.

    Parameters
    ----------
    garment_type:
        One of ``"traditional"``, ``"suit"``, ``"shirt"``, ``"trousers"``.
    """

    def __init__(self, garment_type: str = "traditional") -> None:
        self._garment_type = garment_type
        self._ease = get_ease_table(garment_type)

    def map(self, measurements_cm: Dict[str, float]) -> TailoringRecommendations:
        """
        Produce garment-ready measurements.

        Parameters
        ----------
        measurements_cm:
            Raw body measurements from ``MeasurementEngine``.

        Returns
        -------
        TailoringRecommendations
        """
        ease = self._ease
        get = measurements_cm.get  # shorthand

        def add_ease(key: str, default: Optional[float] = None) -> Optional[float]:
            val = get(key, default)
            if val is None:
                return None
            e = ease.get(key, 0.0)
            return round(val + e, 1)

        # Derived measurements
        shoulder_to_crotch = get("shoulder_to_crotch_height")
        inside_leg         = get("inside_leg_height")
        rise = (
            round(shoulder_to_crotch - inside_leg, 1)
            if shoulder_to_crotch is not None and inside_leg is not None
            else None
        )

        neck  = get("neck_circumference")
        collar = round(neck + ease.get("neck_circumference", 2.0), 1) if neck else None

        # Shoulder seam = shoulder_breadth / 2 (distance from collar to sleeve)
        sh_breadth = get("shoulder_breadth")
        shoulder_seam = round(sh_breadth / 2.0, 1) if sh_breadth else None

        rec = TailoringRecommendations(
            garment_type=self._garment_type,
            collar_size_cm=collar,
            chest_with_ease_cm=add_ease("chest_circumference"),
            waist_with_ease_cm=add_ease("waist_circumference"),
            hip_with_ease_cm=add_ease("hip_circumference"),
            rise_cm=rise,
            sleeve_length_cm=add_ease("arm_right_length"),
            inseam_cm=add_ease("inside_leg_height"),
            shoulder_seam_cm=shoulder_seam,
            ease_table=ease,
        )

        logger.info(
            "Tailoring map complete: garment=%s, chest_ease=%.1f, waist_ease=%.1f",
            self._garment_type,
            ease.get("chest_circumference", 0),
            ease.get("waist_circumference", 0),
        )
        return rec
