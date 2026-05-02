"""
tailorvision.measure.measurement_engine
=========================================
Extracts anthropometric measurements from SMPL-X T-pose vertices using the
SMPL-Anthropometry library (DavidBoja/SMPL-Anthropometry).

Integration strategy
--------------------
SMPL-Anthropometry is NOT on PyPI.  It must be cloned into:
    third_party/SMPL-Anthropometry/

This module adds that path to sys.path at import time so ``measure``,
``measurement_definitions``, and ``landmark_definitions`` are importable.

Setup command (run once):
    git clone https://github.com/DavidBoja/SMPL-Anthropometry third_party/SMPL-Anthropometry

Standard measurements (SMPL-Anthropometry labels A–P):
    A head circumference        B neck circumference
    C shoulder-to-crotch        D chest circumference
    E waist circumference       F hip circumference
    G wrist right circ.         H bicep right circ.
    I forearm right circ.       J arm right length
    K inside leg height         L thigh left circ.
    M calf left circ.           N ankle left circ.
    O shoulder breadth          P height
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import MeasurementError, MeasurementLibraryError
from tailorvision.schema import ScaleResult

logger = logging.getLogger(__name__)

# Human-readable name map (label → snake_case key for output dict)
_LABEL_TO_KEY: Dict[str, str] = {
    "A": "head_circumference",
    "B": "neck_circumference",
    "C": "shoulder_to_crotch_height",
    "D": "chest_circumference",
    "E": "waist_circumference",
    "F": "hip_circumference",
    "G": "wrist_right_circumference",
    "H": "bicep_right_circumference",
    "I": "forearm_right_circumference",
    "J": "arm_right_length",
    "K": "inside_leg_height",
    "L": "thigh_left_circumference",
    "M": "calf_left_circumference",
    "N": "ankle_left_circumference",
    "O": "shoulder_breadth",
    "P": "height",
}


def _ensure_smpl_anthropometry(smpl_anthropometry_dir: Path) -> None:
    """Add SMPL-Anthropometry to sys.path so it is importable."""
    path_str = str(smpl_anthropometry_dir.resolve())
    if path_str not in sys.path:
        if not smpl_anthropometry_dir.exists():
            raise MeasurementLibraryError(
                f"SMPL-Anthropometry not found at: {smpl_anthropometry_dir}\n"
                "Clone it with:\n"
                "  git clone https://github.com/DavidBoja/SMPL-Anthropometry "
                "third_party/SMPL-Anthropometry"
            )
        sys.path.insert(0, path_str)
        logger.debug("Added SMPL-Anthropometry to sys.path: %s", path_str)


class MeasurementEngine:
    """
    Wraps SMPL-Anthropometry's ``MeasureBody`` to extract all standard
    measurements from T-pose SMPL-X vertices.

    Parameters
    ----------
    config:
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        _ensure_smpl_anthropometry(config.smpl_anthropometry_dir)

    def measure(
        self,
        tpose_vertices: np.ndarray,      # (10475, 3) in metres
        scale_result: ScaleResult,
        known_height_cm: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Extract all 16 standard anthropometric measurements in centimetres.

        Parameters
        ----------
        tpose_vertices:
            SMPL-X T-pose mesh vertices in metres (SMPL-X native units).
        scale_result:
            Scale recovery result providing the scale factor.
        known_height_cm:
            If provided and scale mode is NORMALIZED, used for
            height-normalisation via SMPL-Anthropometry's built-in method.

        Returns
        -------
        Dict[str, float]
            Mapping of measurement name → value in centimetres.
        """
        try:
            from measure import MeasureBody                         # type: ignore
            from measurement_definitions import STANDARD_LABELS     # type: ignore
        except ImportError as exc:
            raise MeasurementLibraryError(
                "Cannot import from SMPL-Anthropometry. "
                "Ensure the repository is cloned into third_party/SMPL-Anthropometry."
            ) from exc

        model_type = self._cfg.model_type  # "smplx" or "smpl"

        try:
            measurer = MeasureBody(model_type)
            # from_verts accepts metres; SMPL-Anthropometry converts internally
            import torch
            verts_tensor = torch.from_numpy(tpose_vertices).float()
            measurer.from_verts(verts=verts_tensor)
            all_names = measurer.all_possible_measurements
            measurer.measure(all_names)
            measurer.label_measurements(STANDARD_LABELS)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise MeasurementError(
                f"SMPL-Anthropometry measurement extraction failed: {exc}"
            ) from exc

        raw: Dict[str, float] = measurer.measurements  # already in cm

        # Apply scale factor (converts model cm → real-world cm)
        sf = scale_result.scale_factor
        scaled: Dict[str, float] = {k: round(v * sf, 2) for k, v in raw.items()}

        # Height-normalise if in NORMALIZED mode and height known
        from tailorvision.schema import ScaleMode
        if scale_result.mode == ScaleMode.NORMALIZED and known_height_cm:
            try:
                measurer.height_normalize_measurements(known_height_cm)
                normed: Dict[str, float] = measurer.height_normalized_measurements
                scaled.update({k: round(v, 2) for k, v in normed.items()})
                logger.info("Measurements height-normalised to %.1f cm.", known_height_cm)
            except Exception as exc:
                logger.warning("Height normalisation failed: %s", exc)

        # Re-map to human-readable snake_case keys
        result: Dict[str, float] = {}
        labeled: Dict[str, str] = measurer.labels2names  # {label: name}
        for label, name in labeled.items():
            key = _LABEL_TO_KEY.get(label, name.replace(" ", "_").lower())
            if name in scaled:
                result[key] = scaled[name]

        logger.info("Measurements extracted: %d values", len(result))
        return result
