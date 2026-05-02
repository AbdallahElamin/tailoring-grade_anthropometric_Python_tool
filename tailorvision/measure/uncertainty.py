"""
tailorvision.measure.uncertainty
=================================
Monte-Carlo uncertainty estimation via repeated fitting with perturbed keypoints.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from tailorvision.schema import ConfidenceLevel, ScaleResult
from tailorvision.vision.keypoint_lifter import BiViewPose

if TYPE_CHECKING:
    from tailorvision.config import PipelineConfig
    from tailorvision.fit.body_model_adapter import BodyModelAdapter
    from tailorvision.fit.pose_fit_engine import PoseFitEngine
    from tailorvision.measure.measurement_engine import MeasurementEngine

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyReport:
    """Per-measurement uncertainty estimates."""
    std_cm: Dict[str, float]
    confidence: Dict[str, ConfidenceLevel]


def _confidence_from_std(std: float) -> ConfidenceLevel:
    """Map std-dev (cm) to a qualitative confidence label."""
    if std < 0.8:
        return ConfidenceLevel.HIGH
    if std < 2.0:
        return ConfidenceLevel.MEDIUM
    if std < 4.0:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.UNRELIABLE


class UncertaintyEstimator:
    """
    Monte-Carlo uncertainty estimation.

    Parameters
    ----------
    config:
        Pipeline configuration.
    adapter:
        Shared BodyModelAdapter.
    measurement_engine:
        Shared MeasurementEngine.
    """

    def __init__(self, config, adapter, measurement_engine) -> None:
        self._cfg = config
        self._adapter = adapter
        self._engine = measurement_engine
        self._rng = np.random.default_rng(config.random_seed + 99)

    def estimate(
        self,
        biview: BiViewPose,
        nominal_betas: np.ndarray,
        scale_result: ScaleResult,
        gender: Optional[str] = None,
    ) -> UncertaintyReport:
        """
        Run N perturbed fits and collect per-measurement distributions.

        Returns
        -------
        UncertaintyReport
        """
        # Lazy import to avoid torch at collection time
        from tailorvision.fit.pose_fit_engine import PoseFitEngine

        n = self._cfg.uncertainty_n_samples
        sigma = self._cfg.uncertainty_keypoint_noise_px
        gender = (gender or self._cfg.gender).lower()
        fitter = PoseFitEngine(self._cfg, self._adapter)

        all_measurements: List[Dict[str, float]] = []

        for i in range(n):
            noisy = self._perturb_biview(biview, sigma)
            try:
                fit = fitter.fit(noisy, gender=gender, initial_betas=nominal_betas)
                verts = self._adapter.get_tpose_vertices(fit.betas, gender)
                meas = self._engine.measure(verts, scale_result)
                all_measurements.append(meas)
            except Exception as exc:
                logger.debug("Uncertainty sample %d failed: %s", i, exc)

        if not all_measurements:
            logger.warning("All uncertainty samples failed; returning zero uncertainty.")
            return UncertaintyReport(std_cm={}, confidence={})

        keys = list(all_measurements[0].keys())
        std_cm: Dict[str, float] = {}
        confidence: Dict[str, ConfidenceLevel] = {}

        for k in keys:
            vals = [m[k] for m in all_measurements if k in m]
            s = float(np.std(vals)) if len(vals) >= 2 else 0.0
            std_cm[k] = round(s, 2)
            confidence[k] = _confidence_from_std(s)

        return UncertaintyReport(std_cm=std_cm, confidence=confidence)

    def _perturb_biview(self, biview: BiViewPose, sigma: float) -> BiViewPose:
        """Return a copy of BiViewPose with Gaussian noise on keypoints."""
        fw, fh = biview.front_img_wh
        sw, sh = biview.side_img_wh

        noise_f = self._rng.normal(0, sigma, biview.front_kp_px.shape).astype(np.float32)
        noise_s = self._rng.normal(0, sigma, biview.side_kp_px.shape).astype(np.float32)

        f_kp = np.clip(biview.front_kp_px + noise_f, [0, 0], [fw, fh])
        s_kp = np.clip(biview.side_kp_px  + noise_s, [0, 0], [sw, sh])

        return BiViewPose(
            front_kp_px=f_kp, side_kp_px=s_kp,
            front_vis=biview.front_vis, side_vis=biview.side_vis,
            front_img_wh=biview.front_img_wh, side_img_wh=biview.side_img_wh,
            pixel_height_front=biview.pixel_height_front,
            pixel_height_side=biview.pixel_height_side,
            joint_map=biview.joint_map,
            overall_quality=biview.overall_quality,
            consistency_score=biview.consistency_score,
        )
