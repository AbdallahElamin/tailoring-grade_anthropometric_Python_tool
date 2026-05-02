"""
tailorvision.pipeline
======================
Main orchestrator — wires together all 8 stages.

Usage
-----
    from tailorvision import TailorVisionPipeline, PipelineConfig

    config = PipelineConfig(
        known_height_cm=175.0,
        gender="male",
        garment_type="traditional",
    )
    result = TailorVisionPipeline(config).run("front.jpg", "side.jpg")
    result.save_json("output/result.json")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tailorvision.config import PipelineConfig
from tailorvision.exceptions import TailorVisionError
from tailorvision.fit.body_model_adapter import BodyModelAdapter
from tailorvision.fit.pose_fit_engine import FitResult, PoseFitEngine
from tailorvision.input.preprocessor import preprocess_image_pair
from tailorvision.input.validator import QualityGate
from tailorvision.measure.measurement_engine import MeasurementEngine
from tailorvision.measure.uncertainty import UncertaintyEstimator
from tailorvision.quality.quality_reporter import QualityReporter
from tailorvision.scale.scale_recovery_engine import ScaleRecoveryEngine
from tailorvision.schema import (
    MeasurementResult, SMPLXParameters, ScaleMode,
)
from tailorvision.tailor.tailoring_mapper import TailoringMapper
from tailorvision.vision.keypoint_lifter import KeypointLifter
from tailorvision.vision.pose_estimator import MediapipePoseEstimator
from tailorvision.vision.segmentor import MediapipeSegmentor

logger = logging.getLogger(__name__)


class TailorVisionPipeline:
    """
    8-stage anthropometric measurement pipeline.

    Parameters
    ----------
    config:
        Master configuration object. See ``PipelineConfig`` for defaults.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._cfg = config or PipelineConfig()
        self._setup_logging()

        # Instantiate all engines (lazy where expensive)
        self._gate        = QualityGate(self._cfg)
        self._pose_est    = MediapipePoseEstimator(self._cfg)
        self._segmentor   = MediapipeSegmentor(self._cfg)
        self._lifter      = KeypointLifter(self._cfg)
        self._adapter     = BodyModelAdapter(self._cfg)
        self._fitter      = PoseFitEngine(self._cfg, self._adapter)
        self._scale_eng   = ScaleRecoveryEngine(self._cfg, self._adapter)
        self._meas_eng    = MeasurementEngine(self._cfg)
        self._uncertainty = UncertaintyEstimator(self._cfg, self._adapter, self._meas_eng)
        self._quality     = QualityReporter()
        self._tailor      = TailoringMapper(self._cfg.garment_type)

    def run(
        self,
        front_image_path: str | Path,
        side_image_path: str | Path,
    ) -> MeasurementResult:
        """
        Execute the full pipeline on a front+side image pair.

        Parameters
        ----------
        front_image_path:
            Path to the front-view photograph.
        side_image_path:
            Path to the side-view photograph.

        Returns
        -------
        MeasurementResult
            Complete structured output with measurements, confidence, and
            tailoring recommendations.
        """
        front_path = Path(front_image_path)
        side_path  = Path(side_image_path)
        gender     = self._cfg.gender
        cfg        = self._cfg

        logger.info("═══ TailorVision pipeline START ═══")
        logger.info("Front: %s  |  Side: %s", front_path.name, side_path.name)

        # ── Stage 1: Input validation ─────────────────────────────────────────
        logger.info("Stage 1/8 · Input validation")
        validation = self._gate.validate(front_path, side_path, raise_on_fail=False)

        # ── Stage 2: Preprocessing ────────────────────────────────────────────
        logger.info("Stage 2/8 · Preprocessing")
        images = preprocess_image_pair(front_path, side_path, cfg)

        # ── Stage 3: 2D pose & segmentation ──────────────────────────────────
        logger.info("Stage 3/8 · Pose estimation")
        front_pose = self._pose_est.estimate(images.front)
        side_pose  = self._pose_est.estimate(images.side)

        front_seg = None
        side_seg  = None
        try:
            front_seg = self._segmentor.segment(images.front)
            side_seg  = self._segmentor.segment(images.side)
        except Exception as exc:
            logger.warning("Segmentation failed (non-fatal): %s", exc)

        # ── Stage 4: Keypoint fusion & consistency ────────────────────────────
        logger.info("Stage 4/8 · Keypoint fusion")
        biview = self._lifter.lift(front_pose, side_pose)

        # ── Stage 5: SMPL-X shape fitting ─────────────────────────────────────
        logger.info("Stage 5/8 · SMPL-X shape fitting")
        fit: FitResult = self._fitter.fit(biview, gender=gender)
        tpose_verts = self._adapter.get_tpose_vertices(fit.betas, gender)

        # ── Stage 6: Scale recovery ────────────────────────────────────────────
        logger.info("Stage 6/8 · Scale recovery")
        scale = self._scale_eng.recover(fit.betas, biview, front_seg)

        # ── Stage 7: Measurement extraction ──────────────────────────────────
        logger.info("Stage 7/8 · Measurement extraction")
        measurements = self._meas_eng.measure(
            tpose_verts, scale, cfg.known_height_cm
        )

        # ── Uncertainty estimation ─────────────────────────────────────────────
        logger.info("Stage 7b · Uncertainty estimation")
        uncertainty_report = self._uncertainty.estimate(
            biview, fit.betas, scale, gender=gender
        )

        # ── Stage 8: Quality report & tailoring ───────────────────────────────
        logger.info("Stage 8/8 · Quality report & tailoring")
        quality_scores, warnings = self._quality.build(
            validation, biview, fit, scale, measurements
        )
        tailoring = self._tailor.map(measurements)

        # ── Persist debug artifacts ────────────────────────────────────────────
        debug_paths = {}
        if cfg.save_debug_artifacts:
            debug_paths = self._save_debug(
                images.front, images.side, front_pose, side_pose, tpose_verts
            )

        # ── Assemble result ────────────────────────────────────────────────────
        result = MeasurementResult(
            body_model_type=cfg.model_type,
            gender=gender,
            smplx_parameters=SMPLXParameters(
                betas=fit.betas.tolist(),
                pose_neutralized=True,
                gender=gender,
            ),
            measurements_cm=measurements,
            measurement_confidence=uncertainty_report.confidence,
            uncertainty_cm=uncertainty_report.std_cm,
            scale=scale,
            quality_scores=quality_scores,
            validation_report=validation,
            warnings=warnings,
            tailoring_recommendations=tailoring,
            debug_artifact_paths=debug_paths,
        )

        logger.info("═══ TailorVision pipeline COMPLETE ═══")
        logger.info("Measurements: %s", {k: f"{v:.1f}" for k, v in measurements.items()})

        # Auto-save JSON if output_dir configured
        json_path = cfg.output_path("result.json")
        result.save_json(json_path)
        logger.info("Result saved → %s", json_path)

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self._cfg.log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    def _save_debug(self, front, side, front_pose, side_pose, verts) -> dict:
        """Write debug overlays and mesh to output_dir. Returns path dict."""
        import cv2
        import numpy as np

        paths = {}
        cfg = self._cfg

        try:
            # Pose overlays
            for img, pose, name in [
                (front, front_pose, "front_pose"),
                (side,  side_pose,  "side_pose"),
            ]:
                overlay = img.copy()
                kp = pose.keypoints_px.astype(int)
                vis = pose.visibility
                for i, (x, y) in enumerate(kp):
                    if vis[i] > 0.4:
                        cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
                out = cfg.output_path(f"{name}.jpg")
                cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                paths[name] = str(out)

            # Mesh OBJ (vertices only — no faces; use Blender/MeshLab to view)
            obj_path = cfg.output_path("smplx_tpose.obj")
            with open(obj_path, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            paths["smplx_mesh_obj"] = str(obj_path)

        except Exception as exc:
            logger.warning("Debug artifact save failed (non-fatal): %s", exc)

        return paths
