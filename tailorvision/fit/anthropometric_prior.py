"""
tailorvision.fit.anthropometric_prior
Differentiable anthropometric prior loss for shape fitting regularisation.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import torch

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _Prior:
    name: str
    mean: float
    std: float
    k: float = 2.5

_PRIORS = (
    _Prior("waist_to_chest",    0.800, 0.055),
    _Prior("hip_to_chest",      0.990, 0.060),
    _Prior("shoulder_to_chest", 0.455, 0.030),
)

class AnthropometricPrior:
    """Hinge-loss penalty when body proportions deviate from population norms."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)

    def compute_loss(self, betas: torch.Tensor) -> torch.Tensor:
        try:
            chest    = 0.945 + (betas[0] * torch.tensor(
                [0.045,0.032,0.015,0.008,0.004,0.002,0.001,0.001,0.,0.],
                device=betas.device)).sum()
            waist    = 0.760 + (betas[0] * torch.tensor(
                [0.040,0.028,0.010,0.012,0.005,0.002,0.001,0.001,0.,0.],
                device=betas.device)).sum()
            hip      = 0.930 + (betas[0] * torch.tensor(
                [0.038,0.020,0.008,0.018,0.006,0.003,0.001,0.001,0.,0.],
                device=betas.device)).sum()
            shoulder = 0.418 + (betas[0] * torch.tensor(
                [0.022,0.018,0.006,0.003,0.002,0.001,0.,0.,0.,0.],
                device=betas.device)).sum()
            eps = 1e-6
            ratios = {
                "waist_to_chest":    waist    / (chest + eps),
                "hip_to_chest":      hip      / (chest + eps),
                "shoulder_to_chest": shoulder / (chest + eps),
            }
            loss = torch.zeros(1, device=self._device)
            for p in _PRIORS:
                r = ratios[p.name]
                dev = (r - p.mean) / p.std
                loss = loss + torch.clamp(dev.abs() - p.k, min=0.0) ** 2
            return loss.squeeze()
        except Exception as e:
            logger.debug("AnthropometricPrior skipped: %s", e)
            return torch.zeros(1, device=self._device).squeeze()
