from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .model import GlobalBasisResidualModel
from .types import WeightedPhaseSet


@dataclass(frozen=True)
class GBRConfig:
    basis_rank: int = 4
    hidden_dim: int = 64
    hidden_layers: int = 2
    phase_harmonics: int = 4
    learning_rate: float = 1e-2
    steps: int = 250
    temporal_weight: float = 0.05
    periodic_weight: float = 0.05
    residual_weight: float = 1e-3
    device: str = "cpu"


@dataclass
class GBRResult:
    phase_centers: list[float]
    deformed_vertices: np.ndarray
    loss_history: list[float]
    data_loss: float
    temporal_loss: float
    periodic_loss: float
    residual_loss: float


def _symmetric_weighted_chamfer(predicted: torch.Tensor, observed: torch.Tensor, obs_weights: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(predicted, observed)
    pred_to_obs = distances.min(dim=1).values.mean()
    obs_to_pred = (distances.min(dim=0).values * obs_weights).sum() / obs_weights.sum().clamp_min(1e-6)
    return pred_to_obs + obs_to_pred


class GBRTrainer:
    def __init__(self, config: GBRConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def fit(self, reference_vertices: np.ndarray, phase_sets: Sequence[WeightedPhaseSet]) -> GBRResult:
        if not phase_sets:
            raise ValueError("At least one weighted phase set is required")

        reference = torch.as_tensor(reference_vertices, dtype=torch.float32, device=self.device)
        model = GlobalBasisResidualModel(
            reference_vertices=reference,
            basis_rank=self.config.basis_rank,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            phase_harmonics=self.config.phase_harmonics,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        phase_centers = torch.as_tensor([phase_set.phase_center for phase_set in phase_sets], dtype=torch.float32, device=self.device)
        observed_points = [torch.as_tensor(phase_set.points, dtype=torch.float32, device=self.device) for phase_set in phase_sets]
        observed_weights = [torch.as_tensor(phase_set.point_weights, dtype=torch.float32, device=self.device) for phase_set in phase_sets]

        loss_history: list[float] = []
        data_loss_value = 0.0
        temporal_loss_value = 0.0
        periodic_loss_value = 0.0
        residual_loss_value = 0.0

        for _ in range(self.config.steps):
            optimizer.zero_grad()
            predictions = model(phase_centers)
            coefficients = model.coefficients(phase_centers)
            residuals = model.residual_field(model.reference_vertices, phase_centers)

            data_loss = torch.stack(
                [
                    _symmetric_weighted_chamfer(prediction, points, weights)
                    for prediction, points, weights in zip(predictions, observed_points, observed_weights)
                ]
            ).mean()

            if len(phase_sets) > 1:
                temporal_loss = ((coefficients[1:] - coefficients[:-1]) ** 2).mean()
                periodic_loss = ((coefficients[0] - coefficients[-1]) ** 2).mean()
            else:
                temporal_loss = torch.zeros((), device=self.device)
                periodic_loss = torch.zeros((), device=self.device)

            residual_loss = residuals.square().mean()
            total_loss = (
                data_loss
                + self.config.temporal_weight * temporal_loss
                + self.config.periodic_weight * periodic_loss
                + self.config.residual_weight * residual_loss
            )
            total_loss.backward()
            optimizer.step()

            loss_history.append(float(total_loss.detach().cpu().item()))
            data_loss_value = float(data_loss.detach().cpu().item())
            temporal_loss_value = float(temporal_loss.detach().cpu().item())
            periodic_loss_value = float(periodic_loss.detach().cpu().item())
            residual_loss_value = float(residual_loss.detach().cpu().item())

        final_predictions = model(phase_centers).detach().cpu().numpy()
        return GBRResult(
            phase_centers=[float(value) for value in phase_centers.detach().cpu().tolist()],
            deformed_vertices=final_predictions,
            loss_history=loss_history,
            data_loss=data_loss_value,
            temporal_loss=temporal_loss_value,
            periodic_loss=periodic_loss_value,
            residual_loss=residual_loss_value,
        )


def export_result_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")