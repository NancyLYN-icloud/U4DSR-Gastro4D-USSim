from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .types import ObservationSample, WeightedPhaseSet


def confidence_from_quality(
    image_snr: float,
    contour_score: float,
    angle_score: float,
    pose_score: float,
) -> float:
    snr_score = np.log1p(max(float(image_snr), 1e-6)) / np.log1p(10.0)
    confidence = 0.35 * snr_score + 0.25 * contour_score + 0.20 * angle_score + 0.20 * pose_score
    return float(np.clip(confidence, 0.05, 1.0))


def _circular_phase_distance(lhs: float, rhs: float) -> float:
    delta = abs(float(lhs) - float(rhs)) % 1.0
    return min(delta, 1.0 - delta)


@dataclass
class WeightedPhaseSetBuilder:
    phase_centers: Sequence[float]
    phase_radius: float = 0.12

    def build(self, samples: Sequence[ObservationSample], assigned_phases: Sequence[float]) -> list[WeightedPhaseSet]:
        weighted_sets: list[WeightedPhaseSet] = []
        for phase_center in self.phase_centers:
            collected_points: list[np.ndarray] = []
            collected_weights: list[np.ndarray] = []
            sample_count = 0
            for sample, phase in zip(samples, assigned_phases):
                if not np.isfinite(phase):
                    continue
                if _circular_phase_distance(float(phase), float(phase_center)) > float(self.phase_radius):
                    continue
                confidence = confidence_from_quality(
                    image_snr=sample.image_snr,
                    contour_score=sample.contour_score,
                    angle_score=sample.angle_score,
                    pose_score=sample.pose_score,
                )
                points = np.asarray(sample.points, dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                    continue
                collected_points.append(points)
                collected_weights.append(np.full((len(points),), confidence, dtype=np.float32))
                sample_count += 1

            if not collected_points:
                continue
            weighted_sets.append(
                WeightedPhaseSet(
                    phase_center=float(phase_center % 1.0),
                    points=np.concatenate(collected_points, axis=0),
                    point_weights=np.concatenate(collected_weights, axis=0),
                    sample_count=sample_count,
                )
            )
        weighted_sets.sort(key=lambda item: item.phase_center)
        return weighted_sets