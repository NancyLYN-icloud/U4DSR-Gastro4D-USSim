from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CycleAnchor:
    start_time: float
    peak_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(float(self.end_time - self.start_time), 1e-8)


@dataclass(frozen=True)
class ObservationSample:
    timestamp: float
    points: np.ndarray
    image_snr: float
    contour_score: float
    angle_score: float
    pose_score: float


@dataclass(frozen=True)
class WeightedPhaseSet:
    phase_center: float
    points: np.ndarray
    point_weights: np.ndarray
    sample_count: int