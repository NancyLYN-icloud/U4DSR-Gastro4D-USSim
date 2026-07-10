from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import WeightedPhaseSet


@dataclass(frozen=True)
class PhaseSupportSummary:
    phase_center: float
    support_score: float
    mean_confidence: float
    slice_ratio: float
    coverage_ratio: float
    extent_ratio: float


@dataclass(frozen=True)
class ReferenceTopologySelection:
    reference_phase: float
    support_score: float
    phase_summaries: list[PhaseSupportSummary]


def phase_support_score(
    mean_confidence: float,
    slice_ratio: float,
    coverage_ratio: float,
    extent_ratio: float,
    *,
    lambda_omega: float = 0.35,
    lambda_c: float = 0.20,
    lambda_rho: float = 0.15,
    lambda_gamma: float = 0.20,
    lambda_epsilon: float = 0.10,
    kappa: float = 6.0,
) -> float:
    """Observation support score s(phi) used to select the shared topology anchor."""
    omega = 0.7 * mean_confidence + 0.3 * slice_ratio
    coverage_term = min(kappa * coverage_ratio, 1.0)
    return float(
        lambda_omega * omega
        + lambda_c * mean_confidence
        + lambda_rho * slice_ratio
        + lambda_gamma * coverage_term
        + lambda_epsilon * extent_ratio
    )


def select_reference_phase(
    phase_sets: Sequence[WeightedPhaseSet],
    *,
    slice_ratio: float = 1.0,
    coverage_ratio: float = 1.0,
    extent_ratio: float = 1.0,
) -> ReferenceTopologySelection:
    """Pick phi_0 with maximum observational support for V_0 initialization."""
    if not phase_sets:
        raise ValueError("At least one weighted phase set is required")

    summaries: list[PhaseSupportSummary] = []
    for phase_set in phase_sets:
        mean_confidence = float(np.mean(phase_set.point_weights)) if len(phase_set.point_weights) else 0.0
        support = phase_support_score(
            mean_confidence=mean_confidence,
            slice_ratio=slice_ratio,
            coverage_ratio=coverage_ratio,
            extent_ratio=extent_ratio,
        )
        summaries.append(
            PhaseSupportSummary(
                phase_center=float(phase_set.phase_center),
                support_score=support,
                mean_confidence=mean_confidence,
                slice_ratio=slice_ratio,
                coverage_ratio=coverage_ratio,
                extent_ratio=extent_ratio,
            )
        )

    best = max(summaries, key=lambda item: item.support_score)
    return ReferenceTopologySelection(
        reference_phase=best.phase_center,
        support_score=best.support_score,
        phase_summaries=summaries,
    )
