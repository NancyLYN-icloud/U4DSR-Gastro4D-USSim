from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import CycleAnchor


@dataclass(frozen=True)
class CanonicalizationSummary:
    target_peak_phase: float
    reference_peak_phases: list[float]
    cycle_count: int


class NonlinearPhaseCanonicalizer:
    def __init__(
        self,
        default_peak_phase: float = 0.5,
        min_peak_phase: float = 0.2,
        max_peak_phase: float = 0.8,
    ) -> None:
        self.default_peak_phase = float(default_peak_phase)
        self.min_peak_phase = float(min_peak_phase)
        self.max_peak_phase = float(max_peak_phase)

    def _clip_peak_phase(self, value: float) -> float:
        return float(np.clip(value, self.min_peak_phase, self.max_peak_phase))

    def _cycle_peak_phase(self, cycle: CycleAnchor) -> float:
        return self._clip_peak_phase((cycle.peak_time - cycle.start_time) / cycle.duration)

    def estimate_target_peak_phase(self, reference_cycles: Sequence[CycleAnchor]) -> float:
        if not reference_cycles:
            return self._clip_peak_phase(self.default_peak_phase)
        peaks = np.asarray([self._cycle_peak_phase(cycle) for cycle in reference_cycles], dtype=float)
        return self._clip_peak_phase(float(np.median(peaks)))

    @staticmethod
    def _piecewise_warp(phase: float, source_peak: float, target_peak: float) -> float:
        source_peak = float(np.clip(source_peak, 1e-4, 1.0 - 1e-4))
        target_peak = float(np.clip(target_peak, 1e-4, 1.0 - 1e-4))
        phase = float(np.clip(phase, 0.0, 1.0))
        if phase <= source_peak:
            return float(target_peak * phase / source_peak)
        return float(target_peak + (1.0 - target_peak) * (phase - source_peak) / (1.0 - source_peak))

    def assign_phases(
        self,
        timestamps: Sequence[float],
        sample_cycles: Sequence[CycleAnchor],
        reference_cycles: Sequence[CycleAnchor],
    ) -> tuple[list[float], CanonicalizationSummary]:
        target_peak = self.estimate_target_peak_phase(reference_cycles)
        reference_peak_phases = [self._cycle_peak_phase(cycle) for cycle in reference_cycles]
        summary = CanonicalizationSummary(
            target_peak_phase=target_peak,
            reference_peak_phases=reference_peak_phases,
            cycle_count=len(sample_cycles),
        )
        if not sample_cycles:
            return [float("nan") for _ in timestamps], summary

        phases: list[float] = []
        cycle_index = 0
        for timestamp in timestamps:
            while cycle_index < len(sample_cycles) and timestamp > sample_cycles[cycle_index].end_time:
                cycle_index += 1
            if cycle_index >= len(sample_cycles) or timestamp < sample_cycles[cycle_index].start_time:
                phases.append(float("nan"))
                continue

            cycle = sample_cycles[cycle_index]
            source_peak = self._cycle_peak_phase(cycle)
            linear_phase = float(np.clip((timestamp - cycle.start_time) / cycle.duration, 0.0, 1.0))
            phases.append(self._piecewise_warp(linear_phase, source_peak, target_peak))
        return phases, summary