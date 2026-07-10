from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbr_4d_ums import (
    CycleAnchor,
    GBRConfig,
    GBRTrainer,
    NonlinearPhaseCanonicalizer,
    ObservationSample,
    WeightedPhaseSetBuilder,
    export_result_summary,
    select_reference_phase,
)


OUTPUT_PATH = ROOT / "demo" / "outputs" / "demo_summary.json"


def build_reference_surface(theta_steps: int = 24, phi_steps: int = 12) -> np.ndarray:
    vertices = []
    for phi in np.linspace(0.08, np.pi - 0.08, phi_steps):
        for theta in np.linspace(0.0, 2.0 * np.pi, theta_steps, endpoint=False):
            x = 1.10 * np.sin(phi) * np.cos(theta)
            y = 0.75 * np.sin(phi) * np.sin(theta)
            z = 1.35 * np.cos(phi)
            vertices.append([x, y, z])
    return np.asarray(vertices, dtype=np.float32)


def surface_at_phase(reference_vertices: np.ndarray, phase: float) -> np.ndarray:
    phase = float(phase % 1.0)
    radial = reference_vertices / np.linalg.norm(reference_vertices, axis=1, keepdims=True)
    basis_a = np.column_stack([0.12 * reference_vertices[:, 0], -0.02 * reference_vertices[:, 1], 0.04 * reference_vertices[:, 2]])
    basis_b = np.column_stack([-0.03 * reference_vertices[:, 0], 0.08 * reference_vertices[:, 1], 0.10 * reference_vertices[:, 2]])
    coeff_a = np.sin(2.0 * np.pi * phase)
    coeff_b = np.cos(2.0 * np.pi * phase)
    residual = 0.03 * np.sin(4.0 * np.pi * phase + 2.2 * reference_vertices[:, 2])[:, None] * radial
    return reference_vertices + coeff_a * basis_a + coeff_b * basis_b + residual


def make_observation_samples(reference_vertices: np.ndarray) -> tuple[list[ObservationSample], list[float], dict[float, np.ndarray]]:
    rng = np.random.default_rng(42)
    timestamps = [0.08, 0.27, 0.46, 0.71, 1.07, 1.26, 1.48, 1.73]
    nominal_phases = [0.08, 0.25, 0.44, 0.70, 0.07, 0.24, 0.48, 0.72]
    samples: list[ObservationSample] = []
    ground_truth: dict[float, np.ndarray] = {}
    for timestamp, phase in zip(timestamps, nominal_phases):
        vertices = surface_at_phase(reference_vertices, phase)
        ground_truth[float(phase)] = vertices
        indices = rng.choice(len(vertices), size=128, replace=False)
        points = vertices[indices] + rng.normal(scale=0.01, size=(len(indices), 3))
        samples.append(
            ObservationSample(
                timestamp=timestamp,
                points=points.astype(np.float32),
                image_snr=5.0 + 4.0 * np.cos(2.0 * np.pi * phase) ** 2,
                contour_score=float(np.clip(0.85 - 0.20 * abs(phase - 0.5), 0.4, 0.95)),
                angle_score=float(np.clip(0.75 + 0.15 * np.sin(2.0 * np.pi * phase), 0.4, 0.95)),
                pose_score=float(np.clip(0.90 - 0.25 * abs(phase - 0.25), 0.4, 0.95)),
            )
        )
    return samples, timestamps, ground_truth


def phase_key(value: float) -> float:
    return float(np.round(value % 1.0, 2))


def main() -> None:
    reference_vertices = build_reference_surface()
    samples, timestamps, ground_truth_by_phase = make_observation_samples(reference_vertices)

    sample_cycles = [
        CycleAnchor(0.0, 0.46, 1.0),
        CycleAnchor(1.0, 1.48, 2.0),
    ]
    reference_cycles = [
        CycleAnchor(0.0, 0.50, 1.0),
        CycleAnchor(1.0, 1.50, 2.0),
    ]
    canonicalizer = NonlinearPhaseCanonicalizer()
    assigned_phases, summary = canonicalizer.assign_phases(timestamps, sample_cycles, reference_cycles)

    phase_centers = [0.0, 0.25, 0.5, 0.75, 1.0]
    builder = WeightedPhaseSetBuilder(phase_centers=phase_centers, phase_radius=0.18)
    phase_sets = builder.build(samples, assigned_phases)
    reference_selection = select_reference_phase(phase_sets)

    trainer = GBRTrainer(
        GBRConfig(
            basis_rank=4,
            hidden_dim=48,
            hidden_layers=2,
            phase_harmonics=4,
            learning_rate=1e-2,
            steps=220,
            temporal_weight=0.05,
            periodic_weight=0.05,
            residual_weight=1e-3,
        )
    )
    result = trainer.fit(reference_vertices=reference_vertices, phase_sets=phase_sets)

    phase_vertex_rmse: dict[str, float] = {}
    for phase_center, predicted_vertices in zip(result.phase_centers, result.deformed_vertices):
        wrapped = phase_key(phase_center)
        nearest_gt_key = min(ground_truth_by_phase.keys(), key=lambda candidate: abs(phase_key(candidate) - wrapped))
        ground_truth = ground_truth_by_phase[nearest_gt_key]
        rmse = float(np.sqrt(np.mean((predicted_vertices - ground_truth) ** 2)))
        phase_vertex_rmse[f"{phase_center:.2f}"] = rmse

    payload = {
        "demo": "synthetic_demo",
        "reference_vertex_count": int(len(reference_vertices)),
        "sample_count": int(len(samples)),
        "weighted_phase_set_count": int(len(phase_sets)),
        "phase_centers": [float(value) for value in result.phase_centers],
        "canonicalization": {
            "target_peak_phase": summary.target_peak_phase,
            "reference_peak_phases": summary.reference_peak_phases,
            "cycle_count": summary.cycle_count,
        },
        "reference_topology": {
            "reference_phase": reference_selection.reference_phase,
            "support_score": reference_selection.support_score,
        },
        "loss": {
            "final_total": float(result.loss_history[-1]),
            "data": result.data_loss,
            "temporal": result.temporal_loss,
            "periodic": result.periodic_loss,
            "residual": result.residual_loss,
        },
        "phase_vertex_rmse": phase_vertex_rmse,
        "loss_history": [float(value) for value in result.loss_history],
    }
    export_result_summary(OUTPUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved demo summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()