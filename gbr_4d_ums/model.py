from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhaseEncoder(nn.Module):
    def __init__(self, harmonics: int = 4) -> None:
        super().__init__()
        self.harmonics = max(int(harmonics), 1)
        self.output_dim = 2 * self.harmonics

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        phase = phase.reshape(-1, 1)
        encoded = []
        for frequency in range(1, self.harmonics + 1):
            encoded.append(torch.sin(2.0 * math.pi * frequency * phase))
            encoded.append(torch.cos(2.0 * math.pi * frequency * phase))
        return torch.cat(encoded, dim=-1)


class ResidualField(nn.Module):
    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2, phase_harmonics: int = 4) -> None:
        super().__init__()
        self.phase_encoder = PhaseEncoder(phase_harmonics)
        input_dim = 3 + self.phase_encoder.output_dim
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 3))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(linear_layers[-1].weight, 0.0)
        nn.init.constant_(linear_layers[-1].bias, 0.0)

    def forward(self, reference_vertices: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        batch_size, vertex_count = phase.numel(), reference_vertices.shape[0]
        repeated_vertices = reference_vertices.unsqueeze(0).expand(batch_size, -1, -1)
        phase_features = self.phase_encoder(phase).unsqueeze(1).expand(-1, vertex_count, -1)
        features = torch.cat([repeated_vertices, phase_features], dim=-1)
        return self.network(features)


class BasisCoefficientHead(nn.Module):
    def __init__(self, basis_rank: int, phase_harmonics: int = 4) -> None:
        super().__init__()
        self.phase_encoder = PhaseEncoder(phase_harmonics)
        self.linear = nn.Linear(self.phase_encoder.output_dim, int(basis_rank))
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return self.linear(self.phase_encoder(phase))


class GlobalBasisResidualModel(nn.Module):
    def __init__(
        self,
        reference_vertices: torch.Tensor,
        basis_rank: int = 4,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        phase_harmonics: int = 4,
    ) -> None:
        super().__init__()
        self.register_buffer("reference_vertices", reference_vertices.clone().detach())
        vertex_count = int(reference_vertices.shape[0])
        self.mean_offset = nn.Parameter(torch.zeros((vertex_count, 3), dtype=reference_vertices.dtype))
        self.global_basis = nn.Parameter(torch.zeros((int(basis_rank), vertex_count, 3), dtype=reference_vertices.dtype))
        self.coefficient_head = BasisCoefficientHead(basis_rank=int(basis_rank), phase_harmonics=phase_harmonics)
        self.residual_field = ResidualField(
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            phase_harmonics=phase_harmonics,
        )

    def coefficients(self, phase: torch.Tensor) -> torch.Tensor:
        return self.coefficient_head(phase)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        if phase.dim() == 0:
            phase = phase.unsqueeze(0)
        coefficients = self.coefficients(phase)
        global_offsets = torch.einsum("br,rvc->bvc", coefficients, self.global_basis)
        residual_offsets = self.residual_field(self.reference_vertices, phase)
        return self.reference_vertices.unsqueeze(0) + self.mean_offset.unsqueeze(0) + global_offsets + residual_offsets