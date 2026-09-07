"""Randomly initialized feature encoder for corpus observations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityEncoder(nn.Module):
    def __init__(self, observation_dim: int, feature_dim: int = 128):
        super().__init__()
        if observation_dim <= 0 or feature_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        hidden = max(feature_dim, min(512, observation_dim * 2))
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(observations), dim=-1)


__all__ = ["CuriosityEncoder"]
