"""Intrinsic Curiosity Module with inverse and forward dynamics models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from jarvis_localhost.curiosity.encoder import CuriosityEncoder


@dataclass(frozen=True)
class ICMOutput:
    intrinsic_reward: torch.Tensor
    inverse_loss: torch.Tensor
    forward_loss: torch.Tensor
    total_loss: torch.Tensor


class IntrinsicCuriosityModule(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        *,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        beta: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        super().__init__()
        if action_count <= 1:
            raise ValueError("ICM requires at least two actions")
        if not 0.0 <= beta <= 1.0 or eta <= 0:
            raise ValueError("beta must be in [0,1] and eta must be positive")
        self.action_count = action_count
        self.feature_dim = feature_dim
        self.beta = beta
        self.eta = eta
        self.encoder = CuriosityEncoder(observation_dim, feature_dim)
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_count),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.inverse_model.apply(self._initialize)
        self.forward_model.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    def calculate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> ICMOutput:
        features = self.encoder(states)
        next_features = self.encoder(next_states)
        inverse_logits = self.inverse_model(torch.cat((features, next_features), dim=-1))
        inverse_loss = F.cross_entropy(inverse_logits, actions.long())
        try:
            action_one_hot = F.one_hot(actions.long(), self.action_count).to(
                features.dtype
            )
        except (RuntimeError, NotImplementedError):
            # DirectML 0.2.5 cannot scatter one-hot values for this shape. The
            # action tensor is tiny, so encode it on CPU and transfer only the
            # resulting categorical matrix.
            action_one_hot = F.one_hot(
                actions.detach().cpu().long(), self.action_count
            ).to(device=features.device, dtype=features.dtype)
        predicted_next = self.forward_model(torch.cat((features, action_one_hot), dim=-1))
        errors = 0.5 * (predicted_next - next_features.detach()).pow(2).mean(dim=-1)
        forward_loss = errors.mean()
        total_loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        return ICMOutput(
            intrinsic_reward=self.eta * errors.detach(),
            inverse_loss=inverse_loss,
            forward_loss=forward_loss,
            total_loss=total_loss,
        )


ICM = IntrinsicCuriosityModule


__all__ = ["ICM", "ICMOutput", "IntrinsicCuriosityModule"]
