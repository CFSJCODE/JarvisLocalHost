"""Actor-critic policy for navigating the corpus environment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _selected_log_probabilities(
    log_probabilities: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Select categorical log-probabilities without a device-side scatter.

    DirectML 0.2.5 cannot backpropagate through the ``gather`` pattern used by
    PyTorch distributions for this shape (it reports a partially modified
    scatter dimension).  Actions are tiny categorical indices, so constructing
    their constant one-hot selector on CPU and transferring that matrix keeps
    the differentiable reduction on the accelerator without a CPU gradient
    detour.
    """

    if log_probabilities.ndim < 2:
        raise ValueError("log_probabilities must include an action dimension")
    if actions.shape != log_probabilities.shape[:-1]:
        raise ValueError("actions must match the non-action probability dimensions")
    action_count = log_probabilities.shape[-1]
    if action_count <= 0:
        raise ValueError("action dimension must be positive")
    selectors = F.one_hot(
        actions.detach().cpu().long(), num_classes=action_count
    ).to(device=log_probabilities.device, dtype=log_probabilities.dtype)
    return (log_probabilities * selectors).sum(dim=-1)


class ActorCriticPolicy(nn.Module):
    def __init__(self, feature_dim: int, action_count: int, hidden_dim: int = 192):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_count)
        self.critic = nn.Linear(hidden_dim, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            gain = 0.01 if module.out_features > 1 else 1.0
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(
        self, features: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(features)
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        action = (
            probabilities.argmax(dim=-1)
            if deterministic
            else self._sample(probabilities)
        )
        log_probability = _selected_log_probabilities(log_probabilities, action)
        return action, log_probability, value

    @staticmethod
    def _sample(probabilities: torch.Tensor) -> torch.Tensor:
        try:
            return torch.multinomial(probabilities, 1).squeeze(-1)
        except (RuntimeError, NotImplementedError):
            # Some DirectML releases do not implement multinomial. Sampling is
            # cheap relative to the policy network, and the selected indices
            # are transferred back without moving the differentiable logits.
            return torch.multinomial(
                probabilities.detach().cpu(), 1
            ).squeeze(-1).to(probabilities.device)

    def evaluate_actions(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(features)
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        selected = _selected_log_probabilities(log_probabilities, actions)
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        return selected, values, entropy


Policy = ActorCriticPolicy


__all__ = ["ActorCriticPolicy", "Policy"]
