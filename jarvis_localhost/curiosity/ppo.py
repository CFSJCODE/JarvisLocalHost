"""Proximal Policy Optimization driven by ICM prediction error."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jarvis_localhost.curiosity.environment import CorpusEnv
from jarvis_localhost.curiosity.icm import IntrinsicCuriosityModule
from jarvis_localhost.curiosity.memory import RolloutMemory, TransitionRecord
from jarvis_localhost.curiosity.policy import ActorCriticPolicy


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 96
    update_epochs: int = 4
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    learning_rate: float = 3e-4
    icm_learning_rate: float = 3e-4
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if min(self.rollout_steps, self.update_epochs, self.minibatch_size) <= 0:
            raise ValueError("PPO step/epoch/batch values must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid PPO discount parameters")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0,1)")


@dataclass(frozen=True)
class CuriosityCycle:
    metrics: dict
    records: tuple[TransitionRecord, ...]


class CuriosityAgent:
    def __init__(
        self,
        environment: CorpusEnv,
        *,
        feature_dim: int = 128,
        device: Any = "cpu",
        config: PPOConfig | None = None,
        seed: int = 1_337,
    ) -> None:
        self.environment = environment
        self.device = device
        self.config = config or PPOConfig()
        self.config.validate()
        torch.manual_seed(seed)
        self.icm = IntrinsicCuriosityModule(
            environment.observation_dim,
            environment.action_count,
            feature_dim=feature_dim,
        ).to(device)
        self.policy = ActorCriticPolicy(feature_dim, environment.action_count).to(device)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-3,
            foreach=False,
        )
        self.icm_optimizer = torch.optim.AdamW(
            self.icm.parameters(),
            lr=self.config.icm_learning_rate,
            weight_decay=1e-4,
            foreach=False,
        )

    def _tensor(self, observation: np.ndarray) -> torch.Tensor:
        return torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _collect(self) -> tuple[RolloutMemory, float]:
        memory = RolloutMemory()
        observation = self.environment.reset()
        for _ in range(self.config.rollout_steps):
            state_tensor = self._tensor(observation)
            with torch.no_grad():
                features = self.icm.encoder(state_tensor)
                action, log_probability, value = self.policy.act(features)
            next_observation, external_reward, done, info = self.environment.step(
                int(action.item())
            )
            with torch.no_grad():
                curiosity = self.icm.calculate(
                    state_tensor,
                    action,
                    self._tensor(next_observation),
                ).intrinsic_reward
            transition = info["transition"]
            reward = float(curiosity.item()) + float(external_reward)
            memory.add(
                TransitionRecord(
                    state=observation.copy(),
                    action=int(action.item()),
                    log_probability=float(log_probability.item()),
                    value=float(value.item()),
                    reward=reward,
                    done=done,
                    next_state=next_observation.copy(),
                    previous_chunk_id=transition.previous_chunk_id,
                    next_chunk_id=transition.next_chunk_id,
                )
            )
            observation = self.environment.reset() if done else next_observation
        with torch.no_grad():
            last_feature = self.icm.encoder(self._tensor(observation))
            _, last_value = self.policy(last_feature)
        return memory, float(last_value.item())

    def train_cycle(self) -> CuriosityCycle:
        memory, last_value = self._collect()
        tensors = memory.tensors(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.device,
        )
        with torch.no_grad():
            features = self.icm.encoder(tensors["states"]).detach()
        count = len(memory)
        minibatch = min(self.config.minibatch_size, count)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        self.policy.train()
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, minibatch):
                indices = permutation[start : start + minibatch]
                log_probabilities, values, entropy = self.policy.evaluate_actions(
                    features[indices], tensors["actions"][indices]
                )
                ratio = torch.exp(
                    log_probabilities - tensors["old_log_probabilities"][indices]
                )
                advantages = tensors["advantages"][indices]
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, tensors["returns"][indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_weight * value_loss
                    - self.config.entropy_weight * entropy_mean
                )
                self.policy_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.policy_optimizer.step()
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_mean.detach().cpu().item()))

        self.icm.train()
        icm_output = self.icm.calculate(
            tensors["states"], tensors["actions"], tensors["next_states"]
        )
        self.icm_optimizer.zero_grad(set_to_none=True)
        icm_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.config.max_grad_norm)
        self.icm_optimizer.step()
        rewards = [record.reward for record in memory.records]
        metrics = {
            "algorithm": "ICM+PPO",
            "steps": count,
            "mean_intrinsic_reward": sum(rewards) / len(rewards),
            "max_intrinsic_reward": max(rewards),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": sum(entropies) / len(entropies),
            "icm_inverse_loss": float(icm_output.inverse_loss.detach().cpu().item()),
            "icm_forward_loss": float(icm_output.forward_loss.detach().cpu().item()),
            "config": asdict(self.config),
        }
        return CuriosityCycle(metrics=metrics, records=tuple(memory.records))


PPO = CuriosityAgent


__all__ = ["CuriosityAgent", "CuriosityCycle", "PPO", "PPOConfig"]
