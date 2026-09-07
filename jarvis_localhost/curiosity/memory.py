"""Rollout memory and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TransitionRecord:
    state: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float
    done: bool
    next_state: np.ndarray
    previous_chunk_id: str
    next_chunk_id: str


class RolloutMemory:
    """A bounded, single-cycle transition buffer."""

    DEFAULT_MAX_RECORDS = 4_096

    def __init__(self, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        if int(max_records) <= 0:
            raise ValueError("max_records must be positive")
        self.max_records = int(max_records)
        self.records: list[TransitionRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: TransitionRecord) -> None:
        if len(self.records) >= self.max_records:
            raise OverflowError("rollout memory capacity exceeded")
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def tensors(
        self,
        *,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        if not self.records:
            raise ValueError("rollout memory is empty")
        advantages = np.zeros(len(self.records), dtype=np.float32)
        returns = np.zeros(len(self.records), dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for index in range(len(self.records) - 1, -1, -1):
            record = self.records[index]
            not_done = 0.0 if record.done else 1.0
            delta = record.reward + gamma * next_value * not_done - record.value
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[index] = gae
            returns[index] = gae + record.value
            next_value = record.value
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        if len(advantages) > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )
        return {
            "states": torch.tensor(
                np.stack([record.state for record in self.records]),
                dtype=torch.float32,
                device=device,
            ),
            "next_states": torch.tensor(
                np.stack([record.next_state for record in self.records]),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.tensor(
                [record.action for record in self.records], dtype=torch.long, device=device
            ),
            "old_log_probabilities": torch.tensor(
                [record.log_probability for record in self.records],
                dtype=torch.float32,
                device=device,
            ),
            "advantages": advantage_tensor,
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
        }


__all__ = ["RolloutMemory", "TransitionRecord"]
