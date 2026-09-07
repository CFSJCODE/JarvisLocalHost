"""Compatibility facade for the former monolithic neural module.

New code should import the focused modules directly.  Keeping these re-exports
allows existing callers to migrate without retaining duplicate implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jarvis_localhost.ai.dataset import TextDataset
from jarvis_localhost.ai.language_model import (
    FeedForward,
    JarvisConfig,
    JarvisTransformer,
    LayerNorm,
    MultiHeadCausalAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock,
)
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import JarvisTrainer, TrainConfig, TrainingCancelled
from jarvis_localhost.curiosity.engine import CuriosityEngine, CuriosityState, Insight
from jarvis_localhost.rag.engine import RAGEngine, RAGMode, RAGResponse
from jarvis_localhost.retrieval.vector_store import VectorStore


class EmbeddingEngine:
    """Legacy LM-pooling adapter; production retrieval uses RetrieverEncoder."""

    def __init__(
        self,
        model: JarvisTransformer,
        tokenizer: JarvisTokenizer,
        device: Any = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def embed(self, text: str, max_len: int = 128) -> np.ndarray:
        length = min(max_len, self.model.cfg.context_len)
        encoded = self.tokenizer.encode(text, add_special=True)[:length]
        mask = [1] * len(encoded)
        padded = self.tokenizer.pad_sequence(encoded, length)
        ids = torch.tensor([padded], dtype=torch.long, device=self.device)
        weights = torch.tensor(mask + [0] * (length - len(mask)), device=self.device)
        weights = weights.unsqueeze(0).unsqueeze(-1).to(torch.float32)
        hidden = self.model.hidden_states(ids)
        vector = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(vector, dim=-1).squeeze(0).cpu().numpy()

    def embed_batch(self, texts: list[str], max_len: int = 128) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.cfg.embed_dim), dtype=np.float32)
        return np.stack([self.embed(text, max_len) for text in texts])


__all__ = [
    "CuriosityEngine",
    "CuriosityState",
    "EmbeddingEngine",
    "FeedForward",
    "Insight",
    "JarvisConfig",
    "JarvisTokenizer",
    "JarvisTrainer",
    "JarvisTransformer",
    "LayerNorm",
    "MultiHeadCausalAttention",
    "RAGEngine",
    "RAGMode",
    "RAGResponse",
    "SinusoidalPositionalEncoding",
    "TextDataset",
    "TrainConfig",
    "TrainingCancelled",
    "TransformerBlock",
    "VectorStore",
]
