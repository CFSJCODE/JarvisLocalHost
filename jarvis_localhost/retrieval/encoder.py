"""Bidirectional retrieval encoder trained only on the canonical corpus."""

from __future__ import annotations

import json
import math
import os
import tempfile
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from jarvis_localhost.ai.language_model import LayerNorm
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.sovereign import POLICY, SovereignPolicy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RetrieverConfig:
    vocab_size: int
    context_len: int = 192
    embed_dim: int = 192
    projection_dim: int = 128
    num_heads: int = 6
    num_layers: int = 4
    ff_dim: int = 768
    dropout: float = 0.1

    def validate(self) -> None:
        if min(
            self.vocab_size,
            self.context_len,
            self.embed_dim,
            self.projection_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
        ) <= 0:
            raise ValueError("retriever dimensions must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")


class BidirectionalAttention(nn.Module):
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.embed_dim = config.embed_dim
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = values.shape
        query, key, value = self.qkv(values).split(self.embed_dim, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = map(heads, (query, key, value))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = mask[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~key_mask, -1e4)
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.output(attended) * mask.unsqueeze(-1).to(values.dtype)


class BidirectionalBlock(nn.Module):
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.attention_norm = LayerNorm(config.embed_dim)
        self.attention = BidirectionalAttention(config)
        self.ff_norm = LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values), mask)
        values = values + self.ff(self.ff_norm(values))
        return values * mask.unsqueeze(-1).to(values.dtype)


class RetrieverEncoder(nn.Module):
    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        config: RetrieverConfig,
        *,
        seed: int = 1_337,
        policy: SovereignPolicy | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.cfg = config
        self.seed = int(seed)
        self.policy = policy or POLICY
        self.policy.assert_random_initialization("random")
        self.trained_on_corpus = False
        torch.manual_seed(self.seed)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            BidirectionalBlock(config) for _ in range(config.num_layers)
        )
        self.norm = LayerNorm(config.embed_dim)
        self.projection = nn.Linear(config.embed_dim, config.projection_dim, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have equal (B, T) shapes")
        if input_ids.shape[1] > self.cfg.context_len:
            raise ValueError("retriever sequence exceeds configured context")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        values = self.token_embedding(input_ids) + self.position_embedding(positions)
        values = self.dropout(values)
        for block in self.blocks:
            values = block(values, attention_mask)
        values = self.norm(values)
        weights = attention_mask.unsqueeze(-1).to(values.dtype)
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.projection(pooled), dim=-1)

    def tokenize(
        self, tokenizer: JarvisTokenizer, texts: list[str], device: Any,
        *, padding_policy: str = "fixed", min_bucket_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad to the configured context or an opt-in bounded power-of-two bucket.

        Token order and truncation at context_len are identical in both modes.
        Bucketing changes tensor shapes and therefore the stochastic execution
        sequence during training with dropout; it changes no learned weights.
        """
        if padding_policy not in {"fixed", "bucketed"}:
            raise ValueError("padding_policy must be fixed or bucketed")
        if type(min_bucket_size) is not int or min_bucket_size not in {32, 64}:
            raise ValueError("min_bucket_size must be 32 or 64")
        encoded = [
            tokenizer.encode(text, add_special=True)[: self.cfg.context_len]
            for text in texts
        ]
        length = self.cfg.context_len
        if padding_policy == "bucketed":
            longest = max((len(ids) for ids in encoded), default=1)
            bucket = 1 << (max(1, longest) - 1).bit_length()
            length = min(self.cfg.context_len, max(min_bucket_size, bucket))
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for ids in encoded:
            mask = [1] * len(ids)
            rows.append(tokenizer.pad_sequence(ids, length))
            masks.append(mask + [0] * (length - len(mask)))
        return (
            torch.tensor(rows, dtype=torch.long, device=device).reshape(len(rows), length),
            torch.tensor(masks, dtype=torch.long, device=device).reshape(len(rows), length),
        )

    @torch.no_grad()
    def embed_texts(
        self,
        tokenizer: JarvisTokenizer,
        texts: list[str],
        *,
        device: Any = "cpu",
        batch_size: int = 32,
    ) -> torch.Tensor:
        self.to(device)
        self.eval()
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + batch_size]
            ids, mask = self.tokenize(tokenizer, batch, device)
            outputs.append(self(ids, mask).detach().cpu())
        if not outputs:
            return torch.empty((0, self.cfg.projection_dim), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def save(
        self,
        path: str | Path,
        *,
        corpus_sha256: str,
        tokenizer_sha256: str,
        training: dict | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(
                {
                    "format_version": self.CHECKPOINT_VERSION,
                    "config": asdict(self.cfg),
                    "seed": self.seed,
                    "trained_on_corpus": self.trained_on_corpus,
                    "state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in self.state_dict().items()
                    },
                },
                temporary,
            )
            with temporary.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "format_version": self.CHECKPOINT_VERSION,
            "model": "jarvis-retriever",
            "initialized_from": "random",
            "external_weights": False,
            "corpus_sha256": corpus_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "architecture": asdict(self.cfg),
            "seed": self.seed,
            "trained_on_corpus": self.trained_on_corpus,
            "checkpoint_sha256": _sha256(destination),
            "checkpoint_bytes": destination.stat().st_size,
            "training": {"seed": self.seed, **(training or {})},
        }
        _atomic_json(
            destination.with_suffix(destination.suffix + ".manifest.json"),
            manifest,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: Any = "cpu",
        policy: SovereignPolicy | None = None,
        expected_corpus_sha256: str | None = None,
        expected_tokenizer_sha256: str | None = None,
    ) -> "RetrieverEncoder":
        source = Path(path)
        active_policy = policy or POLICY
        manifest_path = source.with_suffix(source.suffix + ".manifest.json")
        if not source.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("retriever checkpoint or manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported retriever checkpoint manifest version")
        if manifest.get("model") != "jarvis-retriever":
            raise ValueError("unexpected retriever checkpoint model type")
        if manifest.get("checkpoint_bytes") != source.stat().st_size:
            raise ValueError("retriever checkpoint size mismatch")
        if manifest.get("checkpoint_sha256") != _sha256(source):
            raise ValueError("retriever checkpoint checksum mismatch")
        if active_policy.enabled:
            active_policy.assert_random_initialization(manifest.get("initialized_from", ""))
            if manifest.get("external_weights") is not False:
                raise ValueError("retriever declares external weights")
            if not expected_corpus_sha256 or not expected_tokenizer_sha256:
                raise ValueError("sovereign retriever load requires expected lineage digests")
            if manifest.get("corpus_sha256") != expected_corpus_sha256:
                raise ValueError("retriever corpus digest mismatch")
            if manifest.get("tokenizer_sha256") != expected_tokenizer_sha256:
                raise ValueError("retriever tokenizer digest mismatch")
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported retriever checkpoint version")
        if checkpoint.get("config") != manifest.get("architecture"):
            raise ValueError("retriever architecture differs from its manifest")
        if int(checkpoint.get("seed", -1)) != int(manifest.get("seed", -2)):
            raise ValueError("retriever seed differs from its manifest")
        if bool(checkpoint.get("trained_on_corpus", False)) != bool(
            manifest.get("trained_on_corpus", False)
        ):
            raise ValueError("retriever training state differs from its manifest")
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("retriever checkpoint has no state dictionary")
        if any(
            not isinstance(tensor, torch.Tensor)
            or not bool(torch.isfinite(tensor).all().item())
            for tensor in state_dict.values()
        ):
            raise ValueError("retriever checkpoint contains invalid tensors")
        encoder = cls(
            RetrieverConfig(**checkpoint["config"]),
            seed=int(checkpoint.get("seed", 1_337)),
            policy=active_policy,
        )
        encoder.load_state_dict(state_dict, strict=True)
        encoder.trained_on_corpus = bool(checkpoint.get("trained_on_corpus", False))
        return encoder.to(device)


__all__ = ["RetrieverConfig", "RetrieverEncoder"]
