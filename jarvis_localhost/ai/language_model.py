"""Decoder-only language model initialized entirely from random weights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.sovereign import POLICY, SovereignPolicy


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class JarvisConfig:
    vocab_size: int = 8_000
    context_len: int = 256
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 1_024
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "context_len": self.context_len,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError(f"model dimensions must be positive: {integer_fields}")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


class LayerNorm(nn.Module):
    def __init__(self, dimension: int, bias: bool = True, epsilon: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension)) if bias else None
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        variance = (values - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (values - mean) * torch.rsqrt(variance + self.epsilon)
        output = normalized * self.weight
        return output + self.bias if self.bias is not None else output


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, config: JarvisConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        self.output = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_len, config.context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_len, config.context_len))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = values.shape
        query, key, value = self.qkv(values).split(self.embed_dim, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :sequence, :sequence], -1e4)
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.residual_dropout(self.output(attended))


class FeedForward(nn.Module):
    def __init__(self, config: JarvisConfig):
        super().__init__()
        self.input = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)
        self.output = nn.Linear(config.ff_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output(F.gelu(self.input(values))))


class TransformerBlock(nn.Module):
    def __init__(self, config: JarvisConfig):
        super().__init__()
        self.attention_norm = LayerNorm(config.embed_dim, config.bias)
        self.attention = MultiHeadCausalAttention(config)
        self.feed_forward_norm = LayerNorm(config.embed_dim, config.bias)
        self.feed_forward = FeedForward(config)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values))
        return values + self.feed_forward(self.feed_forward_norm(values))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisors = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embed_dim)
        )
        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(positions * divisors)
        if embed_dim > 1:
            encoding[:, 1::2] = torch.cos(positions * divisors[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]].to(values.dtype)


class JarvisTransformer(nn.Module):
    """GPT-style model with explicit sovereign checkpoint lineage."""

    CHECKPOINT_VERSION = 3

    def __init__(
        self,
        config: JarvisConfig,
        *,
        policy: SovereignPolicy | None = None,
        seed: int = 1_337,
    ) -> None:
        super().__init__()
        config.validate()
        self.cfg = config
        self.policy = policy or POLICY
        self.policy.assert_random_initialization("random")
        self.seed = int(seed)
        self.control_training_metadata: dict[str, Any] | None = None
        self.training_sampling_metadata: dict[str, Any] | None = None
        torch.manual_seed(self.seed)
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(config.embed_dim, config.context_len)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.ln_final = LayerNorm(config.embed_dim, config.bias)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.cfg.context_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds {self.cfg.context_len}"
            )
        hidden = self.drop(self.pos_enc(self.token_embed(input_ids)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.ln_final(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.head(self.hidden_states(input_ids))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=JarvisTokenizer.PAD_ID,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        output = prompt_ids.clone()
        finished = torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
        for _ in range(max(0, int(max_new))):
            context = output[:, -self.cfg.context_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1] / temperature
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                next_logits = next_logits.masked_fill(
                    next_logits < values[:, -1].unsqueeze(-1), -1e4
                )
            probabilities = F.softmax(next_logits, dim=-1)
            try:
                next_token = torch.multinomial(
                    probabilities, 1, generator=generator
                )
            except (RuntimeError, NotImplementedError):
                # DirectML releases occasionally lack multinomial. Sampling a
                # single token on CPU preserves the distribution and keeps all
                # expensive Transformer work on the accelerator.
                next_token = torch.multinomial(
                    probabilities.detach().cpu(), 1, generator=generator
                ).to(output.device)
            eos = torch.full_like(next_token, JarvisTokenizer.EOS_ID)
            next_token = torch.where(finished.unsqueeze(-1), eos, next_token)
            output = torch.cat((output, next_token), dim=1)
            finished |= next_token.squeeze(-1).eq(JarvisTokenizer.EOS_ID)
            if bool(finished.all().item()):
                break
        return output

    def count_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def lineage_manifest(
        self,
        *,
        corpus_sha256: str,
        tokenizer_sha256: str,
        training: dict[str, Any] | None = None,
        checkpoint_filename: str = "",
        checkpoint_sha256: str = "",
        checkpoint_size_bytes: int = 0,
    ) -> dict[str, Any]:
        return {
            "format_version": self.CHECKPOINT_VERSION,
            "artifact_envelope_version": 1,
            "model": "jarvis-lm",
            "initialized_from": "random",
            "external_weights": False,
            "corpus_sha256": corpus_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "architecture": asdict(self.cfg),
            "training": {"seed": self.seed, **(training or {})},
            "sovereign_policy": self.policy.to_manifest(),
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": int(checkpoint_size_bytes),
        }

    def save(
        self,
        path: str | Path,
        *,
        corpus_sha256: str = "",
        tokenizer_sha256: str = "",
        training: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        training_payload = dict(training or {})
        if (
            self.control_training_metadata
            and "control_training" not in training_payload
        ):
            training_payload["control_training"] = dict(
                self.control_training_metadata
            )
        if (
            self.training_sampling_metadata
            and "sampling" not in training_payload
        ):
            training_payload["sampling"] = dict(self.training_sampling_metadata)
        checkpoint = {
            "format_version": self.CHECKPOINT_VERSION,
            "config": asdict(self.cfg),
            "seed": self.seed,
            # PrivateUse1/DirectML tensors are not portable checkpoint payloads.
            # Persist a CPU snapshot so weights-only loading works everywhere.
            "state_dict": {
                name: tensor.detach().cpu() for name, tensor in self.state_dict().items()
            },
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(checkpoint, temporary)
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            checkpoint_sha256, checkpoint_size = _sha256_path(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
        _atomic_json(
            manifest_path,
            self.lineage_manifest(
                corpus_sha256=corpus_sha256,
                tokenizer_sha256=tokenizer_sha256,
                training=training_payload,
                checkpoint_filename=destination.name,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_size_bytes=checkpoint_size,
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Any = "cpu",
        *,
        policy: SovereignPolicy | None = None,
        expected_corpus_sha256: str | None = None,
        expected_tokenizer_sha256: str | None = None,
    ) -> "JarvisTransformer":
        source = Path(path)
        active_policy = policy or POLICY
        manifest_path = source.with_suffix(source.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ValueError("checkpoint is missing its cryptographic lineage manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", 0)) != cls.CHECKPOINT_VERSION:
            raise ValueError("legacy or unsupported checkpoint manifest")
        if int(manifest.get("artifact_envelope_version", 0)) != 1:
            raise ValueError("checkpoint manifest has no supported artifact envelope")
        if manifest.get("checkpoint_filename") != source.name:
            raise ValueError("checkpoint filename does not match its manifest")
        expected_digest = str(manifest.get("checkpoint_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            raise ValueError("checkpoint manifest has an invalid SHA-256 digest")
        expected_size = manifest.get("checkpoint_size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError("checkpoint manifest has an invalid byte size")
        if active_policy.enabled:
            active_policy.assert_random_initialization(manifest.get("initialized_from", ""))
            if manifest.get("external_weights") is not False:
                raise ValueError("checkpoint declares external weights")
            if expected_corpus_sha256 and manifest.get("corpus_sha256") != expected_corpus_sha256:
                raise ValueError("checkpoint corpus digest does not match")
            if (
                expected_tokenizer_sha256
                and manifest.get("tokenizer_sha256") != expected_tokenizer_sha256
            ):
                raise ValueError("checkpoint tokenizer digest does not match")

        digest = hashlib.sha256()
        actual_size = 0
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                actual_size += len(block)
            if actual_size != expected_size or digest.hexdigest() != expected_digest:
                raise ValueError("checkpoint bytes do not match the lineage manifest")
            handle.seek(0)
            try:
                checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            except TypeError as exc:
                raise RuntimeError(
                    "this PyTorch version cannot safely load weights-only checkpoints"
                ) from exc
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("invalid Jarvis checkpoint")
        if int(checkpoint.get("format_version", 0)) != cls.CHECKPOINT_VERSION:
            raise ValueError("legacy or unsupported Jarvis checkpoint")
        model = cls(
            JarvisConfig(**checkpoint["config"]),
            policy=active_policy,
            seed=int(checkpoint.get("seed", 1_337)),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        control_training = (manifest.get("training") or {}).get("control_training")
        if isinstance(control_training, dict):
            model.control_training_metadata = dict(control_training)
        sampling = (manifest.get("training") or {}).get("sampling")
        if isinstance(sampling, dict):
            model.training_sampling_metadata = dict(sampling)
        model.to(device)
        return model


__all__ = [
    "FeedForward",
    "JarvisConfig",
    "JarvisTransformer",
    "LayerNorm",
    "MultiHeadCausalAttention",
    "SinusoidalPositionalEncoding",
    "TransformerBlock",
]
