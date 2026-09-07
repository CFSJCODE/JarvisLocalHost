"""Datasets and corpus measurements used to size/train local models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset


SAMPLING_FORMAT_VERSION = 1


def validate_sampling_payload(
    payload: Mapping[str, Any],
    *,
    expected_corpus_sha256: str,
    expected_chunk_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Validate the curiosity engine's corpus-bound sampling contract."""

    if int(payload.get("format_version", 0)) != SAMPLING_FORMAT_VERSION:
        raise ValueError("unsupported sampling-weight format")
    if not expected_corpus_sha256 or payload.get("corpus_sha256") != expected_corpus_sha256:
        raise ValueError("sampling weights belong to a different corpus")
    identifiers = [str(value) for value in payload.get("chunk_ids", ())]
    expected = [str(value) for value in expected_chunk_ids]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("sampling payload has invalid or duplicate chunk IDs")
    if len(expected) != len(set(expected)) or set(identifiers) != set(expected):
        raise ValueError("sampling weights do not cover the active chunk snapshot")
    raw_weights = payload.get("weights", ())
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != len(identifiers):
        raise ValueError("sampling payload weight count does not match its chunk IDs")
    weights = [float(value) for value in raw_weights]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("sampling weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("sampling weights must form a normalized distribution")
    signal_version = int(payload.get("signal_version", -1))
    if signal_version < 0:
        raise ValueError("sampling payload has an invalid signal version")
    mapping = dict(zip(identifiers, weights))
    metadata = {
        "format_version": SAMPLING_FORMAT_VERSION,
        "corpus_sha256": expected_corpus_sha256,
        "signal_version": signal_version,
        "generation": payload.get("generation"),
        "weighted": True,
    }
    return mapping, metadata


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    pages: int
    words: int
    tokens: int
    vocabulary: int
    bytes_utf8: int
    languages: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def measure_corpus(
    texts: Sequence[str],
    token_ids: Sequence[int],
    vocabulary: int,
    *,
    pages: int = 0,
    languages: Sequence[str] = (),
) -> CorpusStats:
    return CorpusStats(
        documents=len(texts),
        pages=max(0, int(pages)),
        words=sum(len(text.split()) for text in texts),
        tokens=len(token_ids),
        vocabulary=int(vocabulary),
        bytes_utf8=sum(len(text.encode("utf-8")) for text in texts),
        languages=tuple(sorted({language for language in languages if language})),
    )


class TextDataset(Dataset):
    """Strided next-token windows with complete coverage of the token stream."""

    def __init__(
        self,
        token_ids: Sequence[int],
        context_len: int,
        *,
        stride: int | None = None,
    ) -> None:
        if context_len <= 0:
            raise ValueError("context_len must be positive")
        self.ids = tuple(int(token_id) for token_id in token_ids)
        self.ctx = int(context_len)
        self.stride = max(1, int(stride or max(1, context_len // 2)))
        last_offset = len(self.ids) - self.ctx - 1
        if last_offset < 0:
            self.offsets = ()
        else:
            self.offsets = tuple(range(0, last_offset + 1, self.stride))
            if self.offsets[-1] != last_offset:
                self.offsets += (last_offset,)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = self.offsets[index]
        window = self.ids[offset : offset + self.ctx + 1]
        return (
            torch.tensor(window[:-1], dtype=torch.long),
            torch.tensor(window[1:], dtype=torch.long),
        )


def deterministic_split(
    token_ids: Sequence[int], validation_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not token_ids:
        return [], []
    validation_count = int(math.floor(len(token_ids) * validation_fraction))
    if validation_count == 0:
        return list(token_ids), []
    split = len(token_ids) - validation_count
    return list(token_ids[:split]), list(token_ids[split:])


__all__ = [
    "CorpusStats",
    "SAMPLING_FORMAT_VERSION",
    "TextDataset",
    "deterministic_split",
    "measure_corpus",
    "validate_sampling_payload",
]
