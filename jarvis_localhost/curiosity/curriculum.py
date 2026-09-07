"""Corpus-bound curriculum priorities and trainer sampling weights."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from jarvis_localhost.corpus.provenance import CanonicalChunk


CURRICULUM_FORMAT_VERSION = 2
SAMPLING_FORMAT_VERSION = 1


@dataclass
class ChunkLearningSignal:
    chunk_id: str
    intrinsic_reward: float = 0.0
    lm_loss: float = 0.0
    retrieval_uncertainty: float = 0.0
    visits: int = 0
    priority: float = 0.0


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return max(0.0, result)


class CurriculumManager:
    """Maintain learning signals only for chunks in one immutable corpus."""

    def __init__(
        self,
        chunks: Iterable[CanonicalChunk],
        *,
        corpus_sha256: str = "",
        intrinsic_weight: float = 0.4,
        lm_loss_weight: float = 0.3,
        retrieval_weight: float = 0.2,
        visit_penalty: float = 0.1,
    ) -> None:
        self.weights = (
            float(intrinsic_weight),
            float(lm_loss_weight),
            float(retrieval_weight),
            float(visit_penalty),
        )
        if any(not math.isfinite(value) or value < 0 for value in self.weights):
            raise ValueError("curriculum weights must be finite and non-negative")
        self.corpus_sha256 = str(corpus_sha256)
        self.signals = {
            chunk.chunk_id: ChunkLearningSignal(chunk.chunk_id) for chunk in chunks
        }
        self.signal_version = 0

    @staticmethod
    def _bounded(value: float) -> float:
        return max(0.0, value) / (1.0 + max(0.0, value))

    def _recompute(self, signal: ChunkLearningSignal) -> None:
        wi, wl, wr, wv = self.weights
        signal.priority = max(
            0.0,
            wi * self._bounded(signal.intrinsic_reward)
            + wl * self._bounded(signal.lm_loss)
            + wr * self._bounded(signal.retrieval_uncertainty)
            - wv * math.log1p(signal.visits) / 10.0,
        )

    def update(
        self,
        chunk_id: str,
        *,
        intrinsic_reward: float | None = None,
        lm_loss: float | None = None,
        retrieval_uncertainty: float | None = None,
        visit_increment: int = 0,
    ) -> ChunkLearningSignal:
        if chunk_id not in self.signals:
            raise KeyError(f"chunk is not part of the active corpus: {chunk_id}")
        increment = int(visit_increment)
        if increment < 0:
            raise ValueError("visit_increment cannot be negative")
        signal = self.signals[chunk_id]
        changed = False
        if intrinsic_reward is not None:
            reward = _finite_nonnegative(
                intrinsic_reward, name="intrinsic_reward"
            )
            signal.intrinsic_reward = 0.8 * signal.intrinsic_reward + 0.2 * reward
            changed = True
        if lm_loss is not None:
            signal.lm_loss = _finite_nonnegative(lm_loss, name="lm_loss")
            changed = True
        if retrieval_uncertainty is not None:
            signal.retrieval_uncertainty = _finite_nonnegative(
                retrieval_uncertainty, name="retrieval_uncertainty"
            )
            changed = True
        if increment:
            signal.visits += increment
            changed = True
        if changed:
            self._recompute(signal)
            self.signal_version += 1
        return signal

    def ranked(self) -> list[ChunkLearningSignal]:
        return sorted(
            self.signals.values(), key=lambda item: (-item.priority, item.chunk_id)
        )

    def sampling_weights(self, chunk_ids: Sequence[str]) -> list[float]:
        identifiers = list(chunk_ids)
        if not identifiers:
            return []
        unknown = [chunk_id for chunk_id in identifiers if chunk_id not in self.signals]
        if unknown:
            raise KeyError(f"chunks are not part of the active corpus: {unknown[:3]}")
        raw = [max(0.0, self.signals[chunk_id].priority) for chunk_id in identifiers]
        total = sum(raw)
        if total <= 0.0:
            uniform = 1.0 / len(identifiers)
            return [uniform for _ in identifiers]
        return [value / total for value in raw]

    def sampling_payload(self, chunk_ids: Sequence[str] | None = None) -> dict:
        identifiers = list(chunk_ids) if chunk_ids is not None else sorted(self.signals)
        return {
            "format_version": SAMPLING_FORMAT_VERSION,
            "corpus_sha256": self.corpus_sha256,
            "signal_version": self.signal_version,
            "chunk_ids": identifiers,
            "weights": self.sampling_weights(identifiers),
        }

    def to_payload(self) -> dict:
        return {
            "format_version": CURRICULUM_FORMAT_VERSION,
            "corpus_sha256": self.corpus_sha256,
            "signal_version": self.signal_version,
            "weights": list(self.weights),
            "signals": [asdict(signal) for signal in self.ranked()],
        }

    def load_payload(self, data: dict) -> None:
        if int(data.get("format_version", 0)) != CURRICULUM_FORMAT_VERSION:
            raise ValueError("unsupported curriculum format")
        if str(data.get("corpus_sha256", "")) != self.corpus_sha256:
            raise ValueError("curriculum belongs to a different corpus")
        persisted_weights = tuple(float(value) for value in data.get("weights", ()))
        if persisted_weights != self.weights:
            raise ValueError("curriculum configuration does not match this runtime")
        restored: dict[str, ChunkLearningSignal] = {}
        for payload in data.get("signals", []):
            signal = ChunkLearningSignal(**payload)
            if signal.chunk_id not in self.signals:
                raise ValueError("curriculum references a chunk outside its corpus")
            if signal.chunk_id in restored:
                raise ValueError("curriculum contains a duplicate chunk")
            signal.intrinsic_reward = _finite_nonnegative(
                signal.intrinsic_reward, name="intrinsic_reward"
            )
            signal.lm_loss = _finite_nonnegative(signal.lm_loss, name="lm_loss")
            signal.retrieval_uncertainty = _finite_nonnegative(
                signal.retrieval_uncertainty, name="retrieval_uncertainty"
            )
            signal.visits = max(0, int(signal.visits))
            self._recompute(signal)
            restored[signal.chunk_id] = signal
        for chunk_id in self.signals:
            restored.setdefault(chunk_id, ChunkLearningSignal(chunk_id))
        self.signals = restored
        self.signal_version = max(0, int(data.get("signal_version", 0)))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if not source.is_file():
            return
        self.load_payload(json.loads(source.read_text(encoding="utf-8")))


__all__ = [
    "CURRICULUM_FORMAT_VERSION",
    "SAMPLING_FORMAT_VERSION",
    "ChunkLearningSignal",
    "CurriculumManager",
]
