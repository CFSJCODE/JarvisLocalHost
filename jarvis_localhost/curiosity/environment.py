"""A corpus of canonical chunks exposed as a navigable RL environment."""

from __future__ import annotations

import hashlib
import math
import random
import re
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import numpy as np

from jarvis_localhost.corpus.provenance import CanonicalChunk


class CorpusRunCancelled(RuntimeError):
    """Raised between environment transitions during cooperative shutdown."""


class CorpusAction(IntEnum):
    NEXT = 0
    PREVIOUS = 1
    RELATED = 2
    CROSS_DOCUMENT = 3
    HIGH_LOSS = 4
    KNOWN_CONNECTION = 5
    UNVISITED = 6


@dataclass(frozen=True)
class CorpusTransition:
    previous_chunk_id: str
    next_chunk_id: str
    action: int
    visits: int


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def hashed_corpus_vector(text: str, dimension: int) -> np.ndarray:
    """Stable feature hashing; unlike Python hash(), it survives restarts."""
    vector = np.zeros(dimension, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class CorpusEnv:
    """Deterministic MDP where states and transitions come only from PDFs."""

    def __init__(
        self,
        chunks: list[CanonicalChunk],
        *,
        text_dimension: int = 128,
        max_episode_steps: int = 128,
        seed: int = 1_337,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("CorpusEnv requires at least one canonical chunk")
        self.chunks = sorted(chunks, key=lambda item: (item.document_id, item.ordinal))
        self.text_dimension = max(16, int(text_dimension))
        self.observation_dim = self.text_dimension + 6
        self.action_count = len(CorpusAction)
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.random = random.Random(seed)
        self.cancel_event = cancel_event
        self.text_vectors = np.stack(
            [hashed_corpus_vector(chunk.text, self.text_dimension) for chunk in self.chunks]
        )
        self.visits = np.zeros(len(self.chunks), dtype=np.int64)
        self.lm_loss = np.zeros(len(self.chunks), dtype=np.float32)
        self.retrieval_uncertainty = np.zeros(len(self.chunks), dtype=np.float32)
        self._lm_loss_known = np.zeros(len(self.chunks), dtype=np.bool_)
        self._retrieval_uncertainty_known = np.zeros(
            len(self.chunks), dtype=np.bool_
        )
        self._index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        self._max_page = max(chunk.page for chunk in self.chunks)
        self._document_ids = {
            document_id: index
            for index, document_id in enumerate(sorted({chunk.document_id for chunk in self.chunks}))
        }
        self.current_index = 0
        self.episode_steps = 0

    @property
    def current_chunk(self) -> CanonicalChunk:
        return self.chunks[self.current_index]

    def set_learning_signals(
        self,
        *,
        lm_loss: Mapping[str, float] | None = None,
        retrieval_uncertainty: Mapping[str, float] | None = None,
    ) -> None:
        for chunk_id, value in (lm_loss or {}).items():
            if chunk_id in self._index_by_id:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("lm_loss values must be finite")
                index = self._index_by_id[chunk_id]
                self.lm_loss[index] = max(0.0, numeric)
                self._lm_loss_known[index] = True
        for chunk_id, value in (retrieval_uncertainty or {}).items():
            if chunk_id in self._index_by_id:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(
                        "retrieval_uncertainty values must be finite"
                    )
                index = self._index_by_id[chunk_id]
                self.retrieval_uncertainty[index] = max(0.0, numeric)
                self._retrieval_uncertainty_known[index] = True

    def state_dict(self) -> dict:
        """Return only safe primitive state for a corpus-bound checkpoint."""

        return {
            "format_version": 1,
            "visits": self.visits.tolist(),
            "lm_loss": self.lm_loss.tolist(),
            "retrieval_uncertainty": self.retrieval_uncertainty.tolist(),
            "lm_loss_known": self._lm_loss_known.tolist(),
            "retrieval_uncertainty_known": (
                self._retrieval_uncertainty_known.tolist()
            ),
            "current_index": self.current_index,
            "episode_steps": self.episode_steps,
            "random_state": self.random.getstate(),
        }

    def load_state_dict(self, payload: dict) -> None:
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError("unsupported environment state format")
        count = len(self.chunks)

        def array(name: str, dtype: np.dtype) -> np.ndarray:
            value = np.asarray(payload.get(name, []), dtype=dtype)
            if value.shape != (count,):
                raise ValueError(f"invalid environment state shape for {name}")
            return value

        visits = array("visits", np.int64)
        lm_loss = array("lm_loss", np.float32)
        retrieval = array("retrieval_uncertainty", np.float32)
        if (
            bool(np.any(visits < 0))
            or not bool(np.isfinite(lm_loss).all())
            or not bool(np.isfinite(retrieval).all())
            or bool(np.any(lm_loss < 0))
            or bool(np.any(retrieval < 0))
        ):
            raise ValueError("environment state contains invalid values")
        current_index = int(payload.get("current_index", 0))
        episode_steps = int(payload.get("episode_steps", 0))
        if not 0 <= current_index < count or episode_steps < 0:
            raise ValueError("environment position is invalid")
        self.visits = visits
        self.lm_loss = lm_loss
        self.retrieval_uncertainty = retrieval
        self._lm_loss_known = array("lm_loss_known", np.bool_)
        self._retrieval_uncertainty_known = array(
            "retrieval_uncertainty_known", np.bool_
        )
        self.current_index = current_index
        self.episode_steps = episode_steps
        self.random.setstate(payload["random_state"])

    def _check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CorpusRunCancelled("curiosity cycle cancelled")

    @staticmethod
    def _bounded(value: float) -> float:
        return value / (1.0 + abs(value))

    def observation(self, index: int | None = None) -> np.ndarray:
        index = self.current_index if index is None else int(index)
        chunk = self.chunks[index]
        document_count = max(1, len(self._document_ids) - 1)
        ordinal_scale = max(1, len(self.chunks) - 1)
        metadata = np.asarray(
            [
                self._document_ids[chunk.document_id] / document_count,
                chunk.page / max(1, self._max_page),
                chunk.ordinal / ordinal_scale,
                math.log1p(int(self.visits[index])) / 10.0,
                self._bounded(float(self.lm_loss[index])),
                self._bounded(float(self.retrieval_uncertainty[index])),
            ],
            dtype=np.float32,
        )
        return np.concatenate((self.text_vectors[index], metadata)).astype(np.float32)

    def reset(self, *, start_index: int | None = None) -> np.ndarray:
        self._check_cancelled()
        if start_index is None:
            minimum_visits = int(self.visits.min())
            candidates = np.flatnonzero(self.visits == minimum_visits).tolist()
            start_index = self.random.choice(candidates)
        if not 0 <= start_index < len(self.chunks):
            raise IndexError("start_index is outside the corpus")
        self.current_index = int(start_index)
        self.episode_steps = 0
        self.visits[self.current_index] += 1
        return self.observation()

    def _most_similar(self, *, different_document: bool = False) -> int:
        similarities = self.text_vectors @ self.text_vectors[self.current_index]
        similarities[self.current_index] = -np.inf
        if different_document:
            current_document = self.current_chunk.document_id
            for index, chunk in enumerate(self.chunks):
                if chunk.document_id == current_document:
                    similarities[index] = -np.inf
        if np.isneginf(similarities).all():
            return self.current_index
        return int(np.argmax(similarities))

    def _target_for_action(self, action: CorpusAction) -> int:
        if action is CorpusAction.NEXT:
            return min(len(self.chunks) - 1, self.current_index + 1)
        if action is CorpusAction.PREVIOUS:
            return max(0, self.current_index - 1)
        if action is CorpusAction.RELATED:
            return self._most_similar()
        if action is CorpusAction.CROSS_DOCUMENT:
            return self._most_similar(different_document=True)
        if action is CorpusAction.HIGH_LOSS:
            if bool(self._lm_loss_known.any()):
                scores = self.lm_loss.copy()
                scores[~self._lm_loss_known] = -np.inf
                return int(np.argmax(scores))
            minimum_visits = int(self.visits.min())
            return int(np.flatnonzero(self.visits == minimum_visits)[0])
        if action is CorpusAction.KNOWN_CONNECTION:
            similarities = self.text_vectors @ self.text_vectors[self.current_index]
            similarities[self.current_index] = -np.inf
            visited = self.visits > 0
            similarities[~visited] = -np.inf
            return (
                int(np.argmax(similarities))
                if not np.isneginf(similarities).all()
                else self._most_similar()
            )
        if action is CorpusAction.UNVISITED:
            minimum_visits = int(self.visits.min())
            candidates = np.flatnonzero(self.visits == minimum_visits)
            uncertainties = self.retrieval_uncertainty[candidates]
            return int(candidates[int(np.argmax(uncertainties))])
        raise ValueError(f"unsupported corpus action: {action}")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self._check_cancelled()
        try:
            corpus_action = CorpusAction(int(action))
        except ValueError as exc:
            raise ValueError(f"action must be between 0 and {self.action_count - 1}") from exc
        previous = self.current_index
        self.current_index = self._target_for_action(corpus_action)
        self.visits[self.current_index] += 1
        self.episode_steps += 1
        done = self.episode_steps >= self.max_episode_steps
        transition = CorpusTransition(
            previous_chunk_id=self.chunks[previous].chunk_id,
            next_chunk_id=self.current_chunk.chunk_id,
            action=int(corpus_action),
            visits=int(self.visits[self.current_index]),
        )
        # Curiosity is supplied by the ICM; there is deliberately no manually
        # encoded domain reward in the environment.
        return self.observation(), 0.0, done, {"transition": transition}


__all__ = [
    "CorpusAction",
    "CorpusEnv",
    "CorpusRunCancelled",
    "CorpusTransition",
    "hashed_corpus_vector",
]
