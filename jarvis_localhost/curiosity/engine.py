"""Corpus-bound autonomous exploration using a genuine ICM + PPO loop."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from jarvis_localhost.corpus.manifest import load_authorized_chunks
from jarvis_localhost.corpus.provenance import CanonicalChunk, corpus_sha256
from jarvis_localhost.curiosity.curriculum import CurriculumManager
from jarvis_localhost.curiosity.environment import (
    CorpusEnv,
    CorpusRunCancelled,
    hashed_corpus_vector,
)
from jarvis_localhost.curiosity.ppo import CuriosityAgent, PPOConfig
from jarvis_localhost.paths import CORPUS_ROOT, CURIOSITY_ROOT


STATE_FORMAT_VERSION = 2
CHECKPOINT_FORMAT_VERSION = 2
MANIFEST_FORMAT_VERSION = 2
VECTOR_DIMENSION = 128
MAX_JSON_STATE_BYTES = 32 * 1024 * 1024
MAX_CORPUS_JSON_STATE_BYTES = 256 * 1024 * 1024
CORPUS_JSON_HEADER_BYTES = 1024 * 1024
CORPUS_JSON_BYTES_PER_CHUNK = {
    "curriculum.json": 512,
    "sampling_weights.json": 256,
}
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def _json_state_size_limit(filename: str, chunk_count: int) -> int:
    """Bound per-chunk state using the validated corpus, never file metadata.

    Pretty-printed signals and sampling arrays grow with the corpus. Other
    JSON artifacts keep the fixed limit, and even large corpora cannot raise
    a single JSON file's budget above 256 MiB.
    """
    per_chunk = CORPUS_JSON_BYTES_PER_CHUNK.get(filename)
    if per_chunk is None:
        return MAX_JSON_STATE_BYTES
    return min(
        MAX_CORPUS_JSON_STATE_BYTES,
        max(MAX_JSON_STATE_BYTES, CORPUS_JSON_HEADER_BYTES + chunk_count * per_chunk),
    )


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
        if len(token) > 2
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _device_tree(value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(item, device) for item in value)
    return value


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            state[key] = _device_tree(value, device)


@dataclass
class Insight:
    id: str
    source: str
    chunk_text: str
    summary: str
    tags: list[str]
    curiosity_score: float
    novelty_score: float
    entropy_score: float
    surprise_score: float
    connections: list[str]
    timestamp: float
    times_surfaced: int = 0
    is_new: bool = True
    page: int = 0
    chunk_id: str = ""
    document_sha256: str = ""
    intrinsic_reward: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CuriosityState:
    total_analyses: int = 0
    insights_found: int = 0
    last_analysis_ts: float = 0.0
    topic_index: dict[str, list[str]] = field(default_factory=dict)
    corpus_sha256: str = ""
    last_metrics: dict = field(default_factory=dict)


class CuriosityLifecycle(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"


class CuriosityEngine:
    """Background explorer with resumable, corpus-isolated local state."""

    MAX_INSIGHTS = 200

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        on_insight: Callable[[Insight], None] | None = None,
        *,
        device: Any = "cpu",
        cycle_interval: float = 300.0,
        top_k: int = 5,
        seed: int = 1_337,
    ) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir else CORPUS_ROOT
        self.output_dir = Path(output_dir) if output_dir else CURIOSITY_ROOT
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_insight = on_insight
        self.device = device
        self.cycle_interval = max(0.05, float(cycle_interval))
        self.top_k = max(1, int(top_k))
        self.seed = int(seed)
        self.random = random.Random(self.seed)

        self.state = CuriosityState()
        self.insights: dict[str, Insight] = {}
        self._chunks: list[CanonicalChunk] = []
        self._agent: CuriosityAgent | None = None
        self._curriculum: CurriculumManager | None = None
        self._cycle = 0
        self._current_generation = ""
        self._namespace: Path | None = None

        self._seen_vectors: dict[str, np.ndarray] = {}
        self._vector_ids: tuple[str, ...] = ()
        self._vector_matrix = np.empty((0, VECTOR_DIMENSION), dtype=np.float32)
        self._insight_id_by_chunk: dict[str, str] = {}

        self._lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = CuriosityLifecycle.STOPPED
        # Kept for compatibility with the former engine and external callers.
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Start one worker; never replace a thread that is still alive."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = None
            self._stop_event.clear()
            self._lifecycle = CuriosityLifecycle.RUNNING
            self._running = True
            thread = threading.Thread(
                target=self._loop,
                name="jarvis-curiosity",
                daemon=False,
            )
            self._thread = thread
            thread.start()
            return True

    def stop(self, timeout: float = 30.0) -> bool:
        """Request cooperative cancellation and report whether shutdown ended."""

        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = None
                self._lifecycle = CuriosityLifecycle.STOPPED
                self._running = False
                self._stop_event.set()
                return True
            self._lifecycle = CuriosityLifecycle.STOPPING
            self._running = False
            self._stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lock:
            stopped = not thread.is_alive()
            if stopped:
                self._thread = None
                self._lifecycle = CuriosityLifecycle.STOPPED
                self._running = False
            return stopped

    def _loop(self) -> None:
        try:
            while not self._stop_event.wait(self.cycle_interval):
                try:
                    self.run_cycle()
                except Exception as exc:  # keep the long-lived worker observable
                    with self._lock:
                        self.state.last_metrics = {
                            "algorithm": "ICM+PPO",
                            "error": str(exc),
                        }
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                self._lifecycle = CuriosityLifecycle.STOPPED
                self._running = False

    def _prepare_direct_cycle(self) -> bool:
        """Clear a stale stop only when no background worker can still own it."""

        with self._lock:
            if self._lifecycle is CuriosityLifecycle.STOPPING:
                return False
            if self._lifecycle is CuriosityLifecycle.STOPPED and not (
                self._thread and self._thread.is_alive()
            ):
                self._stop_event.clear()
            return True

    # -- corpus and agent activation -------------------------------------

    def _load_chunks(self) -> list[CanonicalChunk]:
        return sorted(
            load_authorized_chunks(self.corpus_dir),
            key=lambda chunk: (chunk.document_id, chunk.ordinal, chunk.chunk_id),
        )

    def _namespace_for(self, digest: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid corpus SHA-256")
        return self.output_dir / "corpora" / digest

    def _reset_for_corpus(
        self, chunks: list[CanonicalChunk], digest: str
    ) -> None:
        self._chunks = list(chunks)
        self.state = CuriosityState(corpus_sha256=digest)
        self.insights = {}
        self._agent = None
        self._curriculum = CurriculumManager(chunks, corpus_sha256=digest)
        self._cycle = 0
        self._current_generation = ""
        self._namespace = self._namespace_for(digest)
        self._seen_vectors = {}
        self._refresh_vector_cache()

    def _new_agent(self, chunks: list[CanonicalChunk]) -> CuriosityAgent:
        environment = CorpusEnv(
            chunks,
            text_dimension=VECTOR_DIMENSION,
            max_episode_steps=min(128, max(16, len(chunks) * 2)),
            seed=self.seed,
            cancel_event=self._stop_event,
        )
        config = PPOConfig(
            rollout_steps=min(96, max(16, len(chunks) * 2)),
            update_epochs=2,
            minibatch_size=min(32, max(4, len(chunks))),
        )
        return CuriosityAgent(
            environment,
            feature_dim=96,
            device=self.device,
            config=config,
            seed=self.seed,
        )

    @staticmethod
    def _shape_manifest(module: torch.nn.Module) -> dict[str, list[int]]:
        return {
            name: list(tensor.shape)
            for name, tensor in sorted(module.state_dict().items())
        }

    def _checkpoint_binding(
        self,
        agent: CuriosityAgent,
        chunks: Sequence[CanonicalChunk],
        digest: str,
    ) -> dict:
        chunk_digest = hashlib.sha256(
            "\n".join(chunk.chunk_id for chunk in chunks).encode("ascii")
        ).hexdigest()
        environment = agent.environment
        return {
            "schema_version": CHECKPOINT_FORMAT_VERSION,
            "corpus_sha256": digest,
            "chunk_count": len(chunks),
            "chunk_ids_sha256": chunk_digest,
            "seed": self.seed,
            "engine_config": {
                "algorithm": "ICM+PPO",
                "top_k": self.top_k,
                "max_insights": int(self.MAX_INSIGHTS),
                "vector_dimension": VECTOR_DIMENSION,
            },
            "environment": {
                "text_dimension": environment.text_dimension,
                "observation_dim": environment.observation_dim,
                "action_count": environment.action_count,
                "max_episode_steps": environment.max_episode_steps,
            },
            "ppo_config": asdict(agent.config),
            "icm_architecture": self._shape_manifest(agent.icm),
            "policy_architecture": self._shape_manifest(agent.policy),
        }

    def _ensure_agent(self, chunks: list[CanonicalChunk]) -> None:
        digest = corpus_sha256(chunks)
        if (
            self._agent is not None
            and digest == self.state.corpus_sha256
            and [chunk.chunk_id for chunk in chunks]
            == [chunk.chunk_id for chunk in self._chunks]
        ):
            return

        candidate = self._new_agent(chunks)
        curriculum = CurriculumManager(chunks, corpus_sha256=digest)
        restored, load_error = self._load_generation(
            candidate, curriculum, chunks, digest
        )
        if restored is None and load_error:
            # Loading is transactional at the object level as well as on disk:
            # discard a candidate that may have accepted an early state dict
            # before a later optimizer/environment validation failed.
            candidate = self._new_agent(chunks)
            curriculum = CurriculumManager(chunks, corpus_sha256=digest)
        with self._lock:
            self._chunks = list(chunks)
            self._agent = candidate
            self._curriculum = curriculum
            self._namespace = self._namespace_for(digest)
            if restored is None:
                self.state = CuriosityState(corpus_sha256=digest)
                if load_error:
                    self.state.last_metrics = {
                        "algorithm": "ICM+PPO",
                        "checkpoint_load_error": load_error,
                    }
                self.insights = {}
                self._cycle = 0
                self._current_generation = ""
            else:
                self.state = restored["state"]
                self.insights = restored["insights"]
                self._cycle = restored["cycle"]
                self._current_generation = restored["generation"]
                self.random.setstate(restored["engine_random_state"])
            self._prune_insights()

    def _read_json(self, path: Path, *, chunk_count: int = 0) -> Any:
        limit = _json_state_size_limit(path.name, chunk_count)
        if path.stat().st_size > limit:
            raise ValueError(f"JSON state exceeds the safe size limit ({limit} bytes): {path.name}")
        # Bound the read as well, in case the file grows after the stat check.
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise ValueError(f"JSON state exceeds the safe size limit ({limit} bytes): {path.name}")
        return json.loads(payload.decode("utf-8"))

    def _load_generation(
        self,
        agent: CuriosityAgent,
        curriculum: CurriculumManager,
        chunks: list[CanonicalChunk],
        digest: str,
    ) -> tuple[dict | None, str | None]:
        namespace = self._namespace_for(digest)
        current_path = namespace / "CURRENT"
        if not current_path.is_file():
            return None, None
        try:
            generation = current_path.read_text(encoding="utf-8").strip()
            if not re.fullmatch(r"gen_[0-9a-f_]+", generation):
                raise ValueError("invalid CURRENT generation identifier")
            directory = namespace / "generations" / generation
            manifest = self._read_json(directory / "manifest.json")
            if int(manifest.get("format_version", 0)) != MANIFEST_FORMAT_VERSION:
                raise ValueError("unsupported curiosity manifest")
            if manifest.get("generation") != generation:
                raise ValueError("generation manifest does not match CURRENT")
            if manifest.get("corpus_sha256") != digest:
                raise ValueError("generation belongs to another corpus")
            binding = self._checkpoint_binding(agent, chunks, digest)
            if manifest.get("binding") != binding:
                raise ValueError("checkpoint architecture/configuration mismatch")

            required = {
                "state.json",
                "insights.json",
                "curriculum.json",
                "sampling_weights.json",
                "agent.pt",
            }
            file_manifest = manifest.get("files")
            if not isinstance(file_manifest, dict) or not required.issubset(
                file_manifest
            ):
                raise ValueError("generation manifest is incomplete")
            for filename in required:
                path = directory / filename
                record = file_manifest[filename]
                size_limit = (
                    MAX_CHECKPOINT_BYTES
                    if filename == "agent.pt"
                    else _json_state_size_limit(filename, len(chunks))
                )
                if not path.is_file():
                    raise ValueError(f"missing state artifact: {filename}")
                actual_size = path.stat().st_size
                if actual_size > size_limit:
                    raise ValueError(
                        f"state size limit exceeded for {filename}: "
                        f"{actual_size} bytes > {size_limit} bytes"
                    )
                if (
                    int(record.get("bytes", -1)) != actual_size
                    or record.get("sha256") != _sha256_file(path)
                ):
                    raise ValueError(f"checksum failed for {filename}")

            state_payload = self._read_json(directory / "state.json")
            if (
                int(state_payload.get("format_version", 0))
                != STATE_FORMAT_VERSION
                or state_payload.get("corpus_sha256") != digest
            ):
                raise ValueError("state is not bound to the active corpus")
            raw_state = state_payload.get("state")
            if not isinstance(raw_state, dict):
                raise ValueError("invalid curiosity state")
            allowed_state = set(CuriosityState.__dataclass_fields__)
            state = CuriosityState(
                **{key: value for key, value in raw_state.items() if key in allowed_state}
            )
            if state.corpus_sha256 != digest:
                raise ValueError("curiosity state corpus mismatch")
            cycle = max(0, int(state_payload.get("cycle", 0)))

            chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            insight_payload = self._read_json(directory / "insights.json")
            if (
                int(insight_payload.get("format_version", 0))
                != STATE_FORMAT_VERSION
                or insight_payload.get("corpus_sha256") != digest
            ):
                raise ValueError("insights are not bound to the active corpus")
            insights: dict[str, Insight] = {}
            allowed_insight = set(Insight.__dataclass_fields__)
            for raw in insight_payload.get("insights", []):
                insight = Insight(
                    **{
                        key: value
                        for key, value in raw.items()
                        if key in allowed_insight
                    }
                )
                chunk = chunk_by_id.get(insight.chunk_id)
                if chunk is None or insight.document_sha256 != chunk.document_sha256:
                    raise ValueError("insight references a chunk outside its corpus")
                if insight.id in insights:
                    raise ValueError("duplicate insight identifier")
                insight.is_new = False
                insights[insight.id] = insight

            curriculum.load_payload(
                self._read_json(directory / "curriculum.json", chunk_count=len(chunks))
            )
            sampling = self._read_json(
                directory / "sampling_weights.json", chunk_count=len(chunks)
            )
            expected_sampling = curriculum.sampling_payload(
                [chunk.chunk_id for chunk in chunks]
            )
            if sampling != expected_sampling:
                raise ValueError("sampling weights do not match curriculum state")

            checkpoint = torch.load(
                directory / "agent.pt",
                map_location="cpu",
                weights_only=True,
            )
            if (
                int(checkpoint.get("format_version", 0))
                != CHECKPOINT_FORMAT_VERSION
                or checkpoint.get("binding") != binding
            ):
                raise ValueError("agent checkpoint binding mismatch")
            agent.icm.load_state_dict(checkpoint["icm_state_dict"], strict=True)
            agent.policy.load_state_dict(
                checkpoint["policy_state_dict"], strict=True
            )
            agent.icm_optimizer.load_state_dict(
                checkpoint["icm_optimizer_state_dict"]
            )
            agent.policy_optimizer.load_state_dict(
                checkpoint["policy_optimizer_state_dict"]
            )
            _move_optimizer_state(agent.icm_optimizer, self.device)
            _move_optimizer_state(agent.policy_optimizer, self.device)
            agent.environment.load_state_dict(checkpoint["environment_state"])

            engine_random_state = checkpoint["engine_random_state"]
            validator = random.Random()
            validator.setstate(engine_random_state)
            torch_rng_state = checkpoint["torch_rng_state"]
            if not isinstance(torch_rng_state, torch.Tensor):
                raise ValueError("invalid PyTorch RNG state")
            torch.set_rng_state(torch_rng_state.cpu())
            return {
                "state": state,
                "insights": insights,
                "cycle": cycle,
                "generation": generation,
                "engine_random_state": engine_random_state,
            }, None
        except Exception as exc:
            # weights_only=True plus checksum validation makes falling back to
            # a fresh random agent safe when a generation is partial/corrupt.
            return None, f"{type(exc).__name__}: {exc}"

    # -- signals and curriculum ------------------------------------------

    def _validate_signal_map(
        self,
        values: Mapping[str, float] | None,
        *,
        name: str,
    ) -> dict[str, float]:
        active = {chunk.chunk_id for chunk in self._chunks}
        cleaned: dict[str, float] = {}
        for chunk_id, value in (values or {}).items():
            if chunk_id not in active:
                raise KeyError(f"{name} references a chunk outside the active corpus")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"{name} values must be finite and non-negative")
            cleaned[chunk_id] = numeric
        return cleaned

    def _apply_learning_signals(
        self,
        lm_loss_by_chunk: Mapping[str, float] | None,
        retrieval_uncertainty_by_chunk: Mapping[str, float] | None,
    ) -> None:
        assert self._agent is not None and self._curriculum is not None
        losses = self._validate_signal_map(lm_loss_by_chunk, name="lm_loss")
        uncertainties = self._validate_signal_map(
            retrieval_uncertainty_by_chunk,
            name="retrieval_uncertainty",
        )
        self._agent.environment.set_learning_signals(
            lm_loss=losses,
            retrieval_uncertainty=uncertainties,
        )
        for chunk_id in sorted(set(losses) | set(uncertainties)):
            self._curriculum.update(
                chunk_id,
                lm_loss=losses.get(chunk_id),
                retrieval_uncertainty=uncertainties.get(chunk_id),
            )

    def get_sampling_weights(
        self, chunk_ids: Sequence[str] | None = None
    ) -> dict:
        """Return a versioned corpus-bound payload a trainer can consume."""

        with self._cycle_lock:
            if not self._prepare_direct_cycle():
                raise RuntimeError("curiosity engine is stopping")
            chunks = self._load_chunks()
            digest = corpus_sha256(chunks)
            if len(chunks) < 2:
                if digest != self.state.corpus_sha256:
                    self._reset_for_corpus(chunks, digest)
            else:
                self._ensure_agent(chunks)
            assert self._curriculum is not None
            identifiers = (
                list(chunk_ids)
                if chunk_ids is not None
                else [chunk.chunk_id for chunk in chunks]
            )
            return self._curriculum.sampling_payload(identifiers)

    def update_learning_signals(
        self,
        *,
        lm_loss_by_chunk: Mapping[str, float] | None = None,
        retrieval_uncertainty_by_chunk: Mapping[str, float] | None = None,
    ) -> dict:
        """Accept measured per-chunk signals and persist new sampling weights."""

        with self._cycle_lock:
            if not self._prepare_direct_cycle():
                raise RuntimeError("curiosity engine is stopping")
            chunks = self._load_chunks()
            if len(chunks) < 2:
                raise RuntimeError("at least two canonical chunks are required")
            self._ensure_agent(chunks)
            self._apply_learning_signals(
                lm_loss_by_chunk, retrieval_uncertainty_by_chunk
            )
            generation = self._publish_generation({"signals_only": True})
            payload = self._curriculum.sampling_payload(
                [chunk.chunk_id for chunk in chunks]
            )
            payload["generation"] = generation
            return payload

    # -- analysis ---------------------------------------------------------

    @staticmethod
    def _document_frequency(
        chunks: list[CanonicalChunk],
    ) -> tuple[Counter[str], float]:
        frequencies: Counter[str] = Counter()
        for chunk in chunks:
            frequencies.update(set(_tokens(chunk.text)))
        total = max(1, len(chunks))
        max_idf = max(
            (
                math.log((total + 1) / (count + 1)) + 1.0
                for count in frequencies.values()
            ),
            default=1.0,
        )
        return frequencies, max_idf

    def _scores(
        self,
        chunk: CanonicalChunk,
        document_frequency: Counter[str],
        max_idf: float,
    ) -> tuple[float, float, float, list[str]]:
        words = _tokens(chunk.text)
        counts = Counter(words)
        total = max(1, len(words))
        probabilities = [count / total for count in counts.values()]
        entropy = -sum(
            probability * math.log(probability + 1e-12)
            for probability in probabilities
        )
        entropy /= max(1e-12, math.log(max(2, len(counts))))
        idf_by_term = {
            term: math.log(
                (len(self._chunks) + 1) / (document_frequency[term] + 1)
            )
            + 1.0
            for term in counts
        }
        surprise = (
            sum(idf_by_term[term] * frequency for term, frequency in counts.items())
            / total
            / max_idf
            if counts
            else 0.0
        )
        vector = hashed_corpus_vector(chunk.text, VECTOR_DIMENSION)
        if self._vector_matrix.shape[0]:
            cosine = float(np.max(self._vector_matrix @ vector))
            novelty = max(0.0, min(1.0, 1.0 - cosine))
        else:
            novelty = 1.0
        tags = [
            term
            for term, _ in sorted(
                idf_by_term.items(),
                key=lambda item: (-(item[1] * counts[item[0]]), item[0]),
            )[:5]
        ]
        return (
            novelty,
            max(0.0, min(1.0, entropy)),
            max(0.0, min(1.0, surprise)),
            tags,
        )

    @staticmethod
    def _summary(chunk: CanonicalChunk) -> str:
        words = chunk.text.split()
        excerpt = " ".join(words[:60])
        if len(words) > 60:
            excerpt += "…"
        return (
            f"{excerpt} "
            f"[{chunk.filename} — página {chunk.page} — {chunk.chunk_id}]"
        )

    def _connections(self, chunk: CanonicalChunk) -> list[str]:
        if not self._vector_ids:
            return []
        target = hashed_corpus_vector(chunk.text, VECTOR_DIMENSION)
        similarities = self._vector_matrix @ target
        candidates: list[tuple[float, str]] = []
        for index, chunk_id in enumerate(self._vector_ids):
            if chunk_id == chunk.chunk_id:
                continue
            similarity = float(similarities[index])
            insight_id = self._insight_id_by_chunk.get(chunk_id)
            if insight_id and similarity > 0.35:
                candidates.append((similarity, insight_id))
        return [
            identifier
            for _, identifier in sorted(candidates, reverse=True)[:5]
        ]

    def _refresh_vector_cache(self) -> None:
        self._vector_ids = tuple(sorted(self._seen_vectors))
        self._vector_matrix = (
            np.stack([self._seen_vectors[key] for key in self._vector_ids])
            if self._vector_ids
            else np.empty((0, VECTOR_DIMENSION), dtype=np.float32)
        )
        self._insight_id_by_chunk = {
            insight.chunk_id: insight.id for insight in self.insights.values()
        }

    def _rebuild_topic_index(self) -> None:
        topic_index: dict[str, list[str]] = {}
        for insight in sorted(self.insights.values(), key=lambda item: item.id):
            for tag in dict.fromkeys(insight.tags):
                topic_index.setdefault(tag, []).append(insight.id)
        self.state.topic_index = topic_index

    def _prune_insights(self) -> None:
        limit = max(1, int(self.MAX_INSIGHTS))
        keep = sorted(
            self.insights.values(),
            key=lambda item: (-item.curiosity_score, -item.timestamp, item.id),
        )[:limit]
        self.insights = {insight.id: insight for insight in keep}
        valid_ids = set(self.insights)
        for insight in self.insights.values():
            insight.connections = [
                identifier
                for identifier in dict.fromkeys(insight.connections)
                if identifier in valid_ids and identifier != insight.id
            ][:5]
        chunk_text = {chunk.chunk_id: chunk.text for chunk in self._chunks}
        self._seen_vectors = {
            insight.chunk_id: hashed_corpus_vector(
                chunk_text.get(insight.chunk_id, insight.chunk_text),
                VECTOR_DIMENSION,
            )
            for insight in self.insights.values()
        }
        self._refresh_vector_cache()
        self._rebuild_topic_index()
        self.state.insights_found = len(self.insights)

    def run_cycle(
        self,
        *,
        lm_loss_by_chunk: Mapping[str, float] | None = None,
        retrieval_uncertainty_by_chunk: Mapping[str, float] | None = None,
    ) -> dict:
        callbacks: list[Insight] = []
        with self._cycle_lock:
            if not self._prepare_direct_cycle():
                return {
                    "algorithm": "ICM+PPO",
                    "cancelled": True,
                    "reason": "engine is stopping",
                }
            metrics, callbacks = self._run_cycle_locked(
                lm_loss_by_chunk=lm_loss_by_chunk,
                retrieval_uncertainty_by_chunk=retrieval_uncertainty_by_chunk,
            )
        if self.on_insight:
            for insight in callbacks:
                try:
                    self.on_insight(insight)
                except Exception:
                    # Persistence and training must not be rolled back by UI hooks.
                    continue
        return metrics

    def _run_cycle_locked(
        self,
        *,
        lm_loss_by_chunk: Mapping[str, float] | None,
        retrieval_uncertainty_by_chunk: Mapping[str, float] | None,
    ) -> tuple[dict, list[Insight]]:
        chunks = self._load_chunks()
        digest = corpus_sha256(chunks)
        if len(chunks) < 2:
            if digest != self.state.corpus_sha256:
                self._reset_for_corpus(chunks, digest)
            metrics = {
                "algorithm": "ICM+PPO",
                "skipped": True,
                "reason": "at least two canonical chunks are required",
                "chunks": len(chunks),
                "corpus_sha256": digest,
            }
            with self._lock:
                self.state.last_metrics = metrics
            return metrics, []

        self._ensure_agent(chunks)
        assert self._agent is not None and self._curriculum is not None
        self._apply_learning_signals(
            lm_loss_by_chunk, retrieval_uncertainty_by_chunk
        )
        try:
            cycle = self._agent.train_cycle()
        except CorpusRunCancelled:
            metrics = {
                "algorithm": "ICM+PPO",
                "cancelled": True,
                "corpus_sha256": digest,
            }
            with self._lock:
                self.state.last_metrics = metrics
            return metrics, []
        if self._stop_event.is_set() and (
            self._lifecycle is CuriosityLifecycle.STOPPING
        ):
            metrics = {
                "algorithm": "ICM+PPO",
                "cancelled": True,
                "corpus_sha256": digest,
            }
            with self._lock:
                self.state.last_metrics = metrics
            return metrics, []

        rewards_by_chunk: dict[str, float] = {}
        for record in cycle.records:
            rewards_by_chunk[record.next_chunk_id] = max(
                rewards_by_chunk.get(record.next_chunk_id, 0.0), record.reward
            )
            self._curriculum.update(
                record.next_chunk_id,
                intrinsic_reward=record.reward,
                visit_increment=1,
            )

        document_frequency, max_idf = self._document_frequency(chunks)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ranked = [
            signal
            for signal in self._curriculum.ranked()
            if signal.chunk_id in rewards_by_chunk
        ][: self.top_k]
        now = time.time()
        new_insights: list[Insight] = []
        for signal in ranked:
            chunk = chunk_by_id[signal.chunk_id]
            novelty, entropy, surprise, tags = self._scores(
                chunk, document_frequency, max_idf
            )
            insight_id = f"ins_{chunk.chunk_id.removeprefix('chk_')}"
            previous = self.insights.get(insight_id)
            insight = Insight(
                id=insight_id,
                source=chunk.filename,
                chunk_text=chunk.text[:800],
                summary=self._summary(chunk),
                tags=tags,
                curiosity_score=round(signal.priority, 6),
                novelty_score=round(novelty, 6),
                entropy_score=round(entropy, 6),
                surprise_score=round(surprise, 6),
                connections=self._connections(chunk),
                timestamp=now,
                times_surfaced=previous.times_surfaced if previous else 0,
                is_new=previous is None,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                document_sha256=chunk.document_sha256,
                intrinsic_reward=round(rewards_by_chunk[chunk.chunk_id], 8),
            )
            self.insights[insight_id] = insight
            self._seen_vectors[chunk.chunk_id] = hashed_corpus_vector(
                chunk.text, VECTOR_DIMENSION
            )
            self._refresh_vector_cache()
            if previous is None:
                new_insights.append(insight)

        self._cycle += 1
        self.state.total_analyses += len(cycle.records)
        self.state.last_analysis_ts = now
        self._prune_insights()
        self.state.last_metrics = {
            **cycle.metrics,
            "chunks": len(chunks),
            "new_insights": len(new_insights),
            "corpus_sha256": digest,
            "curriculum_signal_version": self._curriculum.signal_version,
        }
        generation = self._publish_generation(cycle.metrics)
        self.state.last_metrics["generation"] = generation
        return dict(self.state.last_metrics), new_insights

    # -- generation persistence ------------------------------------------

    def _checkpoint_payload(self) -> dict:
        assert self._agent is not None
        binding = self._checkpoint_binding(
            self._agent, self._chunks, self.state.corpus_sha256
        )
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "binding": binding,
            "icm_state_dict": _cpu_tree(self._agent.icm.state_dict()),
            "policy_state_dict": _cpu_tree(self._agent.policy.state_dict()),
            "icm_optimizer_state_dict": _cpu_tree(
                self._agent.icm_optimizer.state_dict()
            ),
            "policy_optimizer_state_dict": _cpu_tree(
                self._agent.policy_optimizer.state_dict()
            ),
            "environment_state": self._agent.environment.state_dict(),
            "engine_random_state": self.random.getstate(),
            "torch_rng_state": torch.get_rng_state().cpu(),
        }

    def _publish_generation(self, training_metrics: dict) -> str:
        assert self._agent is not None and self._curriculum is not None
        digest = self.state.corpus_sha256
        namespace = self._namespace_for(digest)
        generations = namespace / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        generation = f"gen_{self._cycle:012x}_{time.time_ns():x}_{token[:12]}"
        staging = generations / f".staging_{token}"
        destination = generations / generation
        staging.mkdir()
        try:
            state_payload = {
                "format_version": STATE_FORMAT_VERSION,
                "corpus_sha256": digest,
                "cycle": self._cycle,
                "state": asdict(self.state),
            }
            insight_payload = {
                "format_version": STATE_FORMAT_VERSION,
                "corpus_sha256": digest,
                "insights": [
                    insight.to_dict()
                    for insight in sorted(
                        self.insights.values(),
                        key=lambda item: (-item.curiosity_score, item.id),
                    )
                ],
            }
            sampling = self._curriculum.sampling_payload(
                [chunk.chunk_id for chunk in self._chunks]
            )
            _write_json(staging / "state.json", state_payload)
            _write_json(staging / "insights.json", insight_payload)
            _write_json(
                staging / "curriculum.json", self._curriculum.to_payload()
            )
            _write_json(staging / "sampling_weights.json", sampling)
            torch.save(self._checkpoint_payload(), staging / "agent.pt")

            filenames = (
                "state.json",
                "insights.json",
                "curriculum.json",
                "sampling_weights.json",
                "agent.pt",
            )
            file_manifest = {
                filename: {
                    "sha256": _sha256_file(staging / filename),
                    "bytes": (staging / filename).stat().st_size,
                }
                for filename in filenames
            }
            binding = self._checkpoint_binding(self._agent, self._chunks, digest)
            manifest = {
                "format_version": MANIFEST_FORMAT_VERSION,
                "generation": generation,
                "created_at": time.time(),
                "corpus_sha256": digest,
                "binding": binding,
                "initialized_from": "random",
                "external_weights": False,
                "training": training_metrics,
                "files": file_manifest,
            }
            _write_json(staging / "manifest.json", manifest)
            os.replace(staging, destination)

            current_temporary = namespace / f".CURRENT_{token}.tmp"
            current_temporary.write_text(generation + "\n", encoding="utf-8")
            os.replace(current_temporary, namespace / "CURRENT")
            self._current_generation = generation
            self._write_legacy_checkpoint_alias(destination, manifest)
            return generation
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _write_legacy_checkpoint_alias(
        self, generation_directory: Path, manifest: dict
    ) -> None:
        """Keep the former checkpoint path as a non-authoritative alias."""

        source = generation_directory / "agent.pt"
        destination = self.output_dir / "icm_ppo.pt"
        temporary = self.output_dir / f".icm_ppo_{uuid.uuid4().hex}.tmp"
        # Copy instead of hard-linking: writes through the compatibility path
        # must never mutate an immutable generation file.
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        legacy_manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": "jarvis-curiosity-icm-ppo",
            "initialized_from": "random",
            "external_weights": False,
            "corpus_sha256": self.state.corpus_sha256,
            "training": manifest.get("training", {}),
            "seed": self.seed,
            "generation": manifest["generation"],
            "authoritative_path": str(source.relative_to(self.output_dir)),
        }
        manifest_path = self.output_dir / "icm_ppo.pt.manifest.json"
        manifest_temporary = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )
        _write_json(manifest_temporary, legacy_manifest)
        os.replace(manifest_temporary, manifest_path)

    # -- read API ---------------------------------------------------------

    def get_top_insights(
        self, n: int = 10, tag: str | None = None
    ) -> list[Insight]:
        with self._lock:
            pool = list(self.insights.values())
            if tag:
                pool = [insight for insight in pool if tag in insight.tags]
            return sorted(
                pool, key=lambda item: (-item.curiosity_score, item.id)
            )[: max(0, int(n))]

    def get_topics(self) -> dict[str, int]:
        with self._lock:
            return {
                tag: len(identifiers)
                for tag, identifiers in self.state.topic_index.items()
            }

    def get_random_insight(self) -> Insight | None:
        with self._lock:
            if not self.insights:
                return None
            minimum = min(
                insight.times_surfaced for insight in self.insights.values()
            )
            pool = [
                insight
                for insight in self.insights.values()
                if insight.times_surfaced == minimum
            ]
            chosen = self.random.choice(pool)
            chosen.times_surfaced += 1
            chosen.is_new = False
            return chosen

    def search_insights(self, query: str) -> list[Insight]:
        query_terms = set(_tokens(query))
        with self._lock:
            ranked: list[tuple[int, Insight]] = []
            for insight in self.insights.values():
                haystack = set(
                    _tokens(
                        " ".join(
                            (insight.chunk_text, insight.summary, *insight.tags)
                        )
                    )
                )
                overlap = len(query_terms & haystack)
                if overlap:
                    ranked.append((overlap, insight))
            return [
                insight
                for _, insight in sorted(
                    ranked, key=lambda item: (-item[0], item[1].id)
                )[:10]
            ]

    def get_stats(self) -> dict:
        with self._lock:
            topics = {
                tag: len(identifiers)
                for tag, identifiers in self.state.topic_index.items()
            }
            return {
                "algorithm": "ICM+PPO",
                "total_analyses": self.state.total_analyses,
                "insights_found": len(self.insights),
                "topics": len(topics),
                "cycle": self._cycle,
                "last_analysis": self.state.last_analysis_ts,
                "is_running": self._running,
                "lifecycle": self._lifecycle.value,
                "corpus_sha256": self.state.corpus_sha256,
                "generation": self._current_generation,
                "last_metrics": dict(self.state.last_metrics),
                "top_topics": sorted(
                    topics.items(), key=lambda item: (-item[1], item[0])
                )[:8],
            }


__all__ = [
    "CuriosityEngine",
    "CuriosityLifecycle",
    "CuriosityState",
    "Insight",
]
