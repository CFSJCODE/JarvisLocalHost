"""Self-supervised contrastive training for the sovereign retriever."""

from __future__ import annotations

import random
import math
import os
import tempfile
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from jarvis_localhost.ai.dataset import validate_sampling_payload
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import DirectMLAdamW, TrainingCancelled
from jarvis_localhost.corpus.provenance import CanonicalChunk, corpus_sha256
from jarvis_localhost.retrieval.encoder import RetrieverEncoder


@dataclass(frozen=True)
class ContrastivePair:
    anchor: str
    positive: str
    anchor_chunk_id: str
    positive_chunk_id: str


def _deterministic_views(text: str) -> tuple[str, str]:
    """Create two corpus-only surface views without semantic augmentation."""

    units = text.split()
    if len(units) < 4:
        compact = " ".join(units) or text
        return compact, compact
    trim = max(1, len(units) // 4)
    return " ".join(units[:-trim]), " ".join(units[trim:])


def _check_cancelled(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise TrainingCancelled("contrastive training cancelled by operator")


def build_positive_pairs(
    chunks: Iterable[CanonicalChunk],
    *,
    cancellation_event: threading.Event | None = None,
) -> list[ContrastivePair]:
    _check_cancelled(cancellation_event)
    ordered = sorted(chunks, key=lambda chunk: (chunk.document_id, chunk.ordinal))
    by_section: dict[tuple[str, str], list[CanonicalChunk]] = defaultdict(list)
    for chunk in ordered:
        _check_cancelled(cancellation_event)
        if chunk.section:
            by_section[(chunk.document_id, chunk.section)].append(chunk)
    pairs: list[ContrastivePair] = []
    seen: set[tuple[str, str]] = set()
    for index, anchor in enumerate(ordered):
        _check_cancelled(cancellation_event)
        # Every canonical chunk supplies two deterministic overlapping views.
        # This gives one positive per document even when PDFs are short and
        # contain only a single chunk. Other chunks remain in-batch negatives;
        # no labels, synonyms or domain assumptions are fabricated.
        anchor_view, positive_view = _deterministic_views(anchor.text)
        self_key = (anchor.chunk_id, anchor.chunk_id)
        if self_key not in seen:
            seen.add(self_key)
            pairs.append(
                ContrastivePair(
                    anchor_view,
                    positive_view,
                    anchor.chunk_id,
                    anchor.chunk_id,
                )
            )
        candidates: list[CanonicalChunk] = []
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(ordered):
                neighbor = ordered[neighbor_index]
                if neighbor.document_id == anchor.document_id:
                    candidates.append(neighbor)
        if anchor.section and len(candidates) < 2:
            # Preserve the original first-two-candidates rule, including a
            # repeated neighbour consuming a slot before pair deduplication.
            # Interior chunks already have two neighbours. Only document
            # boundaries need the ordered section index, never a corpus scan.
            for candidate in by_section[(anchor.document_id, anchor.section)]:
                if candidate.chunk_id != anchor.chunk_id:
                    candidates.append(candidate)
                    if len(candidates) == 2:
                        break
        for positive in candidates[:2]:
            key = (anchor.chunk_id, positive.chunk_id)
            if key not in seen:
                seen.add(key)
                pairs.append(
                    ContrastivePair(
                        anchor.text,
                        positive.text,
                        anchor.chunk_id,
                        positive.chunk_id,
                    )
                )
    return pairs


class ContrastiveDataset(Dataset):
    def __init__(self, pairs: list[ContrastivePair]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> ContrastivePair:
        return self.pairs[index]


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = anchor @ positive.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
    )


class ContrastiveTrainer:
    def __init__(
        self,
        encoder: RetrieverEncoder,
        tokenizer: JarvisTokenizer,
        chunks: list[CanonicalChunk],
        *,
        device: Any = "cpu",
        learning_rate: float = 2e-4,
        temperature: float = 0.07,
        seed: int = 1_337,
        sampling_payload: Mapping[str, Any] | None = None,
        padding_policy: str = "fixed",
        min_bucket_size: int = 32,
        fuse_pair_encoding: bool = False,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.execution_policy = {
            "format_version": 1,
            "padding_policy": padding_policy,
            "min_bucket_size": min_bucket_size,
            "fuse_pair_encoding": fuse_pair_encoding,
            "max_fused_sequence_length": 256,
        }
        self._validate_execution_policy(self.execution_policy)
        self.execution_policy_transitions: list[dict[str, Any]] = []
        self.encoder = encoder.to(device)
        self.tokenizer = tokenizer
        self.chunks = list(chunks)
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("contrastive corpus contains duplicate chunk IDs")
        self.corpus_sha256 = corpus_sha256(self.chunks)
        self.device = device
        self.temperature = temperature
        self.seed = seed
        self.learning_rate = learning_rate
        self.steps_completed = 0
        self.completed_epochs = 0
        self.resume_metadata: dict[str, Any] = {"resumed": False}
        self.uncertainty_metadata: dict[str, Any] = {}
        torch.manual_seed(seed)
        random.seed(seed)
        self.sampling_weights: dict[str, float] = {}
        self.sampling_metadata: dict[str, Any] = {"weighted": False}
        if sampling_payload is not None:
            self.sampling_weights, self.sampling_metadata = (
                validate_sampling_payload(
                    sampling_payload,
                    expected_corpus_sha256=self.corpus_sha256,
                    expected_chunk_ids=chunk_ids,
                )
            )
        self.optimizer = DirectMLAdamW(
            encoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-2,
        )

    def _encode(self, texts: list[str]) -> torch.Tensor:
        ids, mask = self.encoder.tokenize(
            self.tokenizer, texts, self.device,
            padding_policy=self.execution_policy["padding_policy"],
            min_bucket_size=self.execution_policy["min_bucket_size"],
        )
        return self.encoder(ids, mask)

    def _encode_pairs(
        self, anchors: list[str], positives: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Share a bounded 2B forward while retaining the B-by-B objective.

        Long sequences use separate forwards: on DirectML, joining 512-token
        pairs increased both elapsed time and peak memory in the physical test.
        """
        if len(anchors) != len(positives) or not anchors:
            raise ValueError("anchor and positive batches must be nonempty and equally sized")
        if not self.execution_policy["fuse_pair_encoding"]:
            return self._encode(anchors), self._encode(positives)
        ids, mask = self.encoder.tokenize(
            self.tokenizer, [*anchors, *positives], self.device,
            padding_policy=self.execution_policy["padding_policy"],
            min_bucket_size=self.execution_policy["min_bucket_size"],
        )
        if ids.shape[1] > self.execution_policy["max_fused_sequence_length"]:
            del ids, mask
            return self._encode(anchors), self._encode(positives)
        encoded = self.encoder(ids, mask)
        return encoded[:len(anchors)], encoded[len(anchors):]

    @staticmethod
    def _validate_execution_policy(policy: Any) -> None:
        if (
            not isinstance(policy, dict)
            or set(policy) != {
                "format_version", "padding_policy", "min_bucket_size",
                "fuse_pair_encoding", "max_fused_sequence_length",
            }
            or type(policy.get("format_version")) is not int or policy["format_version"] != 1
            or policy.get("padding_policy") not in {"fixed", "bucketed"}
            or type(policy.get("min_bucket_size")) is not int
            or policy["min_bucket_size"] not in {32, 64}
            or type(policy.get("fuse_pair_encoding")) is not bool
            or type(policy.get("max_fused_sequence_length")) is not int
            or policy["max_fused_sequence_length"] not in {256, 512}
        ):
            raise ValueError("retriever execution policy is invalid")

    @classmethod
    def _normalize_execution_policy(cls, policy: Any) -> dict[str, Any]:
        if not isinstance(policy, dict):
            raise ValueError("retriever execution policy is invalid")
        normalized = dict(policy)
        # Existing experimental fusion policies had no length guard. Preserve
        # that historical meaning so adopting the guard records a transition.
        # Constructors always use 256; 512 is accepted only as saved metadata.
        normalized.setdefault(
            "max_fused_sequence_length", 512 if normalized.get("fuse_pair_encoding") is True else 256,
        )
        cls._validate_execution_policy(normalized)
        return normalized

    def _restore_execution_metadata(self, state: dict[str, Any], steps: int) -> None:
        previous = self._normalize_execution_policy(state.get("execution_policy", {
            "format_version": 1, "padding_policy": "fixed",
            "min_bucket_size": 32, "fuse_pair_encoding": False,
        }))
        transitions = state.get("execution_policy_transitions", [])
        if not isinstance(transitions, list) or len(transitions) > steps + 1:
            raise ValueError("retriever execution policy transitions are invalid")
        normalized_transitions = []
        for entry in transitions:
            if (
                not isinstance(entry, dict) or type(entry.get("at_step")) is not int
                or not 0 <= entry["at_step"] <= steps
                or entry.get("bit_exact_transition") is not False
            ):
                raise ValueError("retriever execution policy transition is invalid")
            normalized_transitions.append({
                **entry,
                "from": self._normalize_execution_policy(entry.get("from")),
                "to": self._normalize_execution_policy(entry.get("to")),
            })
        self.execution_policy_transitions = normalized_transitions
        changed = previous != self.execution_policy
        self.resume_metadata["execution_policy_changed"] = changed
        self.resume_metadata["execution_policy"] = dict(self.execution_policy)
        if changed:
            # The same pairs, moments and RNG state are retained, but changed
            # padding/fusion consumes dropout randomness in a different layout.
            transition = {
                "from": dict(previous), "to": dict(self.execution_policy),
                "at_step": steps, "bit_exact_transition": False,
            }
            self.execution_policy_transitions.append(transition)
            self.resume_metadata["execution_policy_transition"] = transition

    def _training_state_byte_limit(self, epochs: int) -> int:
        model = self.encoder.state_dict()
        model_bytes = sum(value.numel() * value.element_size() for value in model.values())
        # Weights + two Adam moments, plus headroom for tensor envelopes,
        # sampling IDs, RNG state and one history record per completed epoch.
        return (
            4 * model_bytes + 4096 * len(model)
            + 256 * len(self.chunks) + 2048 * epochs + 2 * 1024 * 1024
        )

    def _validate_resume_payload(
        self, state: dict[str, Any], *, epoch: int, batch: int, steps: int,
        batches_per_epoch: int,
    ) -> None:
        def finite_nonnegative(value: Any) -> bool:
            return type(value) in (int, float) and math.isfinite(value) and value >= 0

        history = state.get("history")
        loss_sum = state.get("epoch_loss_sum")
        if (
            not isinstance(history, list) or len(history) != epoch
            or not finite_nonnegative(loss_sum) or (batch == 0 and loss_sum != 0)
            or type(state.get("encoder_trained")) is not bool
        ):
            raise ValueError("retriever checkpoint history or loss accumulator is invalid")
        for index, entry in enumerate(history, start=1):
            if (
                not isinstance(entry, dict)
                or type(entry.get("epoch")) is not int or entry["epoch"] != index
                or type(entry.get("steps")) is not int
                or entry.get("steps") != index * batches_per_epoch
                or not finite_nonnegative(entry.get("loss"))
            ):
                raise ValueError("retriever checkpoint history is invalid")

        def valid_tensor(value: Any, expected: torch.Tensor, label: str) -> None:
            if (
                not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.device.type != "cpu" or value.shape != expected.shape
                or value.dtype != expected.dtype
            ):
                raise ValueError(f"retriever checkpoint {label} tensor shape or dtype is invalid")
            if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"retriever checkpoint {label} tensor contains non-finite values")

        model = state.get("model_state_dict")
        expected_model = self.encoder.state_dict()
        if not isinstance(model, dict) or model.keys() != expected_model.keys():
            raise ValueError("retriever checkpoint model tensor names are invalid")
        for name, expected in expected_model.items():
            valid_tensor(model[name], expected, "model")
        optimizer = state.get("optimizer_state_dict")
        if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
            raise ValueError("retriever checkpoint optimizer is invalid")
        groups = optimizer.get("param_groups")
        if not isinstance(groups, list) or len(groups) != len(self.optimizer.param_groups):
            raise ValueError("retriever checkpoint optimizer groups are invalid")
        expected_parameters: dict[int, torch.Tensor] = {}
        for saved, current in zip(groups, self.optimizer.param_groups):
            ids = saved.get("params") if isinstance(saved, dict) else None
            if not isinstance(ids, list) or len(ids) != len(current["params"]):
                raise ValueError("retriever checkpoint optimizer parameter count is invalid")
            if (
                set(saved) != set(current)
                or any(saved[key] != value for key, value in current.items() if key != "params")
            ):
                raise ValueError("retriever checkpoint optimizer configuration is invalid")
            for identifier, parameter in zip(ids, current["params"]):
                if type(identifier) is not int or identifier in expected_parameters:
                    raise ValueError("retriever checkpoint optimizer parameter IDs are invalid")
                expected_parameters[identifier] = parameter
        if steps > 0 and optimizer["state"].keys() != expected_parameters.keys():
            raise ValueError("retriever checkpoint optimizer moments are incomplete")
        for identifier, values in optimizer["state"].items():
            if identifier not in expected_parameters or not isinstance(values, dict):
                raise ValueError("retriever checkpoint optimizer state is invalid")
            if (
                set(values) != {"step", "exp_avg", "exp_avg_sq"}
                or type(values.get("step")) is not int or not 0 <= values["step"] <= steps
            ):
                raise ValueError("retriever checkpoint optimizer step is invalid")
            for name in ("exp_avg", "exp_avg_sq"):
                valid_tensor(values[name], expected_parameters[identifier], "optimizer")
            if bool((values["exp_avg_sq"] < 0).any().item()):
                raise ValueError("retriever checkpoint optimizer second moment is negative")
        rng = torch.get_rng_state()
        valid_tensor(state.get("torch_rng_state"), rng, "RNG")
        valid_tensor(state.get("epoch_generator_state"), rng, "sampler RNG")

    def _save_training_state(self, path: Path, position: dict[str, Any]) -> None:
        """Commit one portable snapshot; a failed write leaves the old one intact."""
        optimizer = self.optimizer.state_dict()
        optimizer["state"] = {
            key: {
                name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for name, value in values.items()
            }
            for key, values in optimizer["state"].items()
        }
        payload = {
            "format_version": 1,
            "kind": "jarvis-retriever-training",
            "corpus_sha256": self.corpus_sha256,
            "tokenizer_sha256": self.tokenizer.fingerprint(),
            "encoder_config": asdict(self.encoder.cfg),
            "sampling_weights": dict(self.sampling_weights),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in self.encoder.state_dict().items()
            },
            "optimizer_state_dict": optimizer,
            "encoder_trained": self.encoder.trained_on_corpus,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "execution_policy": dict(self.execution_policy),
            "execution_policy_transitions": list(self.execution_policy_transitions),
            **position,
        }
        if str(self.device).startswith("cuda"):
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def train(
        self,
        *,
        epochs: int = 4,
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
        progress_interval: int = 200,
        cancellation_event: threading.Event | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 200,
    ) -> list[dict]:
        """Fit or exactly resume batches using the saved epoch sampler/RNG state.

        A resumed loader replays only its already consumed sample indices;
        those batches perform no encoder or optimizer work. Configuration,
        corpus, tokenizer and sampling weights must match the snapshot.
        """
        if min(epochs, batch_size, progress_interval, checkpoint_interval) <= 0:
            raise ValueError("epochs, batch size and intervals must be positive")
        _check_cancelled(cancellation_event)
        pairs = build_positive_pairs(self.chunks, cancellation_event=cancellation_event)
        if len(pairs) < 2:
            raise ValueError("at least two corpus-derived positive pairs are required")

        def collate(batch: list[ContrastivePair]) -> tuple[list[str], list[str]]:
            return ([pair.anchor for pair in batch], [pair.positive for pair in batch])

        generator = torch.Generator().manual_seed(self.seed)
        sampler: WeightedRandomSampler | None = None
        if self.sampling_weights:
            pair_counts = Counter(pair.anchor_chunk_id for pair in pairs)
            pair_weights = [
                self.sampling_weights[pair.anchor_chunk_id]
                / pair_counts[pair.anchor_chunk_id]
                for pair in pairs
            ]
            sampler = WeightedRandomSampler(
                pair_weights,
                num_samples=len(pairs),
                replacement=True,
                generator=generator,
            )
        loader = DataLoader(
            ContrastiveDataset(pairs),
            batch_size=min(batch_size, len(pairs)),
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=len(pairs) > batch_size,
            collate_fn=collate,
            generator=generator,
        )
        total_batches = len(loader) * epochs
        history: list[dict] = []
        train_config = {
            "epochs": epochs, "batch_size": batch_size,
            "temperature": self.temperature, "seed": self.seed,
            "learning_rate": self.learning_rate, "batches_per_epoch": len(loader),
        }
        state_path = (
            Path(checkpoint_dir) / "retriever_training_state.pt"
            if checkpoint_dir is not None else None
        )
        start_epoch = 0
        start_batch = 0
        steps = 0
        loss_sum = 0.0
        restored_state = None
        epoch_generator_state = generator.get_state()
        if state_path is not None and state_path.is_file():
            if state_path.stat().st_size > self._training_state_byte_limit(epochs):
                raise ValueError("retriever checkpoint exceeds the model/corpus size limit")
            restored_state = torch.load(state_path, map_location="cpu", weights_only=True)
            expected = {
                "format_version": 1, "kind": "jarvis-retriever-training",
                "corpus_sha256": self.corpus_sha256,
                "tokenizer_sha256": self.tokenizer.fingerprint(),
                "encoder_config": asdict(self.encoder.cfg),
                "sampling_weights": self.sampling_weights,
                "train_config": train_config,
            }
            if not isinstance(restored_state, dict) or any(
                restored_state.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("retriever checkpoint lineage or configuration mismatch")
            start_epoch = restored_state["epoch"]
            start_batch = restored_state["batch_in_epoch"]
            steps = restored_state["steps"]
            if (
                any(type(value) is not int for value in (start_epoch, start_batch, steps))
                or not 0 <= start_epoch <= epochs
                or not 0 <= start_batch <= len(loader)
                or steps != start_epoch * len(loader) + start_batch
                or (start_epoch == epochs and start_batch != 0)
            ):
                raise ValueError("retriever checkpoint position is invalid")
            self._validate_resume_payload(
                restored_state, epoch=start_epoch, batch=start_batch,
                steps=steps, batches_per_epoch=len(loader),
            )
            self.resume_metadata = {"resumed": True, "steps": steps, "epoch": start_epoch}
            self._restore_execution_metadata(restored_state, steps)
            self.encoder.load_state_dict(restored_state["model_state_dict"])
            self.optimizer.load_state_dict(restored_state["optimizer_state_dict"])
            self.encoder.trained_on_corpus = restored_state["encoder_trained"]
            epoch_generator_state = restored_state["epoch_generator_state"]
            history = list(restored_state["history"])
            loss_sum = float(restored_state["epoch_loss_sum"])
        else:
            self.resume_metadata = {"resumed": False, "steps": 0, "epoch": 0}
            self.execution_policy_transitions = []
        self.steps_completed = steps
        self.completed_epochs = start_epoch
        if progress_callback is not None:
            progress_callback(steps, total_batches)
        if start_epoch == epochs:
            torch.set_rng_state(restored_state["torch_rng_state"])
            random.setstate(restored_state["python_rng_state"])
            if "cuda_rng_states" in restored_state and str(self.device).startswith("cuda"):
                torch.cuda.set_rng_state_all(restored_state["cuda_rng_states"])
            return history

        def checkpoint(epoch: int, completed_batches: int) -> None:
            if state_path is not None:
                self._save_training_state(state_path, {
                    "train_config": train_config, "epoch": epoch,
                    "batch_in_epoch": completed_batches, "steps": steps,
                    "epoch_loss_sum": loss_sum, "history": list(history),
                    "epoch_generator_state": epoch_generator_state,
                })

        self.encoder.train()
        first_new_step = steps + 1
        for epoch in range(start_epoch, epochs):
            if epoch == start_epoch:
                generator.set_state(epoch_generator_state)
            else:
                epoch_generator_state = generator.get_state()
            iterator = iter(loader)
            skipped = start_batch if epoch == start_epoch else 0
            for _ in range(skipped):
                _check_cancelled(cancellation_event)
                next(iterator)
            if restored_state is not None:
                torch.set_rng_state(restored_state["torch_rng_state"])
                random.setstate(restored_state["python_rng_state"])
                if "cuda_rng_states" in restored_state and str(self.device).startswith("cuda"):
                    torch.cuda.set_rng_state_all(restored_state["cuda_rng_states"])
                restored_state = None
            completed_batches = skipped
            for anchors, positives in iterator:
                if cancellation_event is not None and cancellation_event.is_set():
                    checkpoint(epoch, completed_batches)
                    _check_cancelled(cancellation_event)
                self.optimizer.zero_grad(set_to_none=True)
                encoded_anchors, encoded_positives = self._encode_pairs(anchors, positives)
                loss = info_nce(encoded_anchors, encoded_positives, self.temperature)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("non-finite contrastive loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.optimizer.step()
                loss_sum += float(loss.detach().cpu().item())
                steps += 1
                completed_batches += 1
                self.steps_completed = steps
                if steps == first_new_step or steps % checkpoint_interval == 0:
                    checkpoint(epoch, completed_batches)
                if progress_callback is not None and (
                    steps == first_new_step or steps % progress_interval == 0 or steps == total_batches
                ):
                    progress_callback(steps, total_batches)
            history.append(
                {"epoch": epoch + 1, "loss": loss_sum / len(loader), "steps": steps}
            )
            self.completed_epochs = epoch + 1
            loss_sum = 0.0
            epoch_generator_state = generator.get_state()
            if epoch + 1 == epochs:
                self.encoder.trained_on_corpus = True
            checkpoint(epoch + 1, 0)
            _check_cancelled(cancellation_event)
        self.encoder.trained_on_corpus = True
        return history

    @torch.no_grad()
    def measure_retrieval_uncertainty(
        self,
        *,
        batch_size: int = 32,
        cancellation_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, float]:
        """Measure match uncertainty against deterministic, bounded local negatives.

        For a corpus fitting one batch this is the original all-corpus score.
        Larger corpora use only the other chunks in each batch as negatives;
        these scores are not calibrated against every document in the corpus.
        A final singleton overlaps its predecessor to retain one negative.
        """

        if len(self.chunks) < 2:
            raise RuntimeError("at least two chunks are required to measure uncertainty")
        if batch_size < 2:
            raise ValueError("uncertainty batch_size must be at least two")
        _check_cancelled(cancellation_event)
        self.uncertainty_metadata = {
            "negative_scope": "full_corpus" if len(self.chunks) <= batch_size else "local_batch",
            "batch_size": batch_size,
            "max_negatives_per_chunk": min(batch_size, len(self.chunks)) - 1,
            "tail_policy": "overlap_previous_chunk_if_singleton",
        }
        measured: dict[str, float] = {}
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            for start in range(0, len(self.chunks), batch_size):
                _check_cancelled(cancellation_event)
                end = min(start + batch_size, len(self.chunks))
                batch_start = start - 1 if end - start == 1 else start
                chunks = self.chunks[batch_start:end]
                views = [_deterministic_views(chunk.text) for chunk in chunks]
                anchors = self._encode([view[0] for view in views])
                _check_cancelled(cancellation_event)
                positives = self._encode([view[1] for view in views])
                logits = anchors @ positives.transpose(0, 1)
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("non-finite retrieval uncertainty logits")
                probabilities = F.softmax(logits / self.temperature, dim=-1)
                confidence = probabilities.detach().cpu().diagonal().tolist()
                for chunk, value in zip(chunks[start - batch_start:], confidence[start - batch_start:]):
                    measured[chunk.chunk_id] = max(0.0, min(1.0, 1.0 - float(value)))
                if progress_callback is not None:
                    progress_callback(end, len(self.chunks))
        finally:
            self.encoder.train(was_training)
        return measured

    def save(self, path: str, history: list[dict]) -> None:
        self.encoder.save(
            path,
            corpus_sha256=self.corpus_sha256,
            tokenizer_sha256=self.tokenizer.fingerprint(),
            training={
                "objective": "symmetric InfoNCE",
                "history": history,
                "sampling": dict(self.sampling_metadata),
                "uncertainty": dict(self.uncertainty_metadata),
                "execution_policy": dict(self.execution_policy),
                "execution_policy_transitions": list(self.execution_policy_transitions),
            },
        )


__all__ = [
    "ContrastiveDataset",
    "ContrastivePair",
    "ContrastiveTrainer",
    "build_positive_pairs",
    "info_nce",
]
