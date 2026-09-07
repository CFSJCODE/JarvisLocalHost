"""Resource-aware local training loop for the sovereign language model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import re
import tempfile
import threading
import time
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from jarvis_localhost.ai.dataset import (
    TextDataset,
    deterministic_split,
    validate_sampling_payload,
)
from jarvis_localhost.ai.language_model import JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.corpus.provenance import sha256_file
from jarvis_localhost.paths import MODELS_ROOT


_TRAINING_MUTEX = threading.Lock()


def _legacy_numpy_bytes(value: str, encoding: str) -> bytes:
    if encoding != "latin1":
        raise pickle.UnpicklingError("unsupported legacy NumPy byte encoding")
    return value.encode("latin1")


def _legacy_device_tensor_to_cpu(
    data: np.ndarray, dtype: torch.dtype, device: Any, requires_grad: bool,
) -> torch.Tensor:
    """Rebuild old DirectML optimizer tensors without initializing that device."""
    if (
        not isinstance(data, np.ndarray) or data.dtype.hasobject
        or not isinstance(dtype, torch.dtype) or type(requires_grad) is not bool
    ):
        raise pickle.UnpicklingError("invalid legacy device tensor payload")
    tensor = torch.from_numpy(data).to(dtype=dtype, device="cpu")
    tensor.requires_grad = requires_grad
    return tensor


class _TrainerStateUnpickler(pickle.Unpickler):
    """Read legacy NumPy RNG tuples without allowing arbitrary pickle globals."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            ("_codecs", "encode"): _legacy_numpy_bytes,
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor"): torch._utils._rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
            ("torch._utils", "_rebuild_device_tensor_from_numpy"): _legacy_device_tensor_to_cpu,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        }
        allowed.update({
            ("torch", dtype_name): getattr(torch, dtype_name)
            for dtype_name in (
                "float16", "float32", "float64", "bfloat16", "bool",
                "uint8", "int8", "int16", "int32", "int64",
                "complex64", "complex128",
            )
        })
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                f"unsupported trainer checkpoint global: {module}.{name}"
            )
        return allowed[(module, name)]


class _TrainerStatePickle:
    Unpickler = _TrainerStateUnpickler


def load_trainer_state(path: str | Path) -> dict[str, Any]:
    """Load current sidecars safely and retain compatibility with local F10 states."""

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # PyTorch 2.4 cannot allowlist NumPy's ndarray RNG state through its
        # weights-only loader. Restrict that legacy fallback to known tensor
        # and NumPy constructors instead of unrestricted pickle execution.
        state = torch.load(
            path, map_location="cpu", weights_only=False,
            pickle_module=_TrainerStatePickle,
        )
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("unsupported trainer_state checkpoint format")
    for field in ("step", "tokens_seen"):
        if type(state.get(field)) is not int or state[field] < 0:
            raise ValueError(f"trainer_state {field} must be a non-negative integer")
    return state


def validate_trainer_checkpoint(
    state_path: str | Path,
    state: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify that model bytes, manifest and optimizer sidecar form one checkpoint."""

    path = Path(state_path)
    suffix = ".trainer_state.pt"
    if not path.name.endswith(suffix):
        raise ValueError("invalid trainer_state checkpoint filename")
    checkpoint_path = path.with_name(path.name[:-len(suffix)] + ".pt")
    if manifest is None:
        manifest = json.loads(
            checkpoint_path.with_suffix(".pt.manifest.json").read_text(encoding="utf-8")
        )
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid language-model checkpoint manifest")
    if (
        manifest.get("format_version") != JarvisTransformer.CHECKPOINT_VERSION
        or manifest.get("artifact_envelope_version") != 1
        or manifest.get("checkpoint_filename") != checkpoint_path.name
    ):
        raise ValueError("unsupported or mismatched language-model checkpoint manifest")
    digest = manifest.get("checkpoint_sha256")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or checkpoint_path.stat().st_size != manifest.get("checkpoint_size_bytes")
        or sha256_file(checkpoint_path) != digest
    ):
        raise ValueError("language-model checkpoint bytes do not match its manifest")
    for field in ("corpus_sha256", "tokenizer_sha256"):
        if manifest.get(field) != state.get(field):
            raise ValueError(f"trainer_state {field} does not match model checkpoint")
    training = manifest.get("training")
    if not isinstance(training, Mapping) or any(
        training.get(field) != state.get(field) for field in ("step", "tokens_seen")
    ):
        raise ValueError("trainer_state position does not match model checkpoint")
    # Legacy sidecars lack this field; their manifest and position still have
    # to agree. New states additionally detect a same-step replacement model.
    if "checkpoint_sha256" in state and state["checkpoint_sha256"] != digest:
        raise ValueError("trainer_state model digest does not match checkpoint")
    return dict(manifest)


def _control_training_examples(segments: list[str]) -> tuple[list[str], dict]:
    """Build citation/control examples using only exact supplied passages."""

    evidence_examples = [
        (
            "Pergunta:\nExtraia somente a passagem apoiada pela evidência.\n"
            f"Evidências:\n[E1] {segment}\n"
            f"Resposta:\n{segment} [E1]"
        )
        for segment in segments
    ]
    abstention_examples = [
        "Pergunta:\nResponda somente se houver evidência.\n"
        "Evidências:\n\nResposta:\nNÃO ENCONTRADO."
    ]
    origin_digest = hashlib.sha256(
        "\x1e".join(segments).encode("utf-8")
    ).hexdigest()
    metadata = {
        "format_version": 1,
        "evidence_examples": len(evidence_examples),
        "abstention_examples": len(abstention_examples),
        "evidence_origin": "exact_provided_corpus_segments",
        "evidence_origin_sha256": origin_digest,
        "abstention_origin": "empty_evidence_protocol_only",
        "invented_facts": False,
    }
    return [*evidence_examples, *abstention_examples], metadata


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2_000
    lr_decay: bool = True
    batch_size: int = 16
    gradient_accumulation: int = 1
    context_len: int = 256
    validation_fraction: float = 0.1
    log_interval: int = 25
    eval_interval: int = 200
    eval_iters: int = 20
    checkpoint_dir: str = str(MODELS_ROOT)
    checkpoint_every: int = 500
    num_workers: int = 0
    device: Any = "cpu"
    seed: int = 1_337

    def validate(self) -> None:
        if self.max_steps <= 0 or self.batch_size <= 0 or self.context_len <= 0:
            raise ValueError("steps, batch_size and context_len must be positive")
        if self.gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")


class TrainingCancelled(RuntimeError):
    pass


class DirectMLAdamW(torch.optim.Optimizer):
    """GPU-native AdamW avoiding scalar lerp fallback on DirectML / D3D12."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Pure in-place tensor operations (no scalar lerp)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class JarvisTrainer:
    """AdamW training with deterministic splits and auditable checkpoints."""

    def __init__(
        self,
        model: JarvisTransformer,
        tokenizer: JarvisTokenizer,
        corpus: str,
        cfg: TrainConfig,
        *,
        cancellation_event: threading.Event | None = None,
        chunk_ids: Sequence[str] | None = None,
        canonical_corpus_sha256: str | None = None,
        sampling_payload: Mapping[str, Any] | None = None,
    ) -> None:
        cfg.validate()
        if cfg.context_len != model.cfg.context_len:
            raise ValueError("trainer context_len must match the language model")
        if tokenizer.corpus_sha256 and tokenizer.corpus_sha256 != tokenizer.corpus_digest(corpus):
            raise ValueError("tokenizer and training corpus digests differ")

        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.corpus_sha256 = tokenizer.corpus_digest(corpus)
        self.cancellation_event = cancellation_event or threading.Event()
        self.history: list[dict] = []
        self.step = 0
        self.tokens_seen = 0
        # Set by apply_resume_state() when this run continues a previous,
        # interrupted call to train() against the exact same corpus and
        # tokenizer; train() starts its loop here instead of at 0.
        self._start_step = 0
        # Bounds disk usage for the optimizer/RNG sidecar written by
        # _checkpoint(): only the most recent checkpoint's sidecar is ever
        # useful for resuming, so each new one deletes the previous.
        self._last_trainer_state_path: Path | None = None

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ``_corpus_text`` separates canonical chunks with blank lines. Encode
        # each such unit with BOS/EOS and teach SEP between them; no semantic
        # labels or external vocabulary are introduced by this structure.
        segments = [
            segment.strip()
            for segment in re.split(r"(?:\r?\n[ \t]*){2,}", corpus)
            if segment.strip()
        ]
        if chunk_ids is not None:
            normalized_chunk_ids = tuple(str(value) for value in chunk_ids)
            if (
                len(normalized_chunk_ids) != len(segments)
                or any(not value for value in normalized_chunk_ids)
                or len(normalized_chunk_ids) != len(set(normalized_chunk_ids))
            ):
                raise ValueError("chunk_ids must uniquely align with corpus segments")
        else:
            normalized_chunk_ids = ()
        self.chunk_ids = normalized_chunk_ids
        self.chunk_segments = tuple(segments)
        self.canonical_corpus_sha256 = canonical_corpus_sha256
        sampling_weights: dict[str, float] = {}
        self.sampling_metadata: dict[str, Any] = {"weighted": False}
        if sampling_payload is not None:
            if not normalized_chunk_ids or not canonical_corpus_sha256:
                raise ValueError(
                    "sampling weights require aligned chunk_ids and canonical corpus SHA-256"
                )
            sampling_weights, self.sampling_metadata = validate_sampling_payload(
                sampling_payload,
                expected_corpus_sha256=canonical_corpus_sha256,
                expected_chunk_ids=normalized_chunk_ids,
            )
        control_examples, self.control_training_metadata = _control_training_examples(
            segments
        )
        # Preserve protocol lineage when the orchestrator re-saves the trained
        # model with a canonical corpus digest after this trainer completes.
        self.model.control_training_metadata = dict(self.control_training_metadata)
        training_units = [*segments, *control_examples]
        source_owners: list[str | None] = [
            *(normalized_chunk_ids or (None for _ in segments)),
            *(normalized_chunk_ids or (None for _ in segments)),
            None,
        ]
        token_ids: list[int] = []
        unit_starts: list[int] = []
        for index, unit in enumerate(training_units):
            unit_starts.append(len(token_ids))
            token_ids.extend(tokenizer.encode(unit, add_special=True))
            if index + 1 < len(training_units):
                token_ids.append(tokenizer.SEP_ID)
        self.training_token_ids = tuple(token_ids)
        train_ids, validation_ids = deterministic_split(
            token_ids, validation_fraction=cfg.validation_fraction
        )
        # A validation tail must never consume the only complete next-token
        # window. In that compact-corpus case, keep all tokens for training.
        if len(train_ids) < cfg.context_len + 1 <= len(token_ids):
            train_ids, validation_ids = list(token_ids), []
        self.train_dataset = TextDataset(train_ids, cfg.context_len)
        self.validation_dataset = TextDataset(validation_ids, cfg.context_len)
        if len(self.train_dataset) == 0:
            minimum = cfg.context_len + 1
            raise ValueError(
                f"corpus has {len(token_ids)} tokens; at least {minimum} are required "
                "for the selected context length"
            )

        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        effective_batch = min(cfg.batch_size, len(self.train_dataset))
        sampler: WeightedRandomSampler | None = None
        if sampling_weights:
            owners = [
                source_owners[
                    max(0, bisect_right(unit_starts, offset) - 1)
                ]
                for offset in self.train_dataset.offsets
            ]
            owner_counts = Counter(owners)
            # The empty-evidence abstention control has no canonical chunk ID;
            # keep it trainable at the uniform per-chunk prior.
            fallback_weight = 1.0 / len(sampling_weights)
            window_weights = [
                (
                    sampling_weights.get(owner, fallback_weight)
                    / max(1, owner_counts[owner])
                )
                for owner in owners
            ]
            sampler = WeightedRandomSampler(
                window_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
                generator=generator,
            )
            self.sampling_metadata["training_windows"] = len(window_weights)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=effective_batch,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            num_workers=max(0, cfg.num_workers),
            generator=generator,
            pin_memory=str(self.device).startswith("cuda"),
        )
        self.model.training_sampling_metadata = dict(self.sampling_metadata)

        decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
        no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
        self.optimizer = DirectMLAdamW(
            (
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )

    def cancel(self) -> None:
        self.cancellation_event.set()

    def learning_rate_at(self, step: int) -> float:
        if not self.cfg.lr_decay:
            return self.cfg.learning_rate
        warmup_steps = self.effective_warmup_steps
        if step < warmup_steps:
            return self.cfg.learning_rate * (step + 1) / max(1, warmup_steps)
        progress = min(
            1.0,
            (step - warmup_steps)
            / max(1, self.cfg.max_steps - warmup_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.cfg.min_learning_rate_ratio
        return self.cfg.learning_rate * (floor + (1.0 - floor) * cosine)

    @property
    def effective_warmup_steps(self) -> int:
        """Use at most the first 10% of a finite run for warmup."""

        return min(self.cfg.warmup_steps, self.cfg.max_steps // 10)

    @torch.no_grad()
    def measure_chunk_losses(
        self, *, checkpoint_path: str | Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, float]:
        """Measure exact segment loss, resuming only against the same final LM."""

        if not self.chunk_ids:
            raise RuntimeError("chunk_ids were not supplied to this trainer")
        destination = Path(checkpoint_path) if checkpoint_path is not None else None
        binding = None
        measured: dict[str, float] = {}

        def measurement_digest(values: dict[str, float]) -> str:
            encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        if destination is not None:
            binding = {
                "format_version": 1,
                "model_sha256": sha256_file(Path(self.cfg.checkpoint_dir) / "jarvis_final.pt"),
                "tokenizer_sha256": self.tokenizer.fingerprint(),
                "canonical_corpus_sha256": self.canonical_corpus_sha256,
                "text_corpus_sha256": self.corpus_sha256,
                "context_len": self.cfg.context_len,
            }
            if destination.exists():
                limit = 65536 + 160 * len(self.chunk_ids)
                if destination.stat().st_size > limit:
                    raise ValueError("LM measurement checkpoint exceeds corpus size limit")
                saved = json.loads(destination.read_text(encoding="utf-8"))
                if saved.get("binding") != binding:
                    raise ValueError("LM measurement checkpoint belongs to another model or corpus")
                measured = saved.get("measurements", {})
                active = set(self.chunk_ids)
                if not isinstance(measured, dict) or any(
                    key not in active or type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0
                    for key, value in measured.items()
                ):
                    raise ValueError("invalid LM measurement checkpoint")
                if saved.get("measurements_sha256") != measurement_digest(measured):
                    raise ValueError("LM measurement checkpoint checksum mismatch")

        saved_count = len(measured)

        def persist() -> None:
            nonlocal saved_count
            if destination is None or len(measured) == saved_count:
                return
            descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump({"binding": binding, "measurements": measured, "measurements_sha256": measurement_digest(measured)}, stream, separators=(",", ":"), allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                saved_count = len(measured)
            finally:
                temporary.unlink(missing_ok=True)

        was_training = self.model.training
        self.model.eval()
        try:
            if progress_callback is not None:
                progress_callback(len(measured), len(self.chunk_ids))
            for chunk_id, segment in zip(self.chunk_ids, self.chunk_segments):
                if self.cancellation_event.is_set():
                    raise TrainingCancelled("LM measurement cancelled by operator")
                if chunk_id in measured:
                    continue
                ids = self.tokenizer.encode(segment, add_special=True)
                weighted_loss = 0.0
                target_count = 0
                for start in range(0, max(1, len(ids) - 1), self.cfg.context_len):
                    if self.cancellation_event.is_set():
                        raise TrainingCancelled("LM measurement cancelled by operator")
                    window = ids[start : start + self.cfg.context_len + 1]
                    if len(window) < 2:
                        continue
                    # Limit accelerator graph shapes while keeping exactly the
                    # same causal targets; padding is ignored by the LM loss.
                    count = len(window) - 1
                    width = min(self.cfg.context_len, 1 << (count - 1).bit_length())
                    inputs = torch.tensor(
                        [self.tokenizer.pad_sequence(window[:-1], width)],
                        dtype=torch.long, device=self.device
                    )
                    targets = torch.tensor(
                        [self.tokenizer.pad_sequence(window[1:], width)],
                        dtype=torch.long, device=self.device
                    )
                    _, loss = self.model(inputs, targets)
                    if loss is None:
                        continue
                    numeric = float(loss.detach().cpu().item())
                    if not math.isfinite(numeric):
                        raise FloatingPointError(
                            f"non-finite measured LM loss for {chunk_id}"
                        )
                    weighted_loss += numeric * count
                    target_count += count
                if target_count <= 0:
                    raise ValueError(f"chunk {chunk_id} has no measurable tokens")
                measured[chunk_id] = weighted_loss / target_count
                if len(measured) % 5000 == 0:
                    persist()
                if progress_callback is not None and (len(measured) % 200 == 0 or len(measured) == len(self.chunk_ids)):
                    progress_callback(len(measured), len(self.chunk_ids))
        finally:
            self.model.train(was_training)
            persist()
        return measured

    def release_training_buffers(self) -> None:
        """Release completed-fit data and optimizer allocations before retrieval."""
        if self.step + 1 < self.cfg.max_steps:
            raise RuntimeError("cannot release buffers before language training completes")
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()
        self.training_token_ids = ()
        self.train_loader = None
        self.train_dataset = None
        self.validation_dataset = None

    @torch.no_grad()
    def evaluate(self) -> float | None:
        if len(self.validation_dataset) == 0:
            return None
        loader = DataLoader(
            self.validation_dataset,
            batch_size=min(self.cfg.batch_size, len(self.validation_dataset)),
            shuffle=False,
            num_workers=0,
        )
        self.model.eval()
        losses: list[float] = []
        for index, (inputs, targets) in enumerate(loader):
            if index >= self.cfg.eval_iters:
                break
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            _, loss = self.model(inputs, targets)
            if loss is not None and bool(torch.isfinite(loss).item()):
                losses.append(float(loss.detach().cpu().item()))
        self.model.train()
        return sum(losses) / len(losses) if losses else None

    def apply_resume_state(self, trainer_state_path: str | Path) -> int:
        """Restore optimizer/RNG/position state from a sidecar a previous,
        interrupted call to :meth:`train` wrote via ``_checkpoint``.

        The caller is expected to have already constructed this trainer
        (and, upstream of that, loaded the exact model weights and tokenizer
        the interrupted run last checkpointed) against what it believes is
        the SAME corpus. This method is the last line of defense: it
        independently re-verifies the sidecar's own recorded
        ``corpus_sha256`` / ``canonical_corpus_sha256`` / ``tokenizer_sha256``
        against this trainer's freshly computed values and raises
        ``ValueError`` on any mismatch rather than resuming. That mirrors the
        existing fail-loud lineage checks elsewhere in this class
        (``__init__``'s tokenizer/corpus digest check) and in
        ``JarvisTransformer.load`` / ``JarvisTokenizer.load``
        (``expected_corpus_sha256`` / ``expected_tokenizer_sha256``): a
        resume is always exact-or-refused, never a silent best-effort
        approximation that could quietly continue training on different text
        than the earlier steps saw.

        Returns the step training will resume at. Known, disclosed
        approximations even on a successful resume: the data loader's
        shuffle order restarts from the beginning of a fresh epoch (its
        exact position is not persisted), and curiosity-engine sampling
        weights are recomputed fresh rather than replayed byte-for-byte --
        neither affects which text the model is trained on, only the order
        and relative frequency chunks are revisited in.
        """

        path = Path(trainer_state_path)
        state = load_trainer_state(path)
        if state.get("corpus_sha256") != self.corpus_sha256:
            raise ValueError(
                "trainer_state corpus digest does not match the current "
                "training corpus; refusing to resume against changed data"
            )
        if (
            self.canonical_corpus_sha256
            and state.get("canonical_corpus_sha256") != self.canonical_corpus_sha256
        ):
            raise ValueError(
                "trainer_state canonical corpus digest does not match the "
                "current corpus; refusing to resume against changed data"
            )
        if state.get("tokenizer_sha256") != self.tokenizer.fingerprint():
            raise ValueError(
                "trainer_state tokenizer digest does not match the current "
                "tokenizer; refusing to resume with a different vocabulary"
            )
        saved_step = int(state["step"])
        if not 0 <= saved_step < self.cfg.max_steps:
            raise ValueError("trainer_state step is out of range for this run")
        validate_trainer_checkpoint(path, state)

        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.step = saved_step
        self.tokens_seen = int(state["tokens_seen"])
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])

        history = state.get("history")
        history_path = path.parent / "train_history.json"
        if history is None and history_path.is_file():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # train_history.json is a display/logging convenience, never
                # part of the lineage guarantee above -- losing it costs a
                # gap in the reported loss curve, not correctness.
                history = []
        # An interrupted later save can advance the display history beyond
        # the last committed sidecar. Do not present those lost steps twice.
        self.history = [
            dict(entry) for entry in (history if isinstance(history, list) else [])
            if isinstance(entry, dict)
            and type(entry.get("step")) is int
            and 0 <= entry["step"] <= saved_step
        ]

        self._start_step = saved_step + 1
        self._last_trainer_state_path = path
        return self._start_step

    def train(
        self, callback: Callable[[dict], None] | None = None
    ) -> list[dict]:
        if not _TRAINING_MUTEX.acquire(blocking=False):
            raise RuntimeError("another training run is active in this process")
        try:
            if self._start_step >= self.cfg.max_steps:
                # The LM already finished; retain its weights and let the
                # caller continue the retriever/index stages of the pipeline.
                return list(self.history)
            started = time.monotonic()
            self.model.train()
            iterator = iter(self.train_loader)
            last_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            for step in range(self._start_step, self.cfg.max_steps):
                if self.cancellation_event.is_set():
                    raise TrainingCancelled("training cancelled by operator")
                self.step = step
                learning_rate = self.learning_rate_at(step)
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate

                accumulated_loss = 0.0
                for _ in range(self.cfg.gradient_accumulation):
                    try:
                        inputs, targets = next(iterator)
                    except StopIteration:
                        iterator = iter(self.train_loader)
                        inputs, targets = next(iterator)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    _, loss = self.model(inputs, targets)
                    if loss is None or not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("non-finite language-model loss")
                    (loss / self.cfg.gradient_accumulation).backward()
                    accumulated_loss += float(loss.detach().cpu().item())
                    self.tokens_seen += inputs.numel()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                last_loss = accumulated_loss / self.cfg.gradient_accumulation

                if step % self.cfg.log_interval == 0 or step == self.cfg.max_steps - 1:
                    info = {
                        "step": step,
                        "loss": round(last_loss, 6),
                        "lr": learning_rate,
                        "tokens_seen": self.tokens_seen,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "progress": round((step + 1) / self.cfg.max_steps * 100.0, 2),
                    }
                    self.history.append(info)
                    if callback:
                        callback(dict(info))

                if step > 0 and step % self.cfg.eval_interval == 0:
                    validation_loss = self.evaluate()
                    if validation_loss is not None:
                        event = {"step": step, "val_loss": round(validation_loss, 6)}
                        self.history.append(event)
                        if callback:
                            callback(dict(event))

                if step > 0 and step % self.cfg.checkpoint_every == 0:
                    self._checkpoint(f"step_{step}", last_loss)

            self._checkpoint("final", last_loss)
            return list(self.history)
        finally:
            _TRAINING_MUTEX.release()

    def _checkpoint(self, tag: str, loss: float) -> None:
        directory = Path(self.cfg.checkpoint_dir)
        checkpoint_path = directory / f"jarvis_{tag}.pt"
        training = {
            "tokens_seen": self.tokens_seen,
            "step": self.step,
            "loss": loss,
            "optimizer": "AdamW",
            "control_training": dict(self.control_training_metadata),
            "sampling": dict(self.sampling_metadata),
            "config": {
                key: (str(value) if key == "device" else value)
                for key, value in asdict(self.cfg).items()
            },
        }
        self.model.save(
            checkpoint_path,
            corpus_sha256=self.corpus_sha256,
            tokenizer_sha256=self.tokenizer.fingerprint(),
            training=training,
        )
        history_path = directory / "train_history.json"
        temporary = history_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, history_path)

        # (audit fix, 2026-08-31) Persist enough state to genuinely resume a
        # run this process (or its whole host) crashes out of, instead of
        # only ever being able to load the model's frozen final weights.
        # This is a SEPARATE sidecar from jarvis_{tag}.pt/.manifest.json on
        # purpose: that pair is also the format used for the shipped,
        # inference-loaded production model (JarvisTransformer.load), whose
        # format_version/lineage contract must stay exactly as hardened as
        # it is today. Optimizer moments and RNG state have no business in
        # that contract, so they live here instead, and their absence is a
        # perfectly normal, expected state (any checkpoint saved before this
        # fix, or the final production artifact once a caller chooses not to
        # keep it) rather than a corruption to raise on.
        state_path = directory / f"jarvis_{tag}.trainer_state.pt"
        numpy_state = np.random.get_state()
        # Primitive RNG payloads allow new files to use weights-only loading.
        numpy_state = (
            numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]
        )
        optimizer_state = self.optimizer.state_dict()
        optimizer_state["state"] = {
            parameter_id: {
                name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for name, value in values.items()
            }
            for parameter_id, values in optimizer_state["state"].items()
        }
        trainer_state = {
            "format_version": 1,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "corpus_sha256": self.corpus_sha256,
            "canonical_corpus_sha256": self.canonical_corpus_sha256,
            "tokenizer_sha256": self.tokenizer.fingerprint(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "optimizer_state_dict": optimizer_state,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": numpy_state,
            "python_rng_state": random.getstate(),
            "history": list(self.history),
        }
        descriptor, temporary_state_name = tempfile.mkstemp(
            prefix=state_path.name + ".", suffix=".tmp", dir=directory
        )
        os.close(descriptor)
        temporary_state = Path(temporary_state_name)
        try:
            torch.save(trainer_state, temporary_state)
            with temporary_state.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_state, state_path)
        finally:
            temporary_state.unlink(missing_ok=True)

        # Only the newest sidecar can ever be resumed from (a later
        # checkpoint always supersedes an earlier one within the same run),
        # so drop the previous one now that the new one is safely on disk --
        # otherwise a long run with frequent checkpoints would accumulate one
        # full optimizer-state blob (roughly 2x the model's own size, for
        # AdamW's two moment tensors per parameter) per checkpoint forever.
        if (
            self._last_trainer_state_path is not None
            and self._last_trainer_state_path != state_path
            and self._last_trainer_state_path.exists()
        ):
            self._last_trainer_state_path.unlink(missing_ok=True)
        self._last_trainer_state_path = state_path


__all__ = [
    "JarvisTrainer", "TrainConfig", "TrainingCancelled",
    "load_trainer_state", "validate_trainer_checkpoint",
]
