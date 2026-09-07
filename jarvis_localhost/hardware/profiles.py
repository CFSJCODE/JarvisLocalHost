"""Corpus- and hardware-derived training resource profiles.

Profiles are planning ceilings, not memory reservations or enforced process RSS
limits. All values are computed locally from corpus volume and the detected
machine. Defaults stay conservative; explicit RAM requests keep a physical
reserve for the desktop and warn when current availability is insufficient.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .device import (
    CPU_THREADS_ENV,
    GIB,
    INTEROP_THREADS_ENV,
    MIB,
    DeviceDescriptor,
    HardwareInfo,
    detect_hardware,
    select_compute_device,
)


DEFAULT_MAX_HOST_MEMORY_BUDGET_BYTES = 16 * GIB
MAX_HOST_MEMORY_BUDGET_BYTES = 24 * GIB
MIN_HOST_MEMORY_RESERVE_BYTES = 4 * GIB
MAX_ACCELERATOR_MEMORY_BUDGET_BYTES = 6 * GIB
MIN_HOST_MEMORY_BUDGET_BYTES = 512 * MIB
MAX_CONTEXT_LENGTH = 512
MAX_GRADIENT_ACCUMULATION_STEPS = 16
MAX_TRAINING_STEPS = 10_000

HOST_MEMORY_ENV = "JARVIS_MAX_RAM_GB"
ACCELERATOR_MEMORY_ENV = "JARVIS_MAX_GPU_MEMORY_GB"
BATCH_SIZE_ENV = "JARVIS_BATCH_SIZE"
CONTEXT_LENGTH_ENV = "JARVIS_CONTEXT_LENGTH"
GRADIENT_ACCUMULATION_ENV = "JARVIS_GRAD_ACCUMULATION_STEPS"
MAX_STEPS_ENV = "JARVIS_MAX_STEPS"
DATALOADER_WORKERS_ENV = "JARVIS_DATALOADER_WORKERS"

_TEXT_EXTENSIONS = {
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".rst",
    ".text",
    ".txt",
    ".xml",
}
_DOCUMENT_EXTENSIONS = _TEXT_EXTENSIONS | {".pdf"}


@dataclass(frozen=True)
class CorpusStats:
    """Aggregate corpus measurements with no source paths or document text."""

    document_count: int = 0
    byte_count: int = 0
    character_count: int = 0
    estimated_tokens: int = 0
    source_kind: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResourceProfile:
    """Bounded settings derived from corpus volume and available resources."""

    profile_version: int
    backend: str
    corpus: CorpusStats
    model_tier: str
    vocabulary_size: int
    context_length: int
    embedding_dimension: int
    attention_heads: int
    transformer_layers: int
    feed_forward_dimension: int
    estimated_parameter_count: int
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    max_steps: int
    evaluation_interval: int
    checkpoint_interval: int
    host_memory_budget_bytes: int
    accelerator_memory_budget_bytes: int
    cpu_threads: int
    interop_threads: int
    dataloader_workers: int
    rag_chunk_tokens: int
    rag_overlap_tokens: int
    pretrained_weights: bool = False
    runtime_downloads_allowed: bool = False
    reason: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["corpus"] = self.corpus.to_dict()
        result["warnings"] = list(self.warnings)
        result["host_memory_budget_mib"] = round(
            self.host_memory_budget_bytes / MIB
        )
        result["accelerator_memory_budget_mib"] = round(
            self.accelerator_memory_budget_bytes / MIB
        )
        return result


CorpusInput = str | Path | Iterable[str | Path] | None


def detect_corpus_stats(corpus: CorpusInput = None) -> CorpusStats:
    """Measure raw text, one path, or an iterable of local document paths.

    Directory scans include common text formats and PDFs. Only aggregate counts
    leave this function; file names, paths and text are intentionally excluded
    from the returned object and from the hardware probe JSON.
    """

    if corpus is None:
        return CorpusStats()
    if isinstance(corpus, Path):
        return _stats_from_paths((corpus,), "path")
    if isinstance(corpus, str):
        path = _existing_path(corpus)
        if path is not None:
            return _stats_from_paths((path,), "path")
        return _stats_from_text(corpus)
    return _stats_from_paths((Path(item) for item in corpus), "paths")


def build_training_profile(
    hardware: HardwareInfo | None = None,
    corpus: CorpusInput = None,
    corpus_stats: CorpusStats | None = None,
    device: DeviceDescriptor | None = None,
    env: Mapping[str, str] | None = None,
) -> TrainingResourceProfile:
    """Build a deterministic, capped training plan for this corpus and host."""

    if corpus is not None and corpus_stats is not None:
        raise ValueError("Pass corpus or corpus_stats, not both.")
    hardware = hardware or detect_hardware()
    environment = os.environ if env is None else env
    stats = corpus_stats or detect_corpus_stats(corpus)
    device = device or select_compute_device(hardware, env=environment)
    warnings: list[str] = []

    host_budget = _host_memory_budget(hardware, environment, warnings)
    accelerator_budget = _accelerator_memory_budget(
        hardware, device, environment, warnings
    )
    if device.accelerated and hardware.gpu_shared_memory_bytes > 0:
        warnings.append(
            "Shared accelerator memory uses host RAM; host and accelerator "
            "budgets are not independent allocations."
        )
    tier = _choose_model_tier(stats.estimated_tokens, host_budget, accelerator_budget)
    model = _model_shape(tier, stats.estimated_tokens)
    context = _context_length(stats.estimated_tokens, environment, warnings)
    safe_batch_cap = _safe_batch_cap(device, accelerator_budget, context, tier)
    batch_size = _batch_size(
        device, accelerator_budget, context, tier, safe_batch_cap, environment, warnings
    )
    accumulation = _gradient_accumulation(
        stats.estimated_tokens, batch_size, environment, warnings
    )
    max_steps = _training_steps(
        stats.estimated_tokens, context, batch_size, accumulation, environment, warnings
    )
    cpu_threads, interop_threads, workers = _thread_plan(
        hardware, environment, warnings
    )
    estimated_parameters = _estimate_parameters(
        model["vocabulary_size"],
        model["embedding_dimension"],
        model["transformer_layers"],
        model["feed_forward_dimension"],
    )
    rag_chunk = min(384, max(96, context))
    rag_overlap = max(16, rag_chunk // 8)
    reason = (
        f"{tier} model derived from {stats.estimated_tokens} estimated corpus "
        f"tokens, {round(host_budget / GIB, 2)} GiB host budget and "
        f"{round(accelerator_budget / GIB, 2)} GiB accelerator budget on "
        f"{device.backend}."
    )

    return TrainingResourceProfile(
        profile_version=1,
        backend=device.backend,
        corpus=stats,
        model_tier=tier,
        vocabulary_size=model["vocabulary_size"],
        context_length=context,
        embedding_dimension=model["embedding_dimension"],
        attention_heads=model["attention_heads"],
        transformer_layers=model["transformer_layers"],
        feed_forward_dimension=model["feed_forward_dimension"],
        estimated_parameter_count=estimated_parameters,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        effective_batch_size=batch_size * accumulation,
        max_steps=max_steps,
        evaluation_interval=min(max_steps, min(250, max(1, max_steps // 5))),
        # Checkpoints frequentes o bastante para que uma queda custe pouco.
        # O intervalo anterior (max_steps // 2, teto de 1000) rendia apenas
        # dois checkpoints por execucao: uma queda real em 2026-09-03 (reboot
        # da maquina no step ~725 de 1890, antes do primeiro checkpoint em
        # 945) perdeu ~3h de treino sem deixar nada para o resume do F10
        # retomar. O custo por checkpoint e uma gravacao de poucos MB.
        checkpoint_interval=min(max_steps, max(1, min(250, max_steps // 8))),
        host_memory_budget_bytes=host_budget,
        accelerator_memory_budget_bytes=accelerator_budget,
        cpu_threads=cpu_threads,
        interop_threads=interop_threads,
        dataloader_workers=workers,
        rag_chunk_tokens=rag_chunk,
        rag_overlap_tokens=rag_overlap,
        pretrained_weights=False,
        runtime_downloads_allowed=False,
        reason=reason,
        warnings=tuple(warnings),
    )


def _stats_from_text(text: str) -> CorpusStats:
    encoded_size = len(text.encode("utf-8"))
    characters = len(text)
    tokens = _estimate_text_tokens(characters, encoded_size)
    return CorpusStats(
        document_count=1 if text else 0,
        byte_count=encoded_size,
        character_count=characters,
        estimated_tokens=tokens,
        source_kind="text",
    )


def _stats_from_paths(paths: Iterable[Path], source_kind: str) -> CorpusStats:
    files: list[Path] = []
    saw_directory = False
    for path in paths:
        if path.is_dir():
            saw_directory = True
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in _DOCUMENT_EXTENSIONS
            )
        elif path.is_file() and path.suffix.casefold() in _DOCUMENT_EXTENSIONS:
            files.append(path)

    byte_count = 0
    character_count = 0
    estimated_tokens = 0
    unique_files = sorted(set(files), key=lambda item: str(item).casefold())
    for path in unique_files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        byte_count += max(0, size)
        if path.suffix.casefold() in _TEXT_EXTENSIONS:
            characters = _count_text_characters(path)
            character_count += characters
            estimated_tokens += _estimate_text_tokens(characters, size)
        # A compressed/image-heavy PDF size has no stable relationship with
        # its text-token count. Raw PDFs therefore contribute bytes/documents
        # only; the profile is recomputed from canonical extracted text after
        # ingestion instead of manufacturing a misleading token estimate.
    kind = "directory" if saw_directory and source_kind == "path" else source_kind
    return CorpusStats(
        document_count=len(unique_files),
        byte_count=byte_count,
        character_count=character_count,
        estimated_tokens=estimated_tokens,
        source_kind=kind,
    )


def _count_text_characters(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
    except OSError:
        return 0
    return count


def _estimate_text_tokens(characters: int, byte_count: int) -> int:
    if not characters and not byte_count:
        return 0
    # Four characters/token is a conservative language-agnostic approximation
    # for sizing. The real tokenizer remains the authority during training.
    basis = characters or byte_count
    return max(1, math.ceil(basis / 4))


def _existing_path(value: str) -> Path | None:
    if not value or "\n" in value or "\r" in value or len(value) > 512:
        return None
    try:
        candidate = Path(value)
        return candidate if candidate.exists() else None
    except OSError:
        return None


def _host_memory_budget(
    hardware: HardwareInfo,
    env: Mapping[str, str],
    warnings: list[str],
) -> int:
    total = hardware.total_ram_bytes or 8 * GIB
    available = hardware.available_ram_bytes or total
    default_cap = min(
        DEFAULT_MAX_HOST_MEMORY_BUDGET_BYTES,
        max(1, int(total * 0.45)),
        max(1, int(available * 0.70)),
    )
    default = min(default_cap, max(1, int(total * 0.40)))
    requested_gib = _env_float(env, HOST_MEMORY_ENV, None, warnings)
    if requested_gib is None:
        return default

    # An explicit ceiling must not shrink simply because this process already
    # holds its corpus or another application is temporarily active. It does
    # not allocate memory, so report current pressure instead of treating free
    # RAM as the process's eventual maximum. Installed/UMA memory is not added
    # to OS-visible total RAM, and the accelerator ceiling stays independent.
    reserve = max(MIN_HOST_MEMORY_RESERVE_BYTES, math.ceil(total * 0.20))
    hard_cap = min(MAX_HOST_MEMORY_BUDGET_BYTES, max(1, total - reserve))
    # Compare in GiB before multiplying: a finite input such as 1e300 must
    # clamp normally instead of overflowing during conversion to bytes.
    if requested_gib > hard_cap / GIB:
        warnings.append(f"Clamped {HOST_MEMORY_ENV} to the detected safe memory cap.")
        requested = hard_cap
    else:
        requested = max(1, int(requested_gib * GIB))
    if requested > available:
        warnings.append(
            f"{HOST_MEMORY_ENV} exceeds currently available RAM; this is a "
            "planning ceiling, not a reservation or enforced process RSS limit. "
            "Other applications and shared GPU allocations reduce usable RAM."
        )
    return requested


def _accelerator_memory_budget(
    hardware: HardwareInfo,
    device: DeviceDescriptor,
    env: Mapping[str, str],
    warnings: list[str],
) -> int:
    if not device.accelerated:
        if env.get(ACCELERATOR_MEMORY_ENV):
            warnings.append(
                f"Ignored {ACCELERATOR_MEMORY_ENV} because the selected backend is CPU."
            )
        return 0
    dedicated = hardware.gpu_dedicated_memory_bytes
    if dedicated:
        hard_cap = min(MAX_ACCELERATOR_MEMORY_BUDGET_BYTES, int(dedicated * 0.75))
        default = min(hard_cap, int(dedicated * 0.60))
    else:
        total = hardware.total_ram_bytes or 8 * GIB
        hard_cap = min(4 * GIB, int(total * 0.20))
        default = min(hard_cap, int(total * 0.15))
    hard_cap = max(1, hard_cap)
    default = max(1, default)
    requested_gib = _env_float(env, ACCELERATOR_MEMORY_ENV, None, warnings)
    if requested_gib is None:
        return default
    requested = max(1, int(requested_gib * GIB))
    if requested > hard_cap:
        warnings.append(
            f"Clamped {ACCELERATOR_MEMORY_ENV} to the detected safe accelerator cap."
        )
    return min(requested, hard_cap)


def _choose_model_tier(
    tokens: int, host_budget: int, accelerator_budget: int
) -> str:
    if tokens >= 1_000_000 and host_budget >= 8 * GIB and accelerator_budget >= 3 * GIB:
        return "medium"
    if tokens >= 100_000 and host_budget >= 4 * GIB:
        return "small"
    return "compact"


def _model_shape(tier: str, tokens: int) -> dict[str, int]:
    vocabulary = 2_000 if tokens < 50_000 else 4_000 if tokens < 500_000 else 8_000
    shapes = {
        "compact": (128, 4, 4, 512),
        "small": (192, 6, 6, 768),
        "medium": (256, 8, 8, 1_024),
    }
    embedding, heads, layers, feed_forward = shapes[tier]
    return {
        "vocabulary_size": vocabulary,
        "embedding_dimension": embedding,
        "attention_heads": heads,
        "transformer_layers": layers,
        "feed_forward_dimension": feed_forward,
    }


def _context_length(
    tokens: int, env: Mapping[str, str], warnings: list[str]
) -> int:
    default = 128 if tokens < 100_000 else 256 if tokens < 1_000_000 else 512
    requested = _env_int(env, CONTEXT_LENGTH_ENV, default, warnings)
    bounded = min(MAX_CONTEXT_LENGTH, max(64, requested))
    bounded = max(64, (bounded // 32) * 32)
    if bounded != requested:
        warnings.append(
            f"Clamped {CONTEXT_LENGTH_ENV} to 64..{MAX_CONTEXT_LENGTH} in multiples of 32."
        )
    return bounded


def _safe_batch_cap(
    device: DeviceDescriptor,
    accelerator_budget: int,
    context: int,
    tier: str,
) -> int:
    if not device.accelerated:
        base = 8 if context <= 128 and tier == "compact" else 4
    elif accelerator_budget >= 4 * GIB:
        base = 16
    elif accelerator_budget >= 2 * GIB:
        base = 8
    else:
        base = 4
    if context >= 512:
        base = max(1, base // 2)
    if tier == "medium":
        base = max(1, base // 2)
    return base


def _batch_size(
    device: DeviceDescriptor,
    accelerator_budget: int,
    context: int,
    tier: str,
    safe_cap: int,
    env: Mapping[str, str],
    warnings: list[str],
) -> int:
    del context, tier
    if not device.accelerated:
        default = min(4, safe_cap)
    elif accelerator_budget >= 4 * GIB:
        default = min(8, safe_cap)
    elif accelerator_budget >= 2 * GIB:
        default = min(4, safe_cap)
    else:
        default = min(2, safe_cap)
    requested = _env_int(env, BATCH_SIZE_ENV, default, warnings)
    bounded = min(safe_cap, max(1, requested))
    if bounded != requested:
        warnings.append(f"Clamped {BATCH_SIZE_ENV} to the hardware-safe cap {safe_cap}.")
    return bounded


def _gradient_accumulation(
    tokens: int,
    batch_size: int,
    env: Mapping[str, str],
    warnings: list[str],
) -> int:
    target_effective_batch = 8 if tokens < 100_000 else 16 if tokens < 1_000_000 else 32
    default = max(1, math.ceil(target_effective_batch / batch_size))
    requested = _env_int(env, GRADIENT_ACCUMULATION_ENV, default, warnings)
    bounded = min(MAX_GRADIENT_ACCUMULATION_STEPS, max(1, requested))
    if bounded != requested:
        warnings.append(
            f"Clamped {GRADIENT_ACCUMULATION_ENV} to 1..{MAX_GRADIENT_ACCUMULATION_STEPS}."
        )
    return bounded


def _training_steps(
    tokens: int,
    context: int,
    batch_size: int,
    accumulation: int,
    env: Mapping[str, str],
    warnings: list[str],
) -> int:
    sequences = max(1, math.ceil(max(tokens, context) / context))
    effective_batch = max(1, batch_size * accumulation)
    epochs = 12 if tokens < 100_000 else 6 if tokens < 1_000_000 else 3
    default = min(5_000, max(50, math.ceil(sequences / effective_batch) * epochs))
    requested = _env_int(env, MAX_STEPS_ENV, default, warnings)
    bounded = min(MAX_TRAINING_STEPS, max(1, requested))
    if bounded != requested:
        warnings.append(f"Clamped {MAX_STEPS_ENV} to 1..{MAX_TRAINING_STEPS}.")
    return bounded


def _thread_plan(
    hardware: HardwareInfo,
    env: Mapping[str, str],
    warnings: list[str],
) -> tuple[int, int, int]:
    logical = max(1, hardware.logical_cpu_cores)
    physical = min(logical, max(1, hardware.physical_cpu_cores))
    cpu_threads = _bounded_int_env(
        env, CPU_THREADS_ENV, physical, 1, logical, warnings
    )
    interop = _bounded_int_env(
        env,
        INTEROP_THREADS_ENV,
        min(2, max(1, physical // 3)),
        1,
        min(4, physical),
        warnings,
    )
    worker_cap = min(4, max(0, physical - 1))
    default_workers = 0 if hardware.system.casefold() == "windows" else min(2, worker_cap)
    workers = _bounded_int_env(
        env,
        DATALOADER_WORKERS_ENV,
        default_workers,
        0,
        worker_cap,
        warnings,
    )
    return cpu_threads, interop, workers


def _estimate_parameters(
    vocabulary: int, embedding: int, layers: int, feed_forward: int
) -> int:
    # JarvisTransformer ties lm_head.weight to token_embedding.weight.
    embeddings_and_head = vocabulary * embedding
    per_layer = 4 * embedding * embedding + 2 * embedding * feed_forward + 4 * embedding
    final_norm = 2 * embedding
    return embeddings_and_head + layers * per_layer + final_norm


def _bounded_int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    warnings: list[str],
) -> int:
    requested = _env_int(env, name, default, warnings)
    bounded = min(maximum, max(minimum, requested))
    if bounded != requested:
        warnings.append(f"Clamped {name} to the safe range {minimum}..{maximum}.")
    return bounded


def _env_int(
    env: Mapping[str, str], name: str, default: int, warnings: list[str]
) -> int:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        warnings.append(f"Ignored invalid integer in {name}.")
        return default


def _env_float(
    env: Mapping[str, str],
    name: str,
    default: float | None,
    warnings: list[str],
) -> float | None:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        warnings.append(f"Ignored invalid number in {name}.")
        return default
    if not math.isfinite(value) or value <= 0:
        warnings.append(f"Ignored non-positive number in {name}.")
        return default
    return value


__all__ = [
    "ACCELERATOR_MEMORY_ENV",
    "BATCH_SIZE_ENV",
    "CONTEXT_LENGTH_ENV",
    "CorpusStats",
    "DATALOADER_WORKERS_ENV",
    "GRADIENT_ACCUMULATION_ENV",
    "HOST_MEMORY_ENV",
    "MAX_ACCELERATOR_MEMORY_BUDGET_BYTES",
    "MAX_CONTEXT_LENGTH",
    "MAX_HOST_MEMORY_BUDGET_BYTES",
    "MAX_STEPS_ENV",
    "TrainingResourceProfile",
    "build_training_profile",
    "detect_corpus_stats",
]
