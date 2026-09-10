"""
brain.py — J.A.R.V.I.S Central Intelligence
Orchestrates the neural model, RAG, PDF processor and system monitor.
"""

import re
import gc
import json
import asyncio
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

import torch

from jarvis_localhost.ai.language_model import JarvisConfig, JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.dataset import validate_sampling_payload
from jarvis_localhost.ai.trainer import (
    JarvisTrainer,
    TrainConfig,
    TrainingCancelled,
    load_trainer_state,
    validate_trainer_checkpoint,
)
from jarvis_localhost.corpus.manifest import load_authorized_chunks
from jarvis_localhost.corpus.provenance import (
    CanonicalChunk,
    corpus_sha256,
    sha256_file,
)
from jarvis_localhost.curiosity.engine import CuriosityEngine
from jarvis_localhost.hardware import (
    build_training_profile,
    configure_cpu_threads,
    detect_corpus_stats,
    detect_hardware,
    select_compute_device,
)
from jarvis_localhost.paths import (
    CORPUS_ROOT,
    CURIOSITY_ROOT,
    MODELS_ROOT,
    ensure_runtime_directories,
)
from jarvis_localhost.rag.engine import RAGEngine, RAGMode
from jarvis_localhost.retrieval.contrastive import ContrastiveTrainer
from jarvis_localhost.retrieval.encoder import RetrieverConfig, RetrieverEncoder
from jarvis_localhost.retrieval.retriever import SovereignRetriever
from jarvis_localhost.retrieval.vector_store import VectorStore
from jarvis_localhost.sovereign import POLICY
from jarvis_localhost.processing.pdf_processor import PDFProcessor
from jarvis_localhost.monitoring.system_monitor import SystemMonitor
from jarvis_localhost.storage.database import JarvisDB
from jarvis_localhost.projects.project_manager import ProjectManager
from jarvis_localhost.integrations.cluster_client import ClusterClient, ClusterError
from jarvis_localhost.integrations.local_voice import LocalVoice
from jarvis_localhost.integrations.process_lock import InterProcessFileLock
from jarvis_localhost.logging_config import get_logger


PIPELINE_FORMAT_VERSION = 2

logger = get_logger(__name__)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    """Commit a small JSON control file only after it parses successfully."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _canonical_chunks(corpus_root: Optional[Path] = None) -> List[CanonicalChunk]:
    """Load the exact deterministic snapshot authorized by the manifest."""

    corpus_root = corpus_root or CORPUS_ROOT
    return sorted(
        load_authorized_chunks(corpus_root),
        key=lambda item: (item.document_id, item.ordinal, item.chunk_id),
    )


def _corpus_text(chunks: List[CanonicalChunk]) -> str:
    """Build the exact deterministic text stream used by BPE and the LM."""

    return "\n\n".join(chunk.text for chunk in chunks)


def _find_resumable_training(
    *,
    text_digest: str,
    canonical_digest: str,
    models_root: Optional[Path] = None,
) -> Optional[Tuple[Path, str, Path]]:
    """Look for an interrupted run whose latest checkpoint still matches the
    exact corpus/tokenizer lineage a run starting right now would use.

    Training writes its working state under a ``.training-<run_id>-*``
    staging directory (see ``_run`` below) that is promoted to a permanent
    ``pipeline-*`` directory once the entire multi-stage run finishes.
    Failures and cancellations retain the working directory with whatever
    ``jarvis_<tag>.trainer_state.pt`` sidecar ``JarvisTrainer._checkpoint``
    last wrote successfully.

    Resuming from that sidecar is only safe when the corpus and tokenizer it
    was produced against are byte-identical to the ones this run would use --
    the same lineage guarantee already enforced when loading a finished
    pipeline (``corpus_sha256`` / ``corpus_text_sha256`` below) and when
    constructing a fresh ``JarvisTrainer`` (its own tokenizer/corpus digest
    check). If the corpus changed in between -- a document was added,
    removed, or a ghost document was cleaned up -- this returns ``None`` and
    the caller starts a brand new run exactly as it always has; a stale,
    no-longer-matching checkpoint is never silently reused.

    Returns ``(staging_dir, tag, trainer_state_path)`` for the most advanced
    matching checkpoint found, or ``None`` if nothing is safely resumable.
    """

    root = models_root or MODELS_ROOT
    if not root.is_dir():
        return None

    best: Optional[Tuple[Path, str, Path, int]] = None
    active_dir = None
    allow_pending = True
    try:
        pointer = json.loads((root / "jarvis_pipeline.json").read_text(encoding="utf-8"))
        active_dir = (root / pointer["artifacts"]["language_model"]["path"]).parent.resolve()
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError):
        # Without a readable active pointer a pending marker might belong to
        # an already committed pipeline. Never mutate that directory.
        allow_pending = False
    candidates = [*root.glob(".training-*"), *(
        path for path in root.glob("pipeline-*")
        if allow_pending and (path / ".training-pending.json").is_file()
    )]
    for candidate in sorted(candidates):
        if candidate.resolve() == active_dir:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        tokenizer_path = candidate / "jarvis_tokenizer.json"
        try:
            tokenizer = JarvisTokenizer.load(
                tokenizer_path, expected_corpus_sha256=text_digest
            )
            tokenizer_digest = tokenizer.fingerprint()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for state_path in candidate.glob("jarvis_*.trainer_state.pt"):
            tag = state_path.name[len("jarvis_") : -len(".trainer_state.pt")]
            checkpoint_path = candidate / f"jarvis_{tag}.pt"
            manifest_path = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".manifest.json"
            )
            if not checkpoint_path.is_file() or not manifest_path.is_file():
                continue
            try:
                state = load_trainer_state(state_path)
                if (
                    state.get("corpus_sha256") != text_digest
                    or state.get("canonical_corpus_sha256") != canonical_digest
                    or state.get("tokenizer_sha256") != tokenizer_digest
                ):
                    continue
                manifest = validate_trainer_checkpoint(state_path, state)
                if (
                    manifest.get("initialized_from") != "random"
                    or manifest.get("external_weights") is not False
                ):
                    continue
                step = state["step"]
                max_steps = manifest["training"]["config"]["max_steps"]
                if type(max_steps) is not int or not 0 <= step < max_steps:
                    continue
            except Exception:
                # A sidecar that fails to deserialize (truncated by a crash
                # mid-write, foreign file, etc.) is simply not resumable --
                # never a reason to abort training setup.
                continue
            if best is None or step > best[3]:
                best = (candidate, tag, state_path, step)

    if best is None:
        return None
    staging_dir, tag, state_path, _step = best
    return staging_dir, tag, state_path


def _measurement_summary(values: Dict[str, float]) -> Dict[str, Any]:
    """Compact measured signals for immutable pipeline lineage."""

    numeric = [float(value) for value in values.values()]
    if not numeric:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": sum(numeric) / len(numeric),
    }


def _file_record(path: Path, root: Path) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Pipeline artifact escapes model root: {resolved}")
    return {
        "path": resolved.relative_to(root_resolved).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _resolve_artifact(record: Dict[str, Any], root: Path) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        raise ValueError("Pipeline artifact path must be relative")
    resolved = (root / relative).resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError("Pipeline artifact path escapes the model root")
    if sha256_file(resolved) != record.get("sha256"):
        raise ValueError(f"Pipeline artifact checksum mismatch: {relative}")
    return resolved


# ─── Conversation State ───────────────────────────────────────────────────────

JARVIS_GREETINGS = [
    "Sistemas online, Senhor. Como posso auxiliá-lo?",
    "À disposição, Senhor. Todos os sistemas respondem normalmente.",
    "Prontamente, Senhor. Aguardando suas ordens.",
]

PROJECT_QUESTIONS = {
    "nome":      r"(nome|título|chama|projeto)",
    "tipo":      r"(tipo|categoria|classe|área)",
    "descricao": r"(descrição|descreva|trata-se|sobre)",
}


class ConversationMemory:
    """Stores short-term dialogue context."""
    MAX_HISTORY = 20

    def __init__(self):
        self.turns: List[Dict] = []

    def add(self, role: str, text: str):
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > self.MAX_HISTORY:
            self.turns.pop(0)

    def recent(self, n: int = 6) -> List[Dict]:
        return self.turns[-n:]

    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["text"]
        return ""


# ─── JARVIS Brain ─────────────────────────────────────────────────────────────

class JarvisBrain:
    """
    Central intelligence for J.A.R.V.I.S.
    Manages: model lifecycle, PDF ingestion, RAG, system monitoring, chat.
    """

    MODEL_PATH = MODELS_ROOT / "jarvis_final.pt"
    TOKENIZER_PATH = MODELS_ROOT / "jarvis_tokenizer.json"
    RETRIEVER_PATH = MODELS_ROOT / "jarvis_retriever.pt"
    STORE_PATH = CORPUS_ROOT / "vector_store"
    PIPELINE_MANIFEST_PATH = MODELS_ROOT / "jarvis_pipeline.json"

    def __init__(self, *, start_curiosity: bool = True):
        ensure_runtime_directories()
        self._state_lock = threading.RLock()
        self._training_lock = threading.Lock()
        self._process_training_lock: Optional[InterProcessFileLock] = None
        self._training_cancel = threading.Event()
        self._shutdown_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None

        # Hardware discovery is local and accelerator candidates are accepted
        # only after a tensor smoke test.  The returned torch_device is used by
        # every learned component, including DirectML on supported AMD systems.
        self.hardware = detect_hardware()
        self.device_descriptor = select_compute_device(self.hardware)
        self.thread_settings = configure_cpu_threads(self.hardware)
        self.device = self.device_descriptor.torch_device
        initial_corpus_stats = detect_corpus_stats(CORPUS_ROOT)
        self.training_profile = build_training_profile(
            self.hardware,
            corpus_stats=initial_corpus_stats,
            device=self.device_descriptor,
        )

        self.monitor   = SystemMonitor()
        self.processor = PDFProcessor()
        self.memory    = ConversationMemory()
        self.db        = JarvisDB()
        self.pm        = ProjectManager()
        self.cluster_boot_error: Optional[str] = None
        try:
            self.cluster = ClusterClient.from_env()
        except Exception as e:
            self.cluster = None
            self.cluster_boot_error = str(e)
        self.voice     = LocalVoice.from_env()
        self.session_id = uuid.uuid4().hex[:8]
        self.projects: List[Dict] = self.db.list_projects()

        # Model state
        self.tokenizer: Optional[JarvisTokenizer]  = None
        self.model:     Optional[JarvisTransformer] = None
        self.rag:       Optional[RAGEngine]         = None
        self.store:     VectorStore                 = VectorStore()
        self.retriever_encoder: Optional[RetrieverEncoder] = None
        self.active_corpus_sha256 = ""
        self.active_corpus_text_sha256 = ""
        self.pipeline_error: Optional[str] = None
        self._active_store_prefix: Optional[Path] = None

        self.is_trained    = False
        self.is_training   = False
        self.train_progress: Dict = {}

        # Curiosity engine — autonomous document analysis
        self.curiosity_callback: Optional[Callable] = None
        self.curiosity = CuriosityEngine(
            corpus_dir = CORPUS_ROOT,
            output_dir = CURIOSITY_ROOT,
            on_insight = self._on_new_insight,
            device = self.device,
        )

        # Deterministic responses for special intents
        # (used before/after model is trained)
        self._intent_rules = self._build_intent_rules()

        self._try_load_existing()
        # Start curiosity engine after boot. It only reads local corpora and
        # surfaces insights; it does not execute commands or change code.
        if start_curiosity:
            self.curiosity.start()
        logger.info(
            "[Brain] J.A.R.V.I.S. online (%s, RAG=%s).",
            self.device_descriptor.backend,
            self._rag_mode().value,
        )

    def shutdown(self) -> None:
        """Stop background workers before the FastAPI process exits."""
        self._shutdown_event.set()
        self._training_cancel.set()
        thread = self._training_thread
        if thread and thread.is_alive():
            thread.join(timeout=10.0)
        if getattr(self, "curiosity", None):
            self.curiosity.stop()

    # ─── Boot ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _rag_mode() -> RAGMode:
        configured = os.getenv("JARVIS_RAG_MODE", RAGMode.STRICT.value)
        try:
            return RAGMode(configured.strip().casefold())
        except ValueError:
            return RAGMode.STRICT

    def _build_lexical_rag(
        self,
        chunks: List[CanonicalChunk],
        tokenizer: Optional[JarvisTokenizer] = None,
    ) -> None:
        """Keep canonical strict retrieval available without learned weights."""

        runtime_tokenizer = tokenizer or JarvisTokenizer(vocab_size=512)
        store = VectorStore()
        retriever = SovereignRetriever(
            runtime_tokenizer,
            encoder=None,
            store=store,
            device="cpu",
        )
        retriever.store = store
        retriever.index(chunks)
        self.store = store
        self.rag = RAGEngine(
            None,
            runtime_tokenizer,
            store,
            device="cpu",
            retriever=retriever,
            mode=RAGMode.STRICT,
        )

    def _try_load_existing(self):
        """Load only a checksum-verified pipeline for the active corpus."""

        chunks = _canonical_chunks()
        if not self.PIPELINE_MANIFEST_PATH.is_file():
            if chunks:
                self._build_lexical_rag(chunks)
            return

        try:
            manifest = json.loads(
                self.PIPELINE_MANIFEST_PATH.read_text(encoding="utf-8")
            )
            if int(manifest.get("format_version", 0)) != PIPELINE_FORMAT_VERSION:
                raise ValueError("unsupported pipeline manifest version")
            if manifest.get("initialized_from") != "random":
                raise ValueError("pipeline was not initialized from random weights")
            if manifest.get("external_weights") is not False:
                raise ValueError("pipeline declares external weights")
            if not chunks:
                raise ValueError("pipeline exists but canonical corpus is empty")

            canonical_digest = corpus_sha256(chunks)
            full_corpus = _corpus_text(chunks)
            text_digest = JarvisTokenizer.corpus_digest(full_corpus)
            if manifest.get("corpus_sha256") != canonical_digest:
                raise ValueError("active canonical corpus does not match pipeline")
            if manifest.get("corpus_text_sha256") != text_digest:
                raise ValueError("active corpus text does not match pipeline")

            artifacts = manifest.get("artifacts") or {}
            tokenizer_path = _resolve_artifact(artifacts["tokenizer"], MODELS_ROOT)
            model_path = _resolve_artifact(artifacts["language_model"], MODELS_ROOT)
            _resolve_artifact(artifacts["language_model_lineage"], MODELS_ROOT)
            retriever_path = _resolve_artifact(artifacts["retriever"], MODELS_ROOT)
            _resolve_artifact(artifacts["retriever_lineage"], MODELS_ROOT)
            vector_path = _resolve_artifact(artifacts["vectors"], MODELS_ROOT)
            _resolve_artifact(artifacts["vector_metadata"], MODELS_ROOT)
            _resolve_artifact(artifacts["vector_manifest"], MODELS_ROOT)

            tokenizer = JarvisTokenizer.load(
                tokenizer_path,
                expected_corpus_sha256=text_digest,
            )
            tokenizer_fingerprint = tokenizer.fingerprint()
            if manifest.get("tokenizer_sha256") != tokenizer_fingerprint:
                raise ValueError("tokenizer fingerprint does not match pipeline")
            model = JarvisTransformer.load(
                model_path,
                device=self.device,
                expected_corpus_sha256=canonical_digest,
                expected_tokenizer_sha256=tokenizer_fingerprint,
            )
            encoder = RetrieverEncoder.load(
                retriever_path,
                device=self.device,
                expected_corpus_sha256=canonical_digest,
                expected_tokenizer_sha256=tokenizer_fingerprint,
            )
            if not encoder.trained_on_corpus:
                raise ValueError("retriever checkpoint is not corpus-trained")

            # ``vector_path`` is used to derive the shared prefix; its own
            # manifest validates both arrays before state becomes visible.
            suffix = "_vecs.npy"
            if not vector_path.name.endswith(suffix):
                raise ValueError("invalid vector artifact name")
            store_prefix = vector_path.with_name(vector_path.name[: -len(suffix)])
            store = VectorStore()
            store.load(
                store_prefix,
                expected_corpus_sha256=canonical_digest,
                expected_tokenizer_sha256=tokenizer_fingerprint,
                expected_encoder_sha256=sha256_file(retriever_path),
            )
            expected_ids = {chunk.chunk_id for chunk in chunks}
            actual_ids = {str(item.get("chunk_id", "")) for item in store.metadata}
            if actual_ids != expected_ids:
                raise ValueError("vector index chunk lineage does not match corpus")

            retriever = SovereignRetriever(
                tokenizer,
                encoder=None,
                store=store,
                device=self.device,
            )
            retriever.index(chunks)
            retriever.encoder = encoder
            rag = RAGEngine(
                model,
                tokenizer,
                store,
                device=self.device,
                retriever=retriever,
                mode=self._rag_mode(),
            )
            with self._state_lock:
                self.tokenizer = tokenizer
                self.model = model
                self.retriever_encoder = encoder
                self.store = store
                self.rag = rag
                self.active_corpus_sha256 = canonical_digest
                self.active_corpus_text_sha256 = text_digest
                self._active_store_prefix = store_prefix
                self.is_trained = True
                self.pipeline_error = None
            logger.info("[Brain] Pipeline soberano validado. RAG ativo.")
        except Exception as exc:
            self.pipeline_error = str(exc)
            self.is_trained = False
            if chunks:
                self._build_lexical_rag(chunks)
            logger.warning("[Brain] Pipeline recusado com segurança: %s", exc)

    # ─── Intent Rules ─────────────────────────────────────────────────────────

    def _build_intent_rules(self):
        return [
            # Wake / greeting
            {
                "pattern": r"jarvis.*(acord|acor|awake|online|ai\b)",
                "response": "Para o senhor sempre.",
                "intent":  "wake",
            },
            # Status
            {
                "pattern": r"\b(status|relat[oó]rio|diagnóstico|como (está|vai))\b",
                "response": None,   # dynamic
                "intent":  "status",
            },
            # New project
            {
                "pattern": r"(quero criar|criar|novo).*(projeto|arquivo|file)",
                "response": None,
                "intent":  "new_project",
            },
            # Train model
            {
                "pattern": r"(treinar|trein|train|aprender|learn).*(model|ia|neural|jarvis)",
                "response": None,
                "intent":  "train",
            },
            # Curiosity / autonomous learning
            {
                "pattern": r"(curiosidade|curioso|insight|descobriu|aprendeu|tau)",
                "response": None,
                "intent": "curiosity",
            },
            # Optional local/LAN cluster offload
            {
                "pattern": r"(cluster|aether|worker|workers|offload|processamento pesado|gpt local)",
                "response": None,
                "intent": "cluster",
            },
            # Offline voice/TTS status
            {
                "pattern": r"(voz|falar|fala|tts|wake word|microfone)",
                "response": None,
                "intent": "voice",
            },
            # System info
            {
                "pattern": r"(cpu|memória|mem[oó]ria|disco|rede|temperatura|sistema|hardware)",
                "response": None,
                "intent":  "system_info",
            },
            # Time
            {
                "pattern": r"\b(hora|que horas|time)\b",
                "response": None,
                "intent":  "time",
            },
            # Thank you
            {
                "pattern": r"\b(obrigad[ao]|valeu|thanks|thank you)\b",
                "response": "É sempre um prazer servir, Senhor.",
                "intent":  "thanks",
            },
        ]

    def _detect_intent(self, text: str) -> str:
        lower = text.lower()
        for rule in self._intent_rules:
            if re.search(rule["pattern"], lower):
                return rule["intent"]
        return "chat"

    # ─── Chat ─────────────────────────────────────────────────────────────────

    async def chat(self, user_text: str) -> Dict[str, Any]:
        """
        Main chat entry point. Returns response dict with:
        text, intent, action, data.
        """
        self.memory.add("user", user_text)
        self.db.save_message("user", user_text, self.session_id, intent=None)
        intent   = self._detect_intent(user_text)
        response = await self._handle_intent(intent, user_text)
        self.memory.add("jarvis", response["text"])
        self.db.save_message("jarvis", response["text"], self.session_id,
                             intent=intent,
                             sources=response.get("data", {}).get("sources"))
        return response

    async def _handle_intent(self, intent: str, text: str) -> Dict:
        # Fixed-response intents
        for rule in self._intent_rules:
            if rule["intent"] == intent and rule["response"]:
                return {"text": rule["response"], "intent": intent, "action": None, "data": {}}

        # Dynamic intents
        if intent == "status":
            return self._handle_status()

        elif intent == "new_project":
            return {
                "text":   "Claro, Senhor. Ativei o formulário de novo projeto no painel lateral. "
                          "Preencha os dados e registrarei imediatamente nos arquivos da Stark Industries.",
                "intent": intent,
                "action": "open_project_form",
                "data":   {},
            }

        elif intent == "train":
            return {
                "text":   "Entendido, Senhor. Para iniciar o treinamento do modelo neural, "
                          "faça upload de documentos PDF primeiro usando o painel de documentos. "
                          "Quando pronto, pressione 'Iniciar Treinamento'.",
                "intent": intent,
                "action": "show_train_panel",
                "data":   {"is_trained": self.is_trained},
            }

        elif intent == "curiosity":
            ins = self.get_random_insight()
            if ins:
                return {
                    "text": (
                        "Encontrei um ponto curioso nos documentos, Senhor: "
                        f"{ins.get('summary', '')}"
                    ),
                    "intent": intent,
                    "action": "show_curiosity",
                    "data": {"insight": ins},
                }
            return {
                "text": (
                    "Meu motor de curiosidade está ativo, mas ainda não há insights. "
                    "Carregue PDFs ou aguarde o próximo ciclo de análise local."
                ),
                "intent": intent,
                "action": "show_curiosity",
                "data": {},
            }

        elif intent == "cluster":
            return self._handle_cluster_status()

        elif intent == "voice":
            status = self.get_voice_status()
            enabled = "ativada" if status.get("enabled") else "desativada"
            active = "em escuta lógica" if status.get("active") else "em espera"
            return {
                "text": (
                    f"Voz local {enabled}, Senhor. Wake word: "
                    f"{status.get('wake_word')}. Estado: {active}. "
                    "Nenhum reconhecimento por API externa é usado."
                ),
                "intent": intent,
                "action": None,
                "data": status,
            }

        elif intent == "system_info":
            snap = self.monitor.snapshot()
            cpu  = snap["cpu"]
            mem  = snap["memory"]
            disk = snap["disk"]
            reply = (
                f"Relatório do sistema, Senhor: "
                f"CPU em {cpu['percent']}% a {cpu['freq_mhz']} MHz, "
                f"memória {mem['percent']}% utilizada "
                f"({mem['used_gb']}GB de {mem['total_gb']}GB), "
                f"disco com {disk['percent']}% de uso. "
            )
            if cpu.get("temperature"):
                reply += f"Temperatura da CPU: {cpu['temperature']}°C. "
            alerts = self.monitor.check_alerts()
            if alerts:
                reply += " ⚠ Alertas: " + "; ".join(alerts)
            else:
                reply += "Todos os parâmetros dentro do normal."
            return {"text": reply, "intent": intent, "action": "update_metrics",
                    "data": snap}

        elif intent == "time":
            import datetime
            now = datetime.datetime.now().strftime("%H:%M:%S")
            return {"text": f"São exatamente {now}, Senhor.", "intent": intent,
                    "action": None, "data": {}}

        elif intent == "chat":
            return await self._rag_or_fallback(text)

        return {"text": "Processando, Senhor…", "intent": intent, "action": None, "data": {}}

    def _handle_status(self) -> Dict:
        snap = self.monitor.snapshot()
        cpu  = snap["cpu"]
        mem  = snap["memory"]
        trained_str = "Modelo neural ativo e treinado." if self.is_trained \
                      else "Modelo neural aguardando treinamento."
        indexed_chunks = self._indexed_chunk_count()
        docs_str = f"{indexed_chunks} chunks canônicos indexados." if indexed_chunks > 0 \
                   else "Nenhum documento indexado ainda."
        text = (
            f"Status geral: sistemas operacionais. "
            f"CPU {cpu['percent']}%, RAM {mem['percent']}%. "
            f"{trained_str} {docs_str} "
            f"Backend: {self.device_descriptor.backend}. "
            f"RAG: {self.rag.mode.value if self.rag else self._rag_mode().value}. "
            f"Plataforma: {snap['platform']}. "
            f"Uptime: {snap['uptime_hours']:.1f}h."
        )
        data = {
            **snap,
            "hardware": self.hardware.to_dict(),
            "compute": self.device_descriptor.to_dict(),
            "training_profile": self.training_profile.to_dict(),
            "pipeline_error": self.pipeline_error,
            "active_corpus_sha256": self.active_corpus_sha256,
            "indexed_chunks": indexed_chunks,
        }
        return {"text": text, "intent": "status", "action": "update_metrics",
                "data": data}

    def _handle_cluster_status(self) -> Dict:
        snap = self.get_cluster_snapshot()
        status = snap.get("status", {})
        workers = snap.get("workers", [])
        if self.cluster_boot_error:
            text = f"Conector Aether bloqueado na inicialização: {self.cluster_boot_error}"
        elif not status.get("enabled"):
            text = (
                "Offload Aether está desativado. O Jarvis continua 100% local nesta "
                "máquina. Para liberar workers locais/LAN, configure "
                "JARVIS_CLUSTER_ENABLED=1 e JARVIS_CLUSTER_URL."
            )
        elif status.get("error"):
            text = f"Cluster Aether configurado, mas indisponível: {status['error']}"
        else:
            cluster = status.get("cluster", {})
            text = (
                "Cluster Aether conectado. "
                f"Workers online: {cluster.get('workers_online', len(workers))}; "
                f"tarefas em fila: {cluster.get('tasks_queued', 0)}. "
                "Offload permitido apenas por tarefas explícitas e allowlist local."
            )
        return {
            "text": text,
            "intent": "cluster",
            "action": "show_cluster",
            "data": snap,
        }

    async def _rag_or_fallback(self, text: str) -> Dict:
        """Answer from canonical evidence; strict extractive mode is default."""
        if self.rag and self._indexed_chunk_count() > 0:
            result = await asyncio.to_thread(
                self.rag.answer,
                text,
                3,
                80,
            )
            return {
                "text":   result["answer"],
                "intent": "chat",
                "action": "show_sources" if result["sources"] else None,
                "data":   {
                    "sources": result["sources"],
                    "mode": result.get("mode", RAGMode.STRICT.value),
                    "method": result.get("method"),
                    "grounding": result.get("grounding"),
                    "abstained": result.get("abstained", False),
                    "retrieval_confidence": result.get("retrieval_confidence", 0.0),
                    "reason": result.get("reason", ""),
                },
            }

        # Fallback responses before model is trained
        fallbacks = [
            "Entendido, Senhor. Processando sua solicitação.",
            "Informação registrada. Posso ajudar com mais alguma coisa?",
            "Analisando, Senhor. Quando o modelo neural estiver treinado, "
            "poderei responder com base nos documentos carregados.",
            "Compreendido. Faça upload de documentos PDF para que eu possa "
            "aprender e responder com mais precisão.",
        ]
        import random
        return {"text": random.choice(fallbacks), "intent": "chat",
                "action": None, "data": {}}

    def _indexed_chunk_count(self) -> int:
        retriever = getattr(self.rag, "retriever", None) if self.rag else None
        chunks = getattr(retriever, "chunks", None)
        return len(chunks) if chunks is not None else len(self.store)

    # ─── PDF Processing & Training ────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a PDF and expose its canonical, page-level evidence."""

        result = self.processor.process(pdf_path)
        doc_id = self.db.save_document(
            filename=result.filename,
            path=pdf_path,
            stats=result.stats,
            corpus=result.training_corpus,
            document_id=result.document_id,
            document_sha256=result.document_sha256,
            corpus_sha256=corpus_sha256(result.canonical_chunks),
            chunks_file=result.chunks_path,
            canonical_chunks=len(result.canonical_chunks),
        )

        # Re-read the committed JSONL files rather than trusting transient PDF
        # objects. This guarantees that retrieval and later training see the
        # exact same SHA/page/bbox records.
        chunks = _canonical_chunks()
        current_digest = corpus_sha256(chunks) if chunks else ""
        with self._state_lock:
            if self.is_trained and current_digest == self.active_corpus_sha256:
                # Duplicate ingestion: the immutable trained lineage remains
                # current and every stable chunk id is already present.
                indexed = len(result.canonical_chunks)
            else:
                if self.is_trained and current_digest != self.active_corpus_sha256:
                    self.pipeline_error = (
                        "O corpus mudou após o treinamento; os pesos anteriores "
                        "foram desativados até um novo treinamento soberano."
                    )
                self.is_trained = False
                self.tokenizer = None
                self.model = None
                self.retriever_encoder = None
                self.active_corpus_sha256 = ""
                self.active_corpus_text_sha256 = ""
                self._active_store_prefix = None
                self._build_lexical_rag(chunks)
                indexed = len(result.canonical_chunks)
        self.db.mark_indexed(doc_id)
        return {
            **result.stats,
            "indexed_chunks": indexed,
            "filename": result.filename,
            "doc_id": doc_id,
            "document_id": result.document_id,
            "document_sha256": result.document_sha256,
            "corpus_sha256": current_digest,
            "citations_ready": bool(indexed),
            "requires_retraining": not self.is_trained,
        }

    def _activate_pipeline(
        self,
        *,
        tokenizer: JarvisTokenizer,
        model: JarvisTransformer,
        encoder: RetrieverEncoder,
        store: VectorStore,
        chunks: List[CanonicalChunk],
        canonical_digest: str,
        text_digest: str,
        store_prefix: Path,
        profile: Any,
    ) -> None:
        retriever = SovereignRetriever(
            tokenizer,
            encoder=None,
            store=store,
            device=self.device,
        )
        retriever.store = store
        retriever.index(chunks)
        retriever.encoder = encoder
        rag = RAGEngine(
            model,
            tokenizer,
            store,
            device=self.device,
            retriever=retriever,
            mode=self._rag_mode(),
        )
        with self._state_lock:
            self.tokenizer = tokenizer
            self.model = model
            self.retriever_encoder = encoder
            self.store = store
            self.rag = rag
            self.active_corpus_sha256 = canonical_digest
            self.active_corpus_text_sha256 = text_digest
            self._active_store_prefix = store_prefix
            self.training_profile = profile
            self.pipeline_error = None
            self.is_trained = True

    def _pipeline_manifest(
        self,
        *,
        artifact_dir: Path,
        canonical_digest: str,
        text_digest: str,
        tokenizer: JarvisTokenizer,
        profile: Any,
        lm_history: List[Dict],
        retriever_history: List[Dict],
        curriculum_lineage: Optional[Dict[str, Any]] = None,
        model_filename: str = "jarvis_final.pt",
    ) -> Dict[str, Any]:
        model_path = artifact_dir / model_filename
        retriever_path = artifact_dir / "jarvis_retriever.pt"
        vector_prefix = artifact_dir / "vector_store"
        vector_path, metadata_path, vector_manifest_path = VectorStore._paths(
            vector_prefix
        )
        return {
            "format_version": PIPELINE_FORMAT_VERSION,
            "pipeline": "jarvis-sovereign-pdf-rag",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "initialized_from": "random",
            "external_weights": False,
            "runtime_downloads_allowed": False,
            "corpus_sha256": canonical_digest,
            "corpus_text_sha256": text_digest,
            "tokenizer_sha256": tokenizer.fingerprint(),
            "rag_default_mode": self._rag_mode().value,
            "hardware": self.hardware.to_dict(),
            "compute": self.device_descriptor.to_dict(),
            "threads": self.thread_settings.to_dict(),
            "training_profile": profile.to_dict(),
            "training": {
                "language_model_events": len(lm_history),
                "retriever_history": retriever_history,
                "gradient_accumulation_steps": (
                    profile.gradient_accumulation_steps
                ),
                "dataloader_workers": profile.dataloader_workers,
                "curiosity_curriculum": dict(curriculum_lineage or {}),
            },
            "sovereign_policy": POLICY.to_manifest(),
            "artifacts": {
                "tokenizer": _file_record(
                    artifact_dir / "jarvis_tokenizer.json", MODELS_ROOT
                ),
                "language_model": _file_record(model_path, MODELS_ROOT),
                "language_model_lineage": _file_record(
                    model_path.with_suffix(model_path.suffix + ".manifest.json"),
                    MODELS_ROOT,
                ),
                "retriever": _file_record(retriever_path, MODELS_ROOT),
                "retriever_lineage": _file_record(
                    retriever_path.with_suffix(
                        retriever_path.suffix + ".manifest.json"
                    ),
                    MODELS_ROOT,
                ),
                "vectors": _file_record(vector_path, MODELS_ROOT),
                "vector_metadata": _file_record(metadata_path, MODELS_ROOT),
                "vector_manifest": _file_record(
                    vector_manifest_path, MODELS_ROOT
                ),
            },
        }

    def start_training(
        self,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ) -> None:
        """Train BPE, LM and contrastive retrieval from one corpus snapshot."""

        if not self._training_lock.acquire(blocking=False):
            return
        process_lock_path = (
            self.PIPELINE_MANIFEST_PATH.parent / ".jarvis-training.lock"
        ).resolve(strict=False)
        process_lock = getattr(self, "_process_training_lock", None)
        if process_lock is None or process_lock.path != process_lock_path:
            process_lock = InterProcessFileLock(process_lock_path)
            self._process_training_lock = process_lock
        if not process_lock.acquire(blocking=False):
            self._training_lock.release()
            with self._state_lock:
                self.train_progress = {
                    "error": "Outro processo Jarvis já está treinando este pipeline.",
                    "percent": 0,
                    "updated_at": time.time(),
                }
            return
        with self._state_lock:
            if self.is_training:
                process_lock.release()
                self._training_lock.release()
                return
            self.is_training = True
            self._training_cancel.clear()
            self.train_progress = {
                "percent": 0,
                "message": "Preparando treinamento soberano.",
            }

        def emit(info: Dict) -> None:
            with self._state_lock:
                self.train_progress = {
                    **self.train_progress,
                    **info,
                    "updated_at": time.time(),
                }
            if progress_callback:
                try:
                    progress_callback(dict(info))
                except Exception as exc:
                    logger.warning("[Brain] Progress callback warning: %s", exc)

        def _run():
            run_id: Optional[str] = None
            lm_history: List[Dict] = []
            retriever_history: List[Dict] = []
            curriculum_lineage: Dict[str, Any] = {"enabled": False}
            staging: Optional[Path] = None
            curiosity = getattr(self, "curiosity", None)
            curiosity_was_running = False
            try:
                curiosity_was_running = bool(
                    curiosity and curiosity.get_stats().get("is_running")
                )
                # The ICM/PPO explorer shares the selected accelerator. Pause
                # it while fitting the primary models to avoid VRAM/UMA races.
                if curiosity_was_running:
                    curiosity.stop()
                run_id = self.db.start_training_run(
                    backend=self.device_descriptor.backend,
                )
                emit({"percent": 2, "message": "Fixando snapshot canônico."})
                chunks = _canonical_chunks()
                if len(chunks) < 2:
                    raise ValueError(
                        "São necessários ao menos dois chunks canônicos para "
                        "treinar o recuperador contrastivo."
                    )
                canonical_digest = corpus_sha256(chunks)
                chunk_ids = [chunk.chunk_id for chunk in chunks]
                full_corpus = _corpus_text(chunks)
                if not full_corpus.strip():
                    raise ValueError("O corpus canônico está vazio.")
                text_digest = JarvisTokenizer.corpus_digest(full_corpus)
                sampling_payload = None
                if curiosity is not None:
                    sampling_payload = curiosity.get_sampling_weights(chunk_ids)
                    curriculum_lineage = {
                        "enabled": True,
                        "input_format_version": sampling_payload.get(
                            "format_version"
                        ),
                        "input_signal_version": sampling_payload.get(
                            "signal_version"
                        ),
                        "weighted_chunks": len(sampling_payload.get("weights", [])),
                    }

                corpus_stats = detect_corpus_stats(full_corpus)
                profile = build_training_profile(
                    self.hardware,
                    corpus_stats=corpus_stats,
                    device=self.device_descriptor,
                )
                self.thread_settings = configure_cpu_threads(self.hardware)
                self.db.update_training_run(
                    run_id,
                    corpus_sha256=canonical_digest,
                    backend=self.device_descriptor.backend,
                    profile=profile.to_dict(),
                )
                emit(
                    {
                        "percent": 5,
                        "message": (
                            f"Perfil {profile.model_tier}: "
                            f"{profile.estimated_parameter_count:,} parâmetros; "
                            f"backend {profile.backend}."
                        ),
                        "profile": profile.to_dict(),
                    }
                )

                # (audit fix, 2026-08-31) Before starting a brand new run,
                # check whether a previous one left behind a checkpoint that
                # is genuinely safe to continue from -- same corpus, same
                # tokenizer, byte for byte. This is what turns a crash mid
                # training back into a resume instead of losing all progress
                # since the run started; see _find_resumable_training's own
                # docstring for exactly what guarantees this does and does
                # not carry over.
                resume_match = _find_resumable_training(
                    text_digest=text_digest, canonical_digest=canonical_digest
                )
                resume_state_path: Optional[Path] = None
                if resume_match is not None:
                    staging, resume_tag, resume_state_path = resume_match
                    emit(
                        {
                            "percent": 8,
                            "message": (
                                "Retomando treino anterior interrompido "
                                f"(checkpoint '{resume_tag}')."
                            ),
                        }
                    )
                    tokenizer_path = staging / "jarvis_tokenizer.json"
                    tokenizer = JarvisTokenizer.load(
                        tokenizer_path, expected_corpus_sha256=text_digest
                    )
                    checkpoint_path = staging / f"jarvis_{resume_tag}.pt"
                    model = JarvisTransformer.load(
                        checkpoint_path,
                        device=self.device,
                        expected_corpus_sha256=text_digest,
                        expected_tokenizer_sha256=tokenizer.fingerprint(),
                    )
                    manifest_path = checkpoint_path.with_suffix(
                        checkpoint_path.suffix + ".manifest.json"
                    )
                    saved_config = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )["training"]["config"]
                    train_cfg = TrainConfig(
                        **{
                            **saved_config,
                            "device": self.device,
                            "checkpoint_dir": str(staging),
                        }
                    )
                else:
                    staging = Path(
                        tempfile.mkdtemp(
                            prefix=f".training-{run_id}-", dir=MODELS_ROOT
                        )
                    )

                    # 1. A fresh BPE vocabulary is learned only from this snapshot.
                    emit({"percent": 8, "message": "Treinando tokenizador BPE do zero."})
                    tokenizer = JarvisTokenizer(vocab_size=profile.vocabulary_size)

                    def _bpe_callback(current_vocab: int, total_vocab: int) -> None:
                        pct = round(8.0 + (current_vocab / total_vocab) * 4.0, 1)
                        emit({
                            "percent": pct,
                            "message": f"Treinando tokenizador BPE: {current_vocab}/{total_vocab} tokens.",
                        })

                    tokenizer.train(full_corpus, progress_callback=_bpe_callback)
                    tokenizer_path = staging / "jarvis_tokenizer.json"
                    tokenizer.save(tokenizer_path)

                    # 2. A profile-sized decoder is initialized from random weights.
                    cfg = JarvisConfig(
                        vocab_size=tokenizer.vocab_actual_size,
                        context_len=profile.context_length,
                        embed_dim=profile.embedding_dimension,
                        num_heads=profile.attention_heads,
                        num_layers=profile.transformer_layers,
                        ff_dim=profile.feed_forward_dimension,
                        dropout=0.1,
                    )
                    model = JarvisTransformer(cfg)
                    train_cfg = TrainConfig(
                        max_steps=profile.max_steps,
                        batch_size=profile.batch_size,
                        gradient_accumulation=(
                            profile.gradient_accumulation_steps
                        ),
                        context_len=profile.context_length,
                        log_interval=max(1, min(25, profile.max_steps // 20 or 1)),
                        eval_interval=profile.evaluation_interval,
                        checkpoint_every=profile.checkpoint_interval,
                        checkpoint_dir=str(staging),
                        num_workers=profile.dataloader_workers,
                        device=self.device,
                    )

                def _train_callback(info: Dict) -> None:
                    lm_history.append(dict(info))
                    # JarvisTrainer.train() emits two different event shapes on
                    # the same callback: a per-log_interval step event that
                    # carries "progress" (fraction of max_steps completed), and
                    # a separate eval_interval-triggered event that carries only
                    # {"step", "val_loss"} with no "progress" key at all. Since
                    # eval_interval is always a multiple of log_interval here,
                    # every evaluation fires right after a step event for the
                    # same step — and if this callback always recomputed
                    # "percent" from info.get("progress", 0.0), the eval event's
                    # missing key silently defaulted to 0.0, snapping the
                    # publicly reported percent back down to the phase's 12%
                    # floor (round(12.0 + 0.0 * 0.63, 2)) immediately after the
                    # correct value had just been published. Real evidence:
                    # live /api/train/status reads during this audit showed
                    # step=750, progress=47.06, loss/tokens_seen/val_loss all
                    # current, yet percent=12.0 instead of the expected ~41.65
                    # — exactly this eval-event stomp, reproducible at every
                    # step % eval_interval == 0. Only recompute "percent" when
                    # this event actually reports a progress fraction; an
                    # eval-only event still updates step/val_loss but leaves
                    # the last real percent untouched via emit()'s dict merge.
                    payload = dict(info)
                    if "progress" in info:
                        payload["percent"] = round(
                            12.0 + float(info["progress"]) * 0.63, 2
                        )
                    payload["message"] = "Treinando modelo de linguagem local."
                    emit(payload)

                # Freeze this run's curriculum so retriever checkpoints remain
                # compatible even if curiosity runs between retries.
                sampling_path = staging / "training_sampling.json"
                if sampling_path.exists():
                    if sampling_path.stat().st_size > 65536 + 256 * len(chunk_ids):
                        raise ValueError("training sampling snapshot exceeds corpus size limit")
                    saved_sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
                    if saved_sampling.get("corpus_sha256") != canonical_digest:
                        raise ValueError("training sampling snapshot belongs to another corpus")
                    sampling_payload = saved_sampling["payload"]
                else:
                    _atomic_json(sampling_path, {"corpus_sha256": canonical_digest, "payload": sampling_payload})
                if sampling_payload is not None:
                    validate_sampling_payload(sampling_payload, expected_corpus_sha256=canonical_digest, expected_chunk_ids=chunk_ids)
                    curriculum_lineage.update({
                        "input_format_version": sampling_payload.get("format_version"),
                        "input_signal_version": sampling_payload.get("signal_version"),
                        "weighted_chunks": len(sampling_payload.get("weights", [])),
                    })

                trainer = JarvisTrainer(
                    model,
                    tokenizer,
                    full_corpus,
                    train_cfg,
                    cancellation_event=self._training_cancel,
                    chunk_ids=chunk_ids,
                    canonical_corpus_sha256=canonical_digest,
                    sampling_payload=sampling_payload,
                )
                if resume_state_path is not None:
                    resumed_step = trainer.apply_resume_state(resume_state_path)
                    lm_history.extend(trainer.history)
                    emit(
                        {
                            "percent": round(12 + 63 * resumed_step / train_cfg.max_steps, 2),
                            "step": resumed_step,
                            "resumed_from_step": resumed_step,
                            "message": (
                                f"Retomando a partir do passo {resumed_step}/"
                                f"{train_cfg.max_steps}."
                            ),
                        }
                    )
                trainer.train(callback=_train_callback)
                trainer.release_training_buffers()
                model.to("cpu")
                del full_corpus
                gc.collect()

                # 3. A separate bidirectional encoder learns adjacent/section
                # positives with symmetric InfoNCE from the same chunks.
                emit(
                    {
                        "percent": 77,
                        "message": "Preparando recuperador contrastivo e verificando checkpoint.",
                    }
                )
                retriever_cfg = RetrieverConfig(
                    vocab_size=tokenizer.vocab_actual_size,
                    context_len=profile.context_length,
                    embed_dim=profile.embedding_dimension,
                    projection_dim=min(128, profile.embedding_dimension),
                    num_heads=profile.attention_heads,
                    num_layers=profile.transformer_layers,
                    ff_dim=profile.feed_forward_dimension,
                    dropout=0.1,
                )
                encoder = RetrieverEncoder(retriever_cfg)
                # Execution-only options keep the model, logical batch, pair
                # order and optimizer intact. The retriever checkpoint records
                # a policy transition because dropout layouts may change.
                fused_pairs = os.getenv("JARVIS_RETRIEVER_FUSED_PAIRS", "0")
                if fused_pairs not in {"0", "1"}:
                    raise ValueError("JARVIS_RETRIEVER_FUSED_PAIRS must be 0 or 1")
                contrastive = ContrastiveTrainer(
                    encoder,
                    tokenizer,
                    chunks,
                    device=self.device,
                    sampling_payload=sampling_payload,
                    padding_policy=os.getenv("JARVIS_RETRIEVER_PADDING", "fixed"),
                    min_bucket_size=int(os.getenv("JARVIS_RETRIEVER_MIN_BUCKET", "32")),
                    fuse_pair_encoding=fused_pairs == "1",
                )
                retriever_epochs = max(
                    1, min(6, (profile.max_steps + 999) // 1000)
                )
                # Both views retain attention activations for backward. Budget
                # those quadratic tensors instead of assuming a 6x LM batch
                # always fits just because retrieval has no vocabulary head.
                activation_bytes_per_pair = (
                    4 * profile.context_length * profile.transformer_layers * 2
                    * (3 * profile.attention_heads * profile.context_length
                       + 12 * profile.embedding_dimension)
                )
                memory_budget = (
                    profile.accelerator_memory_budget_bytes
                    or profile.host_memory_budget_bytes
                )
                estimated_batch = max(2, int(memory_budget * 0.66) // max(1, activation_bytes_per_pair))
                retriever_batch_size = max(2, min(profile.batch_size * 6, estimated_batch))
                fit_path = staging / "retriever_fit.json"
                fit_binding = {"corpus_sha256": canonical_digest, "encoder_config": vars(retriever_cfg)}
                if fit_path.exists():
                    if fit_path.stat().st_size > 65536:
                        raise ValueError("retriever fit configuration exceeds size limit")
                    saved_fit = json.loads(fit_path.read_text(encoding="utf-8"))
                    if saved_fit.get("binding") != fit_binding:
                        raise ValueError("retriever fit configuration belongs to another corpus or architecture")
                    retriever_batch_size = int(saved_fit["batch_size"])
                    retriever_epochs = int(saved_fit["epochs"])
                    if not 2 <= retriever_batch_size <= 1024 or not 1 <= retriever_epochs <= 6:
                        raise ValueError("invalid saved retriever fit configuration")
                else:
                    _atomic_json(fit_path, {"binding": fit_binding, "batch_size": retriever_batch_size, "epochs": retriever_epochs})
                def _retriever_progress(step: int, total: int) -> None:
                    emit(
                        {
                            "percent": round(77 + 10 * step / max(1, total), 2),
                            "message": f"Treinando recuperador contrastivo (lote {step}/{total}).",
                            "retriever_step": step,
                            "retriever_total_batches": total,
                            "retriever_resume": dict(contrastive.resume_metadata),
                        }
                    )

                retriever_history = contrastive.train(
                    epochs=retriever_epochs,
                    batch_size=retriever_batch_size,
                    progress_callback=_retriever_progress,
                    progress_interval=10,
                    cancellation_event=self._training_cancel,
                    checkpoint_dir=staging,
                    checkpoint_interval=300,
                )
                emit({"percent": 87, "message": "Avaliando recuperador em lotes limitados."})
                retrieval_uncertainty_by_chunk = (
                    contrastive.measure_retrieval_uncertainty(
                        batch_size=retriever_batch_size,
                        cancellation_event=self._training_cancel,
                        progress_callback=lambda done, total: emit({
                            "percent": round(87 + done / max(1, total), 2),
                            "message": f"Avaliando recuperador: {done}/{total} trechos.",
                        }),
                    )
                )
                retriever_path = staging / "jarvis_retriever.pt"
                contrastive.save(str(retriever_path), retriever_history)

                # This exact per-chunk evaluation can take hours. Persist it
                # independently, after saving the trained retriever, and free
                # the other model's accelerator allocations while it runs.
                contrastive.optimizer.zero_grad(set_to_none=True)
                contrastive.optimizer.state.clear()
                encoder.to("cpu")
                gc.collect()
                model.to(self.device)
                lm_loss_by_chunk = trainer.measure_chunk_losses(
                    checkpoint_path=staging / "lm_measurements.json",
                    progress_callback=lambda done, total: emit({
                        "percent": round(88 + 2 * done / max(1, total), 2),
                        "message": f"Avaliando modelo de linguagem: {done}/{total} trechos.",
                    }),
                )
                encoder.to(self.device)

                if curiosity is not None:
                    updated_sampling = curiosity.update_learning_signals(
                        lm_loss_by_chunk=lm_loss_by_chunk,
                        retrieval_uncertainty_by_chunk=(
                            retrieval_uncertainty_by_chunk
                        ),
                    )
                    curriculum_lineage.update(
                        {
                            "output_signal_version": updated_sampling.get(
                                "signal_version"
                            ),
                            "generation": updated_sampling.get("generation", ""),
                            "lm_loss": _measurement_summary(lm_loss_by_chunk),
                            "retrieval_uncertainty": _measurement_summary(
                                retrieval_uncertainty_by_chunk
                            ),
                        }
                    )

                # 4. Dense vectors retain the complete canonical metadata.
                if self._training_cancel.is_set():
                    raise TrainingCancelled("training cancelled by operator")
                emit({"percent": 90, "message": "Indexando evidências canônicas."})
                store = VectorStore()
                retriever = SovereignRetriever(
                    tokenizer,
                    encoder=encoder,
                    store=store,
                    device=self.device,
                )
                retriever.store = store
                retriever.index(chunks, batch_size=profile.batch_size)
                vector_prefix = staging / "vector_store"
                store.save(
                    vector_prefix,
                    corpus_sha256=canonical_digest,
                    tokenizer_sha256=tokenizer.fingerprint(),
                    encoder_sha256=sha256_file(retriever_path),
                )

                # Keep the trainer's text-digest checkpoint and optimizer pair
                # intact until publication, including failures in final checks.
                model_path = staging / "jarvis_inference.pt"
                final_loss = next(
                    (
                        float(item["loss"])
                        for item in reversed(lm_history)
                        if "loss" in item
                    ),
                    None,
                )
                model.save(
                    model_path,
                    corpus_sha256=canonical_digest,
                    tokenizer_sha256=tokenizer.fingerprint(),
                    training={
                        "objective": "causal-language-modeling",
                        "steps": trainer.step + 1,
                        "tokens_seen": trainer.tokens_seen,
                        "final_loss": final_loss,
                        "gradient_accumulation_steps": (
                            profile.gradient_accumulation_steps
                        ),
                        "dataloader_workers": profile.dataloader_workers,
                    },
                )

                # The corpus may be ingested while training. Never publish
                # weights for a stale snapshot.
                current_chunks = _canonical_chunks()
                if corpus_sha256(current_chunks) != canonical_digest:
                    raise TrainingCancelled(
                        "O corpus mudou durante o treinamento; o resultado "
                        "obsoleto não foi ativado."
                    )

                final_dir = MODELS_ROOT / (
                    f"pipeline-{canonical_digest[:16]}-{run_id}"
                )
                if final_dir.exists():
                    final_dir = MODELS_ROOT / (
                        f"pipeline-{canonical_digest[:16]}-{run_id}-"
                        f"{uuid.uuid4().hex[:8]}"
                    )
                # Discovery can recover this directory even if the process
                # stops between its rename and the small active-pointer commit.
                _atomic_json(staging / ".training-pending.json", {"format_version": 1})
                os.replace(staging, final_dir)
                staging = final_dir
                final_store_prefix = final_dir / vector_prefix.name
                manifest = self._pipeline_manifest(
                    artifact_dir=final_dir,
                    canonical_digest=canonical_digest,
                    text_digest=text_digest,
                    tokenizer=tokenizer,
                    profile=profile,
                    lm_history=lm_history,
                    retriever_history=retriever_history,
                    curriculum_lineage=curriculum_lineage,
                    model_filename=model_path.name,
                )
                # This tiny pointer is the transaction commit marker. Old
                # versioned pipelines remain intact and recoverable.
                _atomic_json(self.PIPELINE_MANIFEST_PATH, manifest)
                staging = None
                try:
                    (final_dir / ".training-pending.json").unlink(missing_ok=True)
                except OSError:
                    # The active pointer excludes this committed directory
                    # from discovery even if marker cleanup is interrupted.
                    logger.warning("[Brain] Deferred publication marker cleanup.")

                self._activate_pipeline(
                    tokenizer=tokenizer,
                    model=model,
                    encoder=encoder,
                    store=store,
                    chunks=chunks,
                    canonical_digest=canonical_digest,
                    text_digest=text_digest,
                    store_prefix=final_store_prefix,
                    profile=profile,
                )
                database_warning = ""
                try:
                    for document_id in {chunk.document_id for chunk in chunks}:
                        self.db.mark_indexed(document_id)
                    self.db.update_training_run(
                        run_id,
                        finished_at=time.time(),
                        status="done",
                        steps=trainer.step + 1,
                        final_loss=final_loss,
                        vocab_size=tokenizer.vocab_actual_size,
                        corpus_sha256=canonical_digest,
                        tokenizer_sha256=tokenizer.fingerprint(),
                        retriever_loss=(
                            retriever_history[-1]["loss"]
                            if retriever_history
                            else None
                        ),
                        backend=self.device_descriptor.backend,
                        profile=profile.to_dict(),
                        history=lm_history[-200:],
                    )
                except Exception as database_exc:
                    # Artifact activation is already committed and remains the
                    # source of truth. A telemetry failure must not roll it back
                    # or falsely report model-training failure.
                    database_warning = str(database_exc)
                    logger.warning(
                        "[Brain] Training DB finalization warning: %s", database_exc
                    )
                emit(
                    {
                        "done": True,
                        "percent": 100,
                        "message": (
                            "Treinamento soberano concluído; BPE, LM, "
                            "recuperador e índice compartilham a mesma linhagem."
                        ),
                        "corpus_sha256": canonical_digest,
                        "warning": database_warning,
                    }
                )

            except Exception as exc:
                error_message = str(exc).strip() or type(exc).__name__
                logger.exception("[Brain] Training error (%s): %s", type(exc).__name__, error_message)
                if run_id is not None:
                    try:
                        self.db.update_training_run(
                            run_id,
                            finished_at=time.time(),
                            status=(
                                "cancelled"
                                if isinstance(exc, TrainingCancelled)
                                else "error"
                            ),
                            history=lm_history[-200:],
                            error=error_message,
                        )
                    except Exception as database_exc:
                        logger.warning("[Brain] Could not record training failure: %s", database_exc)
                emit({"error": error_message, "error_type": type(exc).__name__, "cancelled": isinstance(exc, TrainingCancelled)})
            finally:
                if staging is not None and staging.exists():
                    # This can contain the only resumable optimizer/model
                    # snapshot. Keep it after handled errors and cancellation.
                    emit({"checkpoint_preserved": True})
                with self._state_lock:
                    self.is_training = False
                process_lock.release()
                self._training_lock.release()
                if (
                    curiosity_was_running
                    and not self._shutdown_event.is_set()
                ):
                    try:
                        curiosity.start()
                    except Exception as curiosity_exc:
                        logger.warning(
                            "[Brain] Curiosity restart warning: %s", curiosity_exc
                        )

        thread = threading.Thread(
            target=_run,
            name="jarvis-sovereign-training",
            daemon=False,
        )
        self._training_thread = thread
        try:
            thread.start()
        except Exception:
            with self._state_lock:
                self.is_training = False
            self._training_thread = None
            process_lock.release()
            self._training_lock.release()
            raise

    # ─── Projects ─────────────────────────────────────────────────────────────

    def save_project(self, project: Dict) -> Dict:
        """Save project to SQLite and generate project files."""
        saved = self.db.save_project(
            name        = project.get("name", "Projeto"),
            type_       = project.get("type", ""),
            priority    = project.get("priority", "BETA"),
            description = project.get("description", ""),
            tags        = project.get("tags", []),
        )
        # Generate project files
        try:
            zip_path = self.pm.generate(saved)
            self.db.update_project(saved["id"], files=[zip_path])
            saved["zip_path"] = zip_path
        except Exception as e:
            logger.error("[Brain] Project file gen error: %s", e)
        self.projects = self.db.list_projects()
        return saved

    def list_projects(self) -> List[Dict]:
        self.projects = self.db.list_projects()
        return self.projects

    def get_project(self, pid: str) -> Optional[Dict]:
        return self.db.get_project(pid)

    def get_chat_history(self, limit: int = 50) -> List[Dict]:
        return self.db.get_history(limit=limit)

    def get_db_stats(self) -> Dict:
        return self.db.get_stats()

    # ─── Optional Cluster / Voice Integrations ───────────────────────────────

    def get_cluster_snapshot(self) -> Dict:
        if self.cluster_boot_error or not self.cluster:
            return {
                "status": {
                    "enabled": False,
                    "error": self.cluster_boot_error or "Cluster não inicializado.",
                },
                "workers": [],
                "tasks": [],
            }
        return self.cluster.snapshot()

    def get_cluster_workers(self) -> Dict:
        if not self.cluster:
            return {"enabled": False, "workers": [], "error": self.cluster_boot_error}
        return self.cluster.workers()

    def get_cluster_tasks(self) -> Dict:
        if not self.cluster:
            return {"enabled": False, "tasks": [], "error": self.cluster_boot_error}
        return self.cluster.tasks()

    def submit_cluster_task(
        self,
        command: str,
        required_tags: Optional[List[str]] = None,
        timeout_seconds: int = 120,
        priority: int = 5,
    ) -> Dict:
        if not self.cluster:
            raise ClusterError(self.cluster_boot_error or "Cluster não inicializado.")
        return self.cluster.submit_task(
            command,
            required_tags=required_tags,
            timeout_seconds=timeout_seconds,
            priority=priority,
        )

    def get_voice_status(self) -> Dict:
        return self.voice.status()

    def speak(self, text: str) -> Dict:
        return self.voice.speak(text)

    def update_wake_state(self, text: str) -> Dict:
        return self.voice.update_wake_state(text)

    # ─── System Metrics ───────────────────────────────────────────────────────

    def get_metrics(self) -> Dict:
        return self.monitor.snapshot()
    # ─── Curiosity Engine Integration ─────────────────────────────────────────

    def _on_new_insight(self, insight) -> None:
        """Callback chamado pelo CuriosityEngine quando um novo insight é gerado."""
        # Persist to DB
        try:
            self.db.save_insight(insight.to_dict())
        except Exception as e:
            logger.error("[Brain] Insight DB save error: %s", e)
        if self.curiosity_callback:
            self.curiosity_callback({
                "type":    "curiosity_insight",
                "insight": insight.to_dict(),
            })

    def set_curiosity_callback(self, cb: Callable) -> None:
        """Registra o callback WebSocket para notificações de curiosidade."""
        self.curiosity_callback = cb

    def get_insights(self, n: int = 20, tag: str = None) -> List[Dict]:
        # Try DB first (persisted), fallback to in-memory
        db_ins = self.db.get_insights(limit=n, tag=tag)
        if db_ins:
            return db_ins
        return [i.to_dict() for i in self.curiosity.get_top_insights(n, tag)]

    def get_random_insight(self) -> Optional[Dict]:
        ins = self.db.get_random_insight()
        if ins:
            return ins
        mem_ins = self.curiosity.get_random_insight()
        return mem_ins.to_dict() if mem_ins else None

    def get_curiosity_stats(self) -> Dict:
        stats = self.curiosity.get_stats()
        stats["db_insights"] = self.db.get_stats().get("insights", 0)
        timeline = self.db.get_curiosity_timeline(limit=20)
        if not timeline and stats.get("cycle"):
            timeline = [{
                "cycle_num": stats.get("cycle", 0),
                "insights_found": stats.get("insights_found", 0),
                "docs_scanned": len(list(CORPUS_ROOT.glob("*_chunks.jsonl"))),
                "top_score": 0,
                "ts": stats.get("last_analysis", time.time()),
            }]
        stats["curiosity_timeline"] = timeline
        return stats

    def search_insights(self, query: str) -> List[Dict]:
        db_res = self.db.search_insights(query)
        if db_res:
            return db_res
        return [i.to_dict() for i in self.curiosity.search_insights(query)]

    def get_topics(self) -> Dict:
        db_topics = self.db.get_topics()
        if db_topics:
            return db_topics
        return self.curiosity.get_topics()

    def get_metrics_history(self, minutes: int = 30) -> List[Dict]:
        return self.db.get_metrics_history(minutes)

    def save_metrics_snapshot(self, snap: Dict):
        self.db.save_metric(snap)
