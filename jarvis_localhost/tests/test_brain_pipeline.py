from __future__ import annotations

import dataclasses
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from jarvis_localhost.corpus.provenance import (
    CanonicalChunk,
    DocumentIdentity,
    corpus_sha256,
)
from jarvis_localhost.core import brain as brain_module
from jarvis_localhost.core.brain import JarvisBrain
from jarvis_localhost.hardware.device import (
    CPUThreadSettings,
    DeviceDescriptor,
    HardwareInfo,
)
from jarvis_localhost.hardware.profiles import (
    CorpusStats,
    TrainingResourceProfile,
)
from jarvis_localhost.retrieval.vector_store import VectorStore
from jarvis_localhost.storage.database import JarvisDB
from jarvis_localhost.tests.corpus_fixture import write_closed_corpus


def _hardware() -> HardwareInfo:
    gib = 1024**3
    return HardwareInfo(
        system="Windows",
        release="11",
        machine="AMD64",
        cpu_name="AMD Ryzen 5 4600G",
        physical_cpu_cores=6,
        logical_cpu_cores=12,
        total_ram_bytes=32 * gib,
        available_ram_bytes=20 * gib,
        installed_ram_bytes=32 * gib,
        gpu_name="AMD Radeon(TM) Graphics",
        gpu_dedicated_memory_bytes=8 * gib,
        gpu_shared_memory_bytes=16 * gib,
        directx_version="DirectX 12",
        sources=("test",),
    )


def _device() -> DeviceDescriptor:
    return DeviceDescriptor(
        backend="cpu",
        torch_device=torch.device("cpu"),
        display_name="CPU",
        accelerated=False,
        reason="deterministic unit test",
        smoke_tested=True,
        torch_available=True,
        torch_version=torch.__version__,
    )


def _threads() -> CPUThreadSettings:
    return CPUThreadSettings(
        intra_op_threads=1,
        inter_op_threads=1,
        torch_configured=True,
        environment_configured=False,
    )


def _profile() -> TrainingResourceProfile:
    corpus = CorpusStats(
        document_count=1,
        byte_count=4096,
        character_count=4096,
        estimated_tokens=1024,
        source_kind="text",
    )
    return TrainingResourceProfile(
        profile_version=1,
        backend="cpu",
        corpus=corpus,
        model_tier="test-compact",
        vocabulary_size=128,
        context_length=64,
        embedding_dimension=32,
        attention_heads=4,
        transformer_layers=1,
        feed_forward_dimension=64,
        estimated_parameter_count=25_000,
        batch_size=2,
        gradient_accumulation_steps=2,
        effective_batch_size=4,
        max_steps=1,
        evaluation_interval=1,
        checkpoint_interval=1,
        host_memory_budget_bytes=1024**3,
        accelerator_memory_budget_bytes=0,
        cpu_threads=1,
        interop_threads=1,
        dataloader_workers=0,
        rag_chunk_tokens=64,
        rag_overlap_tokens=8,
        reason="small deterministic integration profile",
    )


def _chunks() -> list[CanonicalChunk]:
    identity = DocumentIdentity(
        document_id="doc_0123456789abcdef01234567",
        document_sha256="a" * 64,
        filename="contrato.pdf",
        byte_size=1234,
    )
    base = (
        "evidencia contrato prazo pagamento obrigacao documento autorizado "
        "pagina clausula verificavel local soberano "
    )
    return [
        CanonicalChunk.build(
            identity,
            page=index + 1,
            section="Clausulas",
            bbox=(10 + index, 20, 500, 700),
            text=(base * 18) + f" identificador pagina {index + 1}",
            ordinal=index,
        )
        for index in range(3)
    ]


def _bare_brain(database: JarvisDB) -> JarvisBrain:
    brain = JarvisBrain.__new__(JarvisBrain)
    brain._state_lock = threading.RLock()
    brain._training_lock = threading.Lock()
    brain._training_cancel = threading.Event()
    brain._shutdown_event = threading.Event()
    brain._training_thread = None
    brain.hardware = _hardware()
    brain.device_descriptor = _device()
    brain.thread_settings = _threads()
    brain.device = torch.device("cpu")
    brain.training_profile = _profile()
    brain.db = database
    brain.tokenizer = None
    brain.model = None
    brain.retriever_encoder = None
    brain.rag = None
    brain.store = VectorStore()
    brain.is_trained = False
    brain.is_training = False
    brain.train_progress = {}
    brain.active_corpus_sha256 = ""
    brain.active_corpus_text_sha256 = ""
    brain.pipeline_error = None
    brain._active_store_prefix = None
    return brain


class DatabaseMigrationTests(unittest.TestCase):
    def test_old_database_is_migrated_and_full_corpus_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.execute(
                """CREATE TABLE documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT,
                    pages INTEGER, words INTEGER, tables INTEGER, images INTEGER,
                    language TEXT, indexed INTEGER DEFAULT 0, corpus TEXT,
                    ts REAL NOT NULL
                )"""
            )
            connection.commit()
            connection.close()

            db = JarvisDB(path)
            long_corpus = "x" * 60_000
            document_id = db.save_document(
                "prova.pdf",
                str(Path(temporary) / "prova.pdf"),
                {"pages": 2, "words": 30},
                long_corpus,
                document_id="doc_stable",
                document_sha256="b" * 64,
                corpus_sha256="c" * 64,
                chunks_file="prova_chunks.jsonl",
                canonical_chunks=4,
            )
            self.assertEqual(document_id, "doc_stable")
            columns = {
                row[1]
                for row in sqlite3.connect(path).execute(
                    "PRAGMA table_info(documents)"
                )
            }
            self.assertTrue(
                {
                    "document_sha256",
                    "corpus_sha256",
                    "chunks_file",
                    "canonical_chunks",
                }.issubset(columns)
            )
            with db._conn() as connection:
                stored = connection.execute(
                    "SELECT corpus, document_sha256 FROM documents WHERE id=?",
                    (document_id,),
                ).fetchone()
            self.assertEqual(len(stored["corpus"]), len(long_corpus))
            self.assertEqual(stored["document_sha256"], "b" * 64)


class CanonicalRuntimeTests(unittest.TestCase):
    def test_process_pdf_indexes_original_sha_page_and_bbox(self) -> None:
        chunks = _chunks()
        with tempfile.TemporaryDirectory() as temporary:
            db = JarvisDB(Path(temporary) / "jarvis.db")
            brain = _bare_brain(db)
            result = SimpleNamespace(
                filename="contrato.pdf",
                stats={"pages": 3, "words": 300, "canonical_chunks": 3},
                training_corpus="\n\n".join(chunk.text for chunk in chunks),
                document_id=chunks[0].document_id,
                document_sha256=chunks[0].document_sha256,
                canonical_chunks=chunks,
                chunks_path=str(Path(temporary) / "contrato_chunks.jsonl"),
            )
            brain.processor = SimpleNamespace(process=lambda _: result)
            with patch.object(brain_module, "_canonical_chunks", return_value=chunks):
                stats = brain.process_pdf(str(Path(temporary) / "contrato.pdf"))

            self.assertEqual(stats["document_sha256"], "a" * 64)
            self.assertEqual(stats["indexed_chunks"], 3)
            retrieved = brain.rag.retrieve("prazo pagamento", top_k=1)[0]
            self.assertEqual(retrieved["document_sha256"], "a" * 64)
            self.assertIn(retrieved["page"], {1, 2, 3})
            self.assertEqual(len(retrieved["bbox"]), 4)
            row = db.list_documents()[0]
            self.assertEqual(row["document_sha256"], "a" * 64)
            self.assertEqual(row["canonical_chunks"], 3)

    def test_checksum_mismatch_rejects_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "weights.bin"
            artifact.write_bytes(b"trusted")
            record = brain_module._file_record(artifact, root)
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                brain_module._resolve_artifact(record, root)


class SovereignTrainingIntegrationTests(unittest.TestCase):
    def test_concurrent_start_and_database_failure_do_not_stick_training(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingDatabase:
            def start_training_run(self, **_: object) -> str:
                entered.set()
                release.wait(timeout=10)
                raise sqlite3.OperationalError("injected database failure")

        with tempfile.TemporaryDirectory() as temporary:
            brain = _bare_brain(BlockingDatabase())
            brain.PIPELINE_MANIFEST_PATH = Path(temporary) / "jarvis_pipeline.json"
            brain.start_training()
            self.assertTrue(entered.wait(timeout=5))
            first_thread = brain._training_thread
            brain.start_training()
            self.assertIs(brain._training_thread, first_thread)
            release.set()
            first_thread.join(timeout=10)
            self.assertFalse(first_thread.is_alive())
            self.assertFalse(brain.is_training)
            self.assertTrue(brain._training_lock.acquire(blocking=False))
            brain._training_lock.release()

    def test_training_builds_one_lineage_for_lm_retriever_and_vectors(self) -> None:
        chunks = _chunks()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "corpus"
            models_root = root / "models"
            corpus_root.mkdir()
            models_root.mkdir()
            write_closed_corpus(corpus_root, chunks)
            db = JarvisDB(root / "jarvis.db")
            brain = _bare_brain(db)
            brain.PIPELINE_MANIFEST_PATH = models_root / "jarvis_pipeline.json"

            class RecordingCuriosity:
                def __init__(self) -> None:
                    self.requested_ids: list[str] = []
                    self.measured: dict[str, dict[str, float]] = {}

                @staticmethod
                def get_stats() -> dict:
                    return {"is_running": False}

                def get_sampling_weights(self, chunk_ids: list[str]) -> dict:
                    self.requested_ids = list(chunk_ids)
                    return {
                        "format_version": 1,
                        "corpus_sha256": corpus_sha256(chunks),
                        "signal_version": 3,
                        "chunk_ids": list(chunk_ids),
                        "weights": [1.0 / len(chunk_ids)] * len(chunk_ids),
                    }

                def update_learning_signals(self, **signals: object) -> dict:
                    self.measured = signals
                    return {
                        "signal_version": 3 + 2 * len(chunks),
                        "generation": "test-generation",
                    }

            curiosity = RecordingCuriosity()
            brain.curiosity = curiosity

            with (
                patch.object(brain_module, "CORPUS_ROOT", corpus_root),
                patch.object(brain_module, "MODELS_ROOT", models_root),
                patch.object(
                    brain_module,
                    "build_training_profile",
                    return_value=_profile(),
                ),
                patch.object(
                    brain_module,
                    "configure_cpu_threads",
                    return_value=_threads(),
                ),
            ):
                brain.start_training()
                self.assertIsNotNone(brain._training_thread)
                brain._training_thread.join(timeout=90)

            self.assertFalse(brain._training_thread.is_alive())
            self.assertFalse(brain.is_training)
            self.assertTrue(brain.is_trained, brain.train_progress)
            self.assertTrue(brain.retriever_encoder.trained_on_corpus)
            self.assertEqual(len(brain.store), len(chunks))
            self.assertEqual(
                {row["chunk_id"] for row in brain.store.metadata},
                {chunk.chunk_id for chunk in chunks},
            )
            self.assertEqual(
                brain.active_corpus_sha256,
                corpus_sha256(chunks),
            )
            self.assertEqual(curiosity.requested_ids, [chunk.chunk_id for chunk in chunks])
            self.assertEqual(
                set(curiosity.measured["lm_loss_by_chunk"]),
                {chunk.chunk_id for chunk in chunks},
            )
            self.assertEqual(
                set(curiosity.measured["retrieval_uncertainty_by_chunk"]),
                {chunk.chunk_id for chunk in chunks},
            )

            manifest = json.loads(
                brain.PIPELINE_MANIFEST_PATH.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["initialized_from"], "random")
            self.assertFalse(manifest["external_weights"])
            self.assertEqual(manifest["corpus_sha256"], corpus_sha256(chunks))
            curriculum = manifest["training"]["curiosity_curriculum"]
            self.assertTrue(curriculum["enabled"])
            self.assertEqual(curriculum["input_signal_version"], 3)
            self.assertEqual(curriculum["generation"], "test-generation")
            self.assertEqual(curriculum["lm_loss"]["count"], len(chunks))
            self.assertEqual(
                manifest["training"]["gradient_accumulation_steps"], 2
            )
            self.assertEqual(manifest["training"]["dataloader_workers"], 0)
            for record in manifest["artifacts"].values():
                resolved = brain_module._resolve_artifact(record, models_root)
                self.assertTrue(resolved.is_file())

            answer = brain.rag.answer("Qual e o prazo de pagamento?", top_k=2)
            self.assertEqual(answer["mode"], "strict")
            self.assertTrue(answer["sources"])
            self.assertEqual(answer["sources"][0]["document_sha256"], "a" * 64)

            reloaded = _bare_brain(db)
            reloaded.PIPELINE_MANIFEST_PATH = brain.PIPELINE_MANIFEST_PATH
            with (
                patch.object(brain_module, "CORPUS_ROOT", corpus_root),
                patch.object(brain_module, "MODELS_ROOT", models_root),
            ):
                reloaded._try_load_existing()
            self.assertTrue(reloaded.is_trained, reloaded.pipeline_error)
            self.assertEqual(
                reloaded.active_corpus_sha256,
                corpus_sha256(chunks),
            )
            self.assertEqual(len(reloaded.store), len(chunks))

    def test_eval_only_progress_event_does_not_regress_percent(self) -> None:
        """Regression test for the F9 percent-stomp bug (found via live
        evidence during real production training on 2026-08-31).

        JarvisTrainer.train() fires the same callback for two different event
        shapes on the language-model training loop: a per-log_interval step
        event carrying "progress" (fraction of max_steps completed), and a
        separate eval_interval-triggered event carrying only
        {"step", "val_loss"} with no "progress" key at all. Because
        eval_interval is always a multiple of the step-logging interval here,
        every evaluation fires immediately after a step event for the same
        step. Before the fix, ``_train_callback`` in brain.py always
        recomputed "percent" from ``info.get("progress", 0.0)``, so the eval
        event's missing key silently defaulted to 0.0 and snapped the
        publicly reported ``train_progress["percent"]`` back down to the
        phase's 12% floor right after the correct value had just been
        published — a real, reproducible regression an operator watching
        ``/api/train/status`` would see as the progress bar jumping
        backwards. This test uses a profile with max_steps=10 (large enough
        to reach step > 0) precisely because the neighboring end-to-end
        test's max_steps=1 never exercises the eval branch at all (its
        ``step > 0`` guard is unreachable at step 0), which is why the
        original 119 passing tests never caught this.
        """

        chunks = _chunks()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "corpus"
            models_root = root / "models"
            corpus_root.mkdir()
            models_root.mkdir()
            write_closed_corpus(corpus_root, chunks)
            db = JarvisDB(root / "jarvis.db")
            brain = _bare_brain(db)
            brain.PIPELINE_MANIFEST_PATH = models_root / "jarvis_pipeline.json"

            class StubCuriosity:
                @staticmethod
                def get_stats() -> dict:
                    return {"is_running": False}

                @staticmethod
                def get_sampling_weights(chunk_ids: list[str]) -> dict:
                    return {
                        "format_version": 1,
                        "corpus_sha256": corpus_sha256(chunks),
                        "signal_version": 3,
                        "chunk_ids": list(chunk_ids),
                        "weights": [1.0 / len(chunk_ids)] * len(chunk_ids),
                    }

                @staticmethod
                def update_learning_signals(**_: object) -> dict:
                    return {"signal_version": 4, "generation": "test-generation"}

            brain.curiosity = StubCuriosity()

            # max_steps=10 with evaluation_interval=2 guarantees evaluations
            # fire at steps 2/4/6/8 (all > 0), each immediately after that
            # step's regular log event - the exact interleaving that exposed
            # the bug on the real 1596-step / 250-eval-interval production
            # run.
            eval_profile = dataclasses.replace(
                _profile(),
                max_steps=10,
                evaluation_interval=2,
                checkpoint_interval=10,
            )

            percent_after_each_event: list[float] = []

            def progress_callback(_info: dict) -> None:
                percent = brain.train_progress.get("percent")
                if percent is not None:
                    percent_after_each_event.append(float(percent))

            with (
                patch.object(brain_module, "CORPUS_ROOT", corpus_root),
                patch.object(brain_module, "MODELS_ROOT", models_root),
                patch.object(
                    brain_module,
                    "build_training_profile",
                    return_value=eval_profile,
                ),
                patch.object(
                    brain_module,
                    "configure_cpu_threads",
                    return_value=_threads(),
                ),
            ):
                brain.start_training(progress_callback=progress_callback)
                self.assertIsNotNone(brain._training_thread)
                brain._training_thread.join(timeout=90)

            self.assertFalse(brain._training_thread.is_alive())
            self.assertTrue(brain.is_trained, brain.train_progress)
            # Sanity check that this test actually exercises the eval-only
            # event path instead of vacuously passing on an empty sequence.
            self.assertGreaterEqual(len(percent_after_each_event), 5)
            running_max = 0.0
            for percent in percent_after_each_event:
                self.assertGreaterEqual(
                    percent,
                    running_max,
                    f"percent regressed to {percent} after reaching "
                    f"{running_max} - an eval-only event stomped it: "
                    f"{percent_after_each_event}",
                )
                running_max = max(running_max, percent)


if __name__ == "__main__":
    unittest.main()
