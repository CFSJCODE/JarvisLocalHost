"""Exercise interruption at the boundary between LM and retrieval on CPU."""

from __future__ import annotations

import tempfile
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from jarvis_localhost.ai.trainer import JarvisTrainer
from jarvis_localhost.core import brain as brain_module
from jarvis_localhost.storage.database import JarvisDB
from jarvis_localhost.tests.corpus_fixture import write_closed_corpus
from jarvis_localhost.tests.test_brain_pipeline import (
    _bare_brain, _chunks, _profile, _threads,
)


class TrainingRecoveryTests(unittest.TestCase):
    def test_optimized_retriever_recovers_without_retraining_completed_lm(self):
        with patch.dict(os.environ, {
            "JARVIS_RETRIEVER_PADDING": "bucketed",
            "JARVIS_RETRIEVER_MIN_BUCKET": "32",
            "JARVIS_RETRIEVER_FUSED_PAIRS": "1",
        }):
            self._check_recovery("retrieval")

    def test_retriever_failure_preserves_completed_lm_and_resumes_pipeline(self):
        self._check_recovery("retrieval")

    def test_final_corpus_read_failure_preserves_resumable_pair(self):
        self._check_recovery("corpus")

    def test_pointer_commit_failure_preserves_discoverable_run(self):
        self._check_recovery("pointer")

    def _check_recovery(self, failure):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root, models_root = root / "corpus", root / "models"
            corpus_root.mkdir()
            models_root.mkdir()
            chunks = _chunks()
            write_closed_corpus(corpus_root, chunks)
            brain = _bare_brain(JarvisDB(root / "jarvis.db"))
            brain.PIPELINE_MANIFEST_PATH = models_root / "jarvis_pipeline.json"
            with (
                patch.object(brain_module, "CORPUS_ROOT", corpus_root),
                patch.object(brain_module, "MODELS_ROOT", models_root),
                patch.object(brain_module, "build_training_profile", return_value=_profile()),
                patch.object(brain_module, "configure_cpu_threads", return_value=_threads()),
            ):
                with ExitStack() as faults:
                    if failure == "retrieval":
                        faults.enter_context(patch.object(brain_module.ContrastiveTrainer, "train", side_effect=RuntimeError("injected retrieval failure")))
                    elif failure == "corpus":
                        faults.enter_context(patch.object(brain_module, "_canonical_chunks", side_effect=[chunks, OSError("injected corpus read failure")]))
                    else:
                        atomic_json = brain_module._atomic_json

                        def fail_pointer(path, payload):
                            if path == brain.PIPELINE_MANIFEST_PATH:
                                raise OSError("injected pointer failure")
                            return atomic_json(path, payload)

                        faults.enter_context(patch.object(brain_module, "_atomic_json", side_effect=fail_pointer))
                    brain.start_training()
                    brain._training_thread.join(timeout=45)
                self.assertFalse(brain._training_thread.is_alive())
                self.assertFalse(brain.is_training)
                self.assertIn("injected", brain.train_progress["error"])
                self.assertTrue(brain.train_progress["checkpoint_preserved"])
                lookup = {
                    "models_root": models_root,
                    "text_digest": brain_module.JarvisTokenizer.corpus_digest(brain_module._corpus_text(chunks)),
                    "canonical_digest": brain_module.corpus_sha256(chunks),
                }
                match = brain_module._find_resumable_training(**lookup)
                self.assertIsNotNone(match)
                self.assertTrue((match[0] / "jarvis_final.trainer_state.pt").is_file())
                # A completed LM must not run/save new gradient steps on retry.
                with patch.object(JarvisTrainer, "_checkpoint", side_effect=AssertionError("LM retrained")):
                    brain.start_training()
                    brain._training_thread.join(timeout=45)
                self.assertFalse(brain._training_thread.is_alive())
                self.assertTrue(brain.is_trained, brain.train_progress)
                self.assertEqual(brain.train_progress["resumed_from_step"], 1)
                self.assertTrue(brain.PIPELINE_MANIFEST_PATH.is_file())
                self.assertEqual(len(brain.store), len(chunks))
                self.assertEqual(list(models_root.glob(".training-*")), [])
                # A crash after committing the pointer but before marker
                # cleanup must never turn the active pipeline into staging.
                final_dir = next(models_root.glob("pipeline-*"))
                (final_dir / ".training-pending.json").write_text("{}", encoding="utf-8")
                self.assertIsNone(brain_module._find_resumable_training(**lookup))
                read_text = Path.read_text

                def unreadable_pointer(path, *args, **kwargs):
                    if path == brain.PIPELINE_MANIFEST_PATH:
                        raise OSError("temporarily unreadable active pointer")
                    return read_text(path, *args, **kwargs)

                with patch.object(Path, "read_text", unreadable_pointer):
                    self.assertIsNone(brain_module._find_resumable_training(**lookup))

    def test_failure_to_record_error_does_not_hide_original_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            brain = _bare_brain(JarvisDB(Path(temporary) / "jarvis.db"))
            brain.PIPELINE_MANIFEST_PATH = Path(temporary) / "pipeline.json"
            with (
                patch.object(brain_module, "_canonical_chunks", side_effect=ValueError("invalid corpus snapshot")),
                patch.object(brain.db, "update_training_run", side_effect=OSError("telemetry unavailable")),
            ):
                brain.start_training()
                brain._training_thread.join(timeout=10)
            self.assertFalse(brain._training_thread.is_alive())
            self.assertFalse(brain.is_training)
            self.assertEqual(brain.train_progress["error"], "invalid corpus snapshot")
            self.assertTrue(brain._training_lock.acquire(blocking=False))
            brain._training_lock.release()

    def test_empty_exception_still_reports_error_type(self):
        with tempfile.TemporaryDirectory() as temporary:
            brain = _bare_brain(JarvisDB(Path(temporary) / "jarvis.db"))
            brain.PIPELINE_MANIFEST_PATH = Path(temporary) / "pipeline.json"
            with patch.object(brain_module, "_canonical_chunks", side_effect=MemoryError()):
                brain.start_training()
                brain._training_thread.join(timeout=10)
            self.assertFalse(brain.is_training)
            self.assertEqual(brain.train_progress["error"], "MemoryError")
            self.assertEqual(brain.train_progress["error_type"], "MemoryError")


if __name__ == "__main__":
    unittest.main()
