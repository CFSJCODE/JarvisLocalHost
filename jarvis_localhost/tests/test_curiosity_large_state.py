"""Large curriculum restoration using only isolated synthetic state artifacts."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from jarvis_localhost.curiosity import engine as engine_module
from jarvis_localhost.curiosity.engine import CuriosityEngine
from jarvis_localhost.tests.test_curiosity_persistence import make_chunks, replace_corpus


class CorpusStateBudgetTests(unittest.TestCase):
    def test_validated_large_corpus_has_room_without_unbounding_json(self):
        mib = 1024 * 1024
        curriculum_limit = engine_module._json_state_size_limit("curriculum.json", 233_223)
        sampling_limit = engine_module._json_state_size_limit("sampling_weights.json", 233_223)
        self.assertGreater(curriculum_limit, 49_000_000)
        self.assertGreater(sampling_limit, 32 * mib)
        self.assertLessEqual(curriculum_limit, 256 * mib)
        self.assertLessEqual(sampling_limit, 256 * mib)
        for filename in ("curriculum.json", "sampling_weights.json"):
            with self.subTest(filename=filename):
                self.assertEqual(engine_module._json_state_size_limit(filename, 0), 32 * mib)
                self.assertEqual(engine_module._json_state_size_limit(filename, 10**9), 256 * mib)

    def test_other_json_artifacts_keep_fixed_limit_even_for_large_corpora(self):
        for filename in ("manifest.json", "state.json", "insights.json", "unknown.json"):
            with self.subTest(filename=filename):
                self.assertEqual(
                    engine_module._json_state_size_limit(filename, 10**9),
                    32 * 1024 * 1024,
                )


class LargeCuriosityStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.corpus_dir = self.root / "corpus"
        self.output_dir = self.root / "curiosity"
        self.stack.enter_context(patch.multiple(
            engine_module,
            MAX_JSON_STATE_BYTES=8192,
            CORPUS_JSON_HEADER_BYTES=1024,
            MAX_CORPUS_JSON_STATE_BYTES=256 * 1024,
        ))

    def new_engine(self):
        return CuriosityEngine(self.corpus_dir, self.output_dir, device="cpu", seed=41)

    def saved_generation(self):
        self.chunks = make_chunks("large-state", count=192)
        self.corpus_dir.mkdir()
        replace_corpus(self.corpus_dir, self.chunks)
        engine = self.new_engine()
        engine.update_learning_signals(
            lm_loss_by_chunk={self.chunks[0].chunk_id: 3.75},
            retrieval_uncertainty_by_chunk={self.chunks[1].chunk_id: 1.25},
        )
        expected = engine.get_sampling_weights()
        directory = (
            self.output_dir / "corpora" / expected["corpus_sha256"]
            / "generations" / engine._current_generation
        )
        return engine, directory, expected

    @staticmethod
    def refresh_record(directory, filename):
        path = directory / filename
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][filename] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_restores_curriculum_and_sampling_larger_than_fixed_json_limit(self):
        original, directory, expected = self.saved_generation()
        for filename in ("curriculum.json", "sampling_weights.json"):
            self.assertGreater((directory / filename).stat().st_size, engine_module.MAX_JSON_STATE_BYTES)
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        restored = self.new_engine()
        self.assertEqual(restored.get_sampling_weights(), expected)
        self.assertEqual(restored._current_generation, original._current_generation)
        self.assertNotIn("checkpoint_load_error", restored.state.last_metrics)
        self.assertEqual(restored._curriculum.signals[self.chunks[0].chunk_id].lm_loss, 3.75)
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})

    def test_preflight_rejects_oversized_curriculum_even_with_matching_hash(self):
        _, directory, _ = self.saved_generation()
        path = directory / "curriculum.json"
        limit = engine_module._json_state_size_limit(path.name, len(self.chunks))
        with path.open("ab") as stream:
            stream.write(b" " * (limit + 1 - path.stat().st_size))
        self.refresh_record(directory, path.name)
        before = path.read_bytes()
        restored = self.new_engine()
        restored.get_sampling_weights()
        error = restored.state.last_metrics["checkpoint_load_error"]
        self.assertIn("size limit", error)
        self.assertIn("curriculum.json", error)
        self.assertNotIn("checksum", error)
        self.assertEqual(restored._current_generation, "")
        self.assertEqual(path.read_bytes(), before)

    def test_preflight_does_not_expand_insights_limit_with_corpus_size(self):
        _, directory, _ = self.saved_generation()
        path = directory / "insights.json"
        with path.open("ab") as stream:
            stream.write(b" " * (engine_module.MAX_JSON_STATE_BYTES + 1 - path.stat().st_size))
        self.refresh_record(directory, path.name)
        restored = self.new_engine()
        restored.get_sampling_weights()
        error = restored.state.last_metrics["checkpoint_load_error"]
        self.assertIn("size limit", error)
        self.assertIn("insights.json", error)

    def test_checksum_mismatch_is_not_reported_as_size_limit(self):
        _, directory, _ = self.saved_generation()
        path = directory / "curriculum.json"
        contents = bytearray(path.read_bytes())
        contents[-1] ^= 1
        path.write_bytes(contents)
        restored = self.new_engine()
        restored.get_sampling_weights()
        error = restored.state.last_metrics["checkpoint_load_error"]
        self.assertIn("checksum failed for curriculum.json", error)
        self.assertNotIn("size limit", error)

    def test_reader_uses_per_artifact_limit_and_rejects_oversized_payload(self):
        reader = self.new_engine()
        for filename in ("curriculum.json", "sampling_weights.json", "state.json"):
            with self.subTest(filename=filename):
                path = self.root / filename
                limit = engine_module._json_state_size_limit(filename, 128)
                path.write_bytes(b"{}" + b" " * (limit - 2))
                self.assertEqual(reader._read_json(path, chunk_count=128), {})
                with path.open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaisesRegex(ValueError, "size limit"):
                    reader._read_json(path, chunk_count=128)

    def test_reader_bounds_payload_even_if_file_grows_after_stat(self):
        reader = self.new_engine()
        path = self.root / "curriculum.json"
        limit = engine_module._json_state_size_limit(path.name, 128)
        path.write_bytes(b"{}" + b" " * (limit - 1))
        with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=2)):
            with self.assertRaisesRegex(ValueError, "size limit"):
                reader._read_json(path, chunk_count=128)


if __name__ == "__main__":
    unittest.main()
