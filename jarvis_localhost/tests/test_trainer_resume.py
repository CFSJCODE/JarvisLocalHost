"""Checkpoint continuation regressions using isolated synthetic CPU training."""

from __future__ import annotations

import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from jarvis_localhost.ai.language_model import JarvisConfig, JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai import trainer as trainer_module
from jarvis_localhost.ai.trainer import (
    JarvisTrainer, TrainConfig, load_trainer_state, validate_trainer_checkpoint,
)


class _DisallowedPickle:
    def __reduce__(self):
        # Harmless callable that is deliberately absent from the allowlist.
        return (eval, ("40 + 2",))


class _LegacyDirectMLTensor:
    def __reduce__(self):
        return (
            torch._utils._rebuild_device_tensor_from_numpy,
            (np.arange(6, dtype=np.float32).reshape(2, 3),
             torch.float32, "privateuseone:0", False),
        )


class TrainerCheckpointTests(unittest.TestCase):
    CORPUS = (
        "O motor sintetico gira e registra os pulsos do sensor.\n\n"
        "A fonte sintetica alimenta o circuito durante a medicao."
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def make_trainer(
        self, directory: str | Path, *, max_steps: int = 3,
        model: JarvisTransformer | None = None,
    ) -> JarvisTrainer:
        tokenizer = JarvisTokenizer(vocab_size=80)
        tokenizer.train(self.CORPUS)
        model = model or JarvisTransformer(
            JarvisConfig(
                vocab_size=tokenizer.vocab_actual_size, context_len=8,
                embed_dim=8, num_heads=2, num_layers=1, ff_dim=16,
                dropout=0.0,
            ), seed=123,
        )
        return JarvisTrainer(
            model, tokenizer, self.CORPUS,
            TrainConfig(
                max_steps=max_steps, context_len=8, batch_size=2,
                checkpoint_dir=str(directory), validation_fraction=0.0,
                log_interval=1, device="cpu",
            ),
            canonical_corpus_sha256="c" * 64,
        )

    def reload_trainer(self, directory: str | Path, max_steps: int = 3) -> JarvisTrainer:
        model = JarvisTransformer.load(
            Path(directory) / "jarvis_final.pt",
            expected_corpus_sha256=JarvisTokenizer.corpus_digest(self.CORPUS),
        )
        return self.make_trainer(directory, max_steps=max_steps, model=model)

    def test_completed_lm_continues_pipeline_without_training_or_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_trainer(directory)
            original_history = first.train()
            state_path = Path(directory) / "jarvis_final.trainer_state.pt"
            checkpoint = Path(directory) / "jarvis_final.pt"
            original_bytes = checkpoint.read_bytes()
            resumed = self.reload_trainer(directory)
            self.assertEqual(resumed.apply_resume_state(state_path), 3)
            with patch.object(resumed.optimizer, "step", side_effect=AssertionError("trained")), \
                 patch.object(resumed, "_checkpoint", side_effect=AssertionError("overwrote")):
                history = resumed.train()
            self.assertEqual(history, original_history)
            self.assertEqual(resumed.tokens_seen, first.tokens_seen)
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            self.assertFalse(trainer_module._TRAINING_MUTEX.locked())

    def test_sidecar_restores_rng_and_optimizer_before_next_training_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_trainer(directory)
            first.train()
            expected = (torch.rand(3), np.random.random(3), random.random())
            resumed = self.reload_trainer(directory, max_steps=5)
            state_path = Path(directory) / "jarvis_final.trainer_state.pt"
            # New sidecars contain no objects requiring unrestricted pickle.
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.assertIsInstance(state["numpy_rng_state"][1], list)
            self.assertEqual(len(resumed.optimizer.state), 0)
            resumed.apply_resume_state(state_path)
            self.assertGreater(len(resumed.optimizer.state), 0)
            self.assertTrue(torch.equal(torch.rand(3), expected[0]))
            np.testing.assert_array_equal(np.random.random(3), expected[1])
            self.assertEqual(random.random(), expected[2])
            history = resumed.train()
            self.assertEqual([entry["step"] for entry in history if "loss" in entry], list(range(5)))

    def test_legacy_numpy_rng_sidecar_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_trainer(directory)
            first.train()
            path = Path(directory) / "jarvis_final.trainer_state.pt"
            state = load_trainer_state(path)
            state.pop("checkpoint_sha256")
            state.pop("history")
            state["numpy_rng_state"] = np.random.get_state()
            torch.save(state, path)
            loaded = load_trainer_state(path)
            self.assertIsInstance(loaded["numpy_rng_state"][1], np.ndarray)
            resumed = self.reload_trainer(directory)
            self.assertEqual(resumed.apply_resume_state(path), 3)
            self.assertEqual(resumed.train(), first.history)

    def test_legacy_loader_rejects_nonallowlisted_pickle_callable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pt"
            torch.save({"payload": _DisallowedPickle()}, path)
            with self.assertRaisesRegex(pickle.UnpicklingError, "unsupported trainer checkpoint global"):
                load_trainer_state(path)

    def test_legacy_directml_numpy_tensor_restores_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_directml.pt"
            torch.save({
                "format_version": 1, "step": 10, "tokens_seen": 60,
                "optimizer_state_dict": {"state": {0: {
                    "exp_avg": _LegacyDirectMLTensor(),
                }}},
            }, path)
            state = load_trainer_state(path)
            moment = state["optimizer_state_dict"]["state"][0]["exp_avg"]
            self.assertEqual(moment.device.type, "cpu")
            self.assertEqual(moment.dtype, torch.float32)
            self.assertTrue(torch.equal(moment, torch.arange(6, dtype=torch.float32).reshape(2, 3)))

    def test_pair_validator_rejects_model_bytes_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.train()
            state_path = Path(directory) / "jarvis_final.trainer_state.pt"
            state = load_trainer_state(state_path)
            checkpoint = Path(directory) / "jarvis_final.pt"
            data = bytearray(checkpoint.read_bytes())
            data[len(data) // 2] ^= 1
            checkpoint.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                validate_trainer_checkpoint(state_path, state)

    def test_pair_validator_rejects_old_optimizer_for_replaced_same_step_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.train()
            path = Path(directory) / "jarvis_final.trainer_state.pt"
            previous_state = load_trainer_state(path)
            with torch.no_grad():
                next(trainer.model.parameters()).add_(0.125)
            trainer._checkpoint("final", 1.0)
            with self.assertRaisesRegex(ValueError, "model digest"):
                validate_trainer_checkpoint(path, previous_state)

    def test_resume_refuses_mismatched_step_without_mutating_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_trainer(directory)
            first.train()
            path = Path(directory) / "jarvis_final.trainer_state.pt"
            state = load_trainer_state(path)
            state["step"] = 1
            torch.save(state, path)
            resumed = self.reload_trainer(directory)
            with self.assertRaisesRegex(ValueError, "position"):
                resumed.apply_resume_state(path)
            self.assertEqual(len(resumed.optimizer.state), 0)
            self.assertEqual(resumed._start_step, 0)

    def test_legacy_history_discards_steps_after_last_committed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_trainer(directory)
            first.train()
            path = Path(directory) / "jarvis_final.trainer_state.pt"
            state = load_trainer_state(path)
            state.pop("history")
            torch.save(state, path)
            future_history = [*first.history, {"step": 99, "loss": 1.0}, None]
            (Path(directory) / "train_history.json").write_text(
                json.dumps(future_history), encoding="utf-8",
            )
            resumed = self.reload_trainer(directory)
            resumed.apply_resume_state(path)
            self.assertEqual(resumed.history, first.history)

    def test_loader_initialization_failure_releases_training_mutex(self) -> None:
        class BrokenLoader:
            def __iter__(self):
                raise RuntimeError("loader setup failed")

        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.train_loader = BrokenLoader()
            try:
                with self.assertRaisesRegex(RuntimeError, "loader setup failed"):
                    trainer.train()
                self.assertFalse(trainer_module._TRAINING_MUTEX.locked())
                self.assertTrue(self.make_trainer(directory).train())
            finally:
                # Let other regressions execute even against the broken baseline.
                if trainer_module._TRAINING_MUTEX.locked():
                    trainer_module._TRAINING_MUTEX.release()

    def test_failed_sidecar_save_preserves_previous_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer._checkpoint("step_0", 1.0)
            previous = Path(directory) / "jarvis_step_0.trainer_state.pt"
            real_save = torch.save

            def save_or_fail(payload, path, *args, **kwargs):
                if ".trainer_state.pt." in str(path):
                    raise OSError("simulated storage failure")
                return real_save(payload, path, *args, **kwargs)

            trainer.step = 1
            with patch.object(torch, "save", side_effect=save_or_fail):
                with self.assertRaisesRegex(OSError, "simulated storage failure"):
                    trainer._checkpoint("step_1", 1.0)
            self.assertTrue(previous.is_file())
            validate_trainer_checkpoint(previous, load_trainer_state(previous))
            self.assertFalse(list(Path(directory).glob("*.trainer_state.pt.*.tmp")))


if __name__ == "__main__":
    unittest.main()
