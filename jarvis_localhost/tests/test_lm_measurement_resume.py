"""Recover long evaluations without repeating completed canonical segments."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from jarvis_localhost.ai.language_model import JarvisConfig, JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import JarvisTrainer, TrainConfig
from jarvis_localhost.core.brain import _corpus_text
from jarvis_localhost.corpus.provenance import corpus_sha256
from jarvis_localhost.tests.test_brain_pipeline import _chunks


def make_trainer(directory):
    chunks = _chunks()
    corpus = _corpus_text(chunks)
    tokenizer = JarvisTokenizer(vocab_size=128)
    tokenizer.train(corpus)
    model = JarvisTransformer(JarvisConfig(
        vocab_size=tokenizer.vocab_actual_size, context_len=64,
        embed_dim=32, num_heads=4, num_layers=1, ff_dim=64,
    ))
    trainer = JarvisTrainer(model, tokenizer, corpus, TrainConfig(
        device="cpu", max_steps=1, batch_size=2, context_len=64,
        checkpoint_dir=str(directory),
    ), chunk_ids=[chunk.chunk_id for chunk in chunks],
        canonical_corpus_sha256=corpus_sha256(chunks))
    trainer.train()
    return trainer


class LMMeasurementResumeTests(unittest.TestCase):
    def test_padding_preserves_exact_causal_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            trainer.model.eval()
            expected = {}
            with torch.no_grad():
                for key, segment in zip(trainer.chunk_ids, trainer.chunk_segments):
                    ids = trainer.tokenizer.encode(segment, add_special=True)
                    numerator, denominator = 0.0, 0
                    for start in range(0, len(ids) - 1, trainer.cfg.context_len):
                        window = ids[start:start + trainer.cfg.context_len + 1]
                        _, loss = trainer.model(torch.tensor([window[:-1]]), torch.tensor([window[1:]]))
                        numerator += loss.item() * (len(window) - 1)
                        denominator += len(window) - 1
                    expected[key] = numerator / denominator
            measured = trainer.measure_chunk_losses()
            for key in expected:
                self.assertAlmostEqual(measured[key], expected[key], places=5)

    def test_failure_saves_completed_segments_and_retry_matches_exact_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            expected = trainer.measure_chunk_losses()
            trainer.release_training_buffers()
            checkpoint = Path(directory) / "lm_measurements.json"
            encode = trainer.tokenizer.encode
            calls = []

            def fail_second(segment, **kwargs):
                calls.append(segment)
                if len(calls) == 2:
                    raise MemoryError()
                return encode(segment, **kwargs)

            with patch.object(trainer.tokenizer, "encode", side_effect=fail_second):
                with self.assertRaises(MemoryError):
                    trainer.measure_chunk_losses(checkpoint_path=checkpoint)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(list(saved["measurements"]), [trainer.chunk_ids[0]])
            self.assertTrue(trainer.model.training)
            with patch.object(trainer.tokenizer, "encode", wraps=encode) as tracked:
                actual = trainer.measure_chunk_losses(checkpoint_path=checkpoint)
            self.assertEqual(actual, expected)
            self.assertNotIn(trainer.chunk_segments[0], [call.args[0] for call in tracked.call_args_list])
            with patch.object(trainer.model, "forward", side_effect=AssertionError("evaluation repeated")):
                self.assertEqual(trainer.measure_chunk_losses(checkpoint_path=checkpoint), expected)

    def test_measurements_from_another_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            checkpoint = Path(directory) / "lm_measurements.json"
            trainer.measure_chunk_losses(checkpoint_path=checkpoint)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            saved["binding"]["model_sha256"] = "0" * 64
            checkpoint.write_text(json.dumps(saved), encoding="utf-8")
            with patch.object(trainer.model, "forward", side_effect=AssertionError("untrusted cache used")):
                with self.assertRaisesRegex(ValueError, "another model or corpus"):
                    trainer.measure_chunk_losses(checkpoint_path=checkpoint)

    def test_finite_measurement_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            checkpoint = Path(directory) / "lm_measurements.json"
            trainer.measure_chunk_losses(checkpoint_path=checkpoint)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            saved["measurements"][trainer.chunk_ids[0]] += 0.125
            checkpoint.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                trainer.measure_chunk_losses(checkpoint_path=checkpoint)


if __name__ == "__main__":
    unittest.main()
