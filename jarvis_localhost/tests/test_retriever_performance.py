"""CPU equivalence and checkpoint tests for optional retriever execution policies."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import torch

from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import TrainingCancelled
from jarvis_localhost.retrieval.contrastive import ContrastiveTrainer, info_nce
from jarvis_localhost.retrieval.encoder import RetrieverConfig, RetrieverEncoder
from jarvis_localhost.tests.test_contrastive_scaling import chunks_for_test


class _LengthTokenizer:
    """Expose exact input lengths, keeping the actual encoder and padding real."""

    def encode(self, text, add_special=True):
        return list(range(1, int(text) + 1))

    def pad_sequence(self, values, length):
        return values + [0] * (length - len(values))


class RetrieverPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def trainer(self, *, bucketed=False, fused=False, dropout=0.0, context_len=128):
        chunks = chunks_for_test(9)
        tokenizer = JarvisTokenizer(vocab_size=120)
        tokenizer.train("\n\n".join(chunk.text for chunk in chunks))
        encoder = RetrieverEncoder(RetrieverConfig(
            vocab_size=tokenizer.vocab_actual_size, context_len=context_len,
            embed_dim=8, projection_dim=4, num_heads=2, num_layers=1,
            ff_dim=16, dropout=dropout,
        ), seed=23)
        return ContrastiveTrainer(
            encoder, tokenizer, chunks, seed=31,
            padding_policy="bucketed" if bucketed else "fixed",
            min_bucket_size=32, fuse_pair_encoding=fused,
        )

    def test_bucket_shapes_token_order_and_truncation_are_bounded(self):
        encoder = RetrieverEncoder(RetrieverConfig(
            vocab_size=600, context_len=512, embed_dim=8,
            projection_dim=4, num_heads=2, num_layers=1, ff_dim=16, dropout=0.0,
        ))
        tokenizer = _LengthTokenizer()
        for texts, minimum, expected in (
            (["5", "10"], 32, 32), (["5", "10"], 64, 64),
            (["5", "33"], 32, 64), (["60", "65"], 32, 128),
            (["129"], 32, 256), (["513"], 32, 512),
        ):
            with self.subTest(texts=texts, minimum=minimum):
                fixed, fixed_mask = encoder.tokenize(tokenizer, texts, "cpu")
                bucket, bucket_mask = encoder.tokenize(
                    tokenizer, texts, "cpu", padding_policy="bucketed", min_bucket_size=minimum,
                )
                self.assertEqual(bucket.shape, (len(texts), expected))
                self.assertTrue(torch.equal(bucket, fixed[:, :expected]))
                self.assertTrue(torch.equal(bucket_mask, fixed_mask[:, :expected]))
                self.assertEqual(bucket_mask.sum().item(), fixed_mask.sum().item())
                self.assertEqual(bucket.dtype, torch.int64)

    def test_eval_embeddings_match_with_less_padding(self):
        encoder = RetrieverEncoder(RetrieverConfig(
            vocab_size=130, context_len=128, embed_dim=8,
            projection_dim=4, num_heads=2, num_layers=2, ff_dim=16, dropout=0.2,
        ))
        encoder.eval()
        tokenizer = _LengthTokenizer()
        for texts in (["5", "10"], ["5", "33"], ["65", "100"]):
            with self.subTest(texts=texts), torch.no_grad():
                fixed = encoder(*encoder.tokenize(tokenizer, texts, "cpu"))
                bucket = encoder(*encoder.tokenize(tokenizer, texts, "cpu", padding_policy="bucketed"))
                torch.testing.assert_close(bucket, fixed, rtol=1e-5, atol=1e-6)

    def test_fused_forward_preserves_logical_objective_and_gradients_without_dropout(self):
        reference = self.trainer()
        accelerated = self.trainer(bucketed=True, fused=True)
        anchors = [chunk.text for chunk in reference.chunks[:8]]
        positives = [chunk.text for chunk in reversed(reference.chunks[1:])]
        shapes = []
        handle = accelerated.encoder.register_forward_pre_hook(
            lambda model, values: shapes.append(tuple(values[0].shape))
        )
        try:
            ref_a, ref_p = reference._encode_pairs(anchors, positives)
            opt_a, opt_p = accelerated._encode_pairs(anchors, positives)
        finally:
            handle.remove()
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0][0], 16)
        self.assertLess(shapes[0][1], 128)
        self.assertEqual((opt_a @ opt_p.T).shape, (8, 8))
        ref_loss = info_nce(ref_a, ref_p, reference.temperature)
        opt_loss = info_nce(opt_a, opt_p, accelerated.temperature)
        torch.testing.assert_close(opt_loss, ref_loss, rtol=1e-5, atol=1e-6)
        ref_loss.backward()
        opt_loss.backward()
        for (name, ref), (_, opt) in zip(reference.encoder.named_parameters(), accelerated.encoder.named_parameters()):
            torch.testing.assert_close(opt.grad, ref.grad, rtol=3e-4, atol=3e-5, msg=name)

    def test_legacy_checkpoint_transition_preserves_weights_moments_and_position(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.trainer(dropout=0.15)
            event = threading.Event()

            def stop_at_two(step, total):
                if step == 2:
                    event.set()

            with self.assertRaises(TrainingCancelled):
                original.train(
                    epochs=2, batch_size=3, checkpoint_dir=directory,
                    cancellation_event=event, progress_interval=1, progress_callback=stop_at_two,
                )
            path = Path(directory) / "retriever_training_state.pt"
            legacy = torch.load(path, map_location="cpu", weights_only=True)
            # These are the only metadata fields absent from the original
            # fixed-padding checkpoint format used by existing installations.
            legacy.pop("execution_policy")
            legacy.pop("execution_policy_transitions")
            torch.save(legacy, path)
            accelerated = self.trainer(bucketed=True, fused=True, dropout=0.15)
            second_event = threading.Event()
            observed = []

            def verify_resume(step, total):
                observed.append(step)
                if step == 2:
                    for name, tensor in accelerated.encoder.state_dict().items():
                        self.assertTrue(torch.equal(tensor, legacy["model_state_dict"][name]))
                    restored_optimizer = accelerated.optimizer.state_dict()
                    for identifier, values in restored_optimizer["state"].items():
                        self.assertEqual(values["step"], legacy["optimizer_state_dict"]["state"][identifier]["step"])
                        for key in ("exp_avg", "exp_avg_sq"):
                            self.assertTrue(torch.equal(values[key], legacy["optimizer_state_dict"]["state"][identifier][key]))
                if step == 3:
                    second_event.set()

            with self.assertRaises(TrainingCancelled):
                accelerated.train(
                    epochs=2, batch_size=3, checkpoint_dir=directory,
                    cancellation_event=second_event, progress_interval=1, progress_callback=verify_resume,
                )
            self.assertEqual(observed, [2, 3])
            self.assertTrue(accelerated.resume_metadata["execution_policy_changed"])
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["steps"], 3)
            self.assertEqual(saved["batch_in_epoch"], 3)
            transition = saved["execution_policy_transitions"][0]
            self.assertEqual(transition["at_step"], 2)
            self.assertEqual(transition["from"]["padding_policy"], "fixed")
            self.assertEqual(transition["to"]["padding_policy"], "bucketed")
            self.assertEqual(transition["from"]["max_fused_sequence_length"], 256)
            self.assertEqual(transition["to"]["max_fused_sequence_length"], 256)
            self.assertFalse(transition["bit_exact_transition"])
            self.assertEqual(saved["train_config"], legacy["train_config"])
            saved["execution_policy"].pop("max_fused_sequence_length")
            transition["from"].pop("max_fused_sequence_length")
            transition["to"].pop("max_fused_sequence_length")
            accelerated._restore_execution_metadata(saved, steps=3)
            self.assertTrue(accelerated.resume_metadata["execution_policy_changed"])
            self.assertEqual(len(accelerated.execution_policy_transitions), 2)
            self.assertEqual(accelerated.execution_policy_transitions[0]["to"]["max_fused_sequence_length"], 512)
            self.assertEqual(accelerated.execution_policy_transitions[1]["from"]["max_fused_sequence_length"], 512)
            self.assertEqual(accelerated.execution_policy_transitions[1]["to"]["max_fused_sequence_length"], 256)

    def test_fusion_guard_uses_two_forwards_above_256_tokens(self):
        trainer = self.trainer(bucketed=True, fused=True, context_len=512)

        class RepeatingTokenizer(_LengthTokenizer):
            def encode(self, text, add_special=True):
                return [1] * int(text)

        trainer.tokenizer = RepeatingTokenizer()
        for length, expected in (
            (256, [(16, 256)]), (257, [(8, 512), (8, 512)]),
            (512, [(8, 512), (8, 512)]),
        ):
            with self.subTest(length=length):
                shapes = []
                handle = trainer.encoder.register_forward_pre_hook(
                    lambda model, values: shapes.append(tuple(values[0].shape))
                )
                try:
                    anchors, positives = trainer._encode_pairs([str(length)] * 8, [str(length)] * 8)
                finally:
                    handle.remove()
                self.assertEqual(shapes, expected)
                self.assertEqual((anchors @ positives.T).shape, (8, 8))
                self.assertTrue(torch.isfinite(info_nce(anchors, positives, trainer.temperature)).item())

    def test_same_opt_in_policy_resumes_identically_with_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            continuous = self.trainer(bucketed=True, fused=True, dropout=0.15)
            expected_history = continuous.train(epochs=2, batch_size=3)
            interrupted = self.trainer(bucketed=True, fused=True, dropout=0.15)
            event = threading.Event()

            def stop_at_two(step, total):
                if step == 2:
                    event.set()

            with self.assertRaises(TrainingCancelled):
                interrupted.train(
                    epochs=2, batch_size=3, checkpoint_dir=directory,
                    cancellation_event=event, progress_interval=1, progress_callback=stop_at_two,
                )
            resumed = self.trainer(bucketed=True, fused=True, dropout=0.15)
            actual_history = resumed.train(epochs=2, batch_size=3, checkpoint_dir=directory)
            self.assertEqual(actual_history, expected_history)
            self.assertFalse(resumed.resume_metadata["execution_policy_changed"])
            self.assertEqual(resumed.execution_policy_transitions, [])
            self.assertEqual(resumed.resume_metadata["execution_policy"]["max_fused_sequence_length"], 256)
            for name, value in resumed.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, continuous.encoder.state_dict()[name]), name)

    def test_checkpoint_with_unbounded_fusion_records_guard_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.trainer(bucketed=True, fused=True)
            expected = original.train(epochs=1, batch_size=3, checkpoint_dir=directory)
            path = Path(directory) / "retriever_training_state.pt"
            older_policy = torch.load(path, map_location="cpu", weights_only=True)
            older_policy["execution_policy"].pop("max_fused_sequence_length")
            torch.save(older_policy, path)
            resumed = self.trainer(bucketed=True, fused=True)
            actual = resumed.train(epochs=1, batch_size=3, checkpoint_dir=directory)
            self.assertEqual(actual, expected)
            self.assertTrue(resumed.resume_metadata["execution_policy_changed"])
            transition = resumed.resume_metadata["execution_policy_transition"]
            self.assertEqual(transition["from"]["max_fused_sequence_length"], 512)
            self.assertEqual(transition["to"]["max_fused_sequence_length"], 256)
            self.assertFalse(transition["bit_exact_transition"])
            self.assertEqual(transition["at_step"], older_policy["steps"])

    def test_invalid_execution_policy_is_refused(self):
        trainer = self.trainer()
        with self.assertRaisesRegex(ValueError, "execution policy"):
            ContrastiveTrainer(trainer.encoder, trainer.tokenizer, trainer.chunks, padding_policy="unknown")
        with self.assertRaisesRegex(ValueError, "min_bucket_size"):
            trainer.encoder.tokenize(trainer.tokenizer, ["test"], "cpu", padding_policy="bucketed", min_bucket_size=7)


if __name__ == "__main__":
    unittest.main()
