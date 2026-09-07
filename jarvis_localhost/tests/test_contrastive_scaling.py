"""Synthetic coverage of pair ordering, bounded evaluation and batch resume."""

from __future__ import annotations

import math
import random
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import TrainingCancelled
from jarvis_localhost.corpus.provenance import CanonicalChunk, DocumentIdentity, corpus_sha256
from jarvis_localhost.retrieval import contrastive as module
from jarvis_localhost.retrieval.contrastive import ContrastivePair, ContrastiveTrainer, build_positive_pairs
from jarvis_localhost.retrieval.encoder import RetrieverConfig, RetrieverEncoder


def chunks_for_test(count: int) -> list[CanonicalChunk]:
    document = DocumentIdentity("doc_synthetic", "a" * 64, "synthetic.pdf", 100)
    return [
        CanonicalChunk.build(
            document, page=index + 1, section="synthetic section",
            bbox=(0, 0, 10, 10),
            text=f"Synthetic authorized circuit sensor number {index} measures pulse duration safely.",
            ordinal=index,
        )
        for index in range(count)
    ]


def original_pairs(chunks) -> list[ContrastivePair]:
    """The previous exhaustive algorithm is an oracle only for tiny inputs."""
    ordered = sorted(chunks, key=lambda c: (c.document_id, c.ordinal))
    pairs, seen = [], set()
    for index, anchor in enumerate(ordered):
        first, second = module._deterministic_views(anchor.text)
        self_key = (anchor.chunk_id, anchor.chunk_id)
        if self_key not in seen:
            seen.add(self_key)
            pairs.append(ContrastivePair(first, second, *self_key))
        candidates = []
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(ordered):
                neighbor = ordered[neighbor_index]
                if neighbor.document_id == anchor.document_id:
                    candidates.append(neighbor)
        if anchor.section:
            candidates.extend(
                candidate for candidate in ordered
                if candidate.chunk_id != anchor.chunk_id
                and candidate.document_id == anchor.document_id
                and candidate.section == anchor.section
            )
        for positive in candidates[:2]:
            key = (anchor.chunk_id, positive.chunk_id)
            if key not in seen:
                seen.add(key)
                pairs.append(ContrastivePair(anchor.text, positive.text, *key))
    return pairs


class PositivePairScalingTests(unittest.TestCase):
    def test_indexed_pairs_match_exhaustive_order_including_duplicate_slots(self):
        rng = random.Random(13)
        for size in range(1, 36):
            values = [
                SimpleNamespace(
                    document_id=f"doc_{rng.randrange(4)}", ordinal=rng.randrange(5),
                    section=rng.choice(["", "A", "B"]),
                    chunk_id=f"chunk_{rng.randrange(max(1, size // 2))}",
                    text=f"synthetic words {index} are retained exactly",
                )
                for index in range(size)
            ]
            with self.subTest(size=size):
                self.assertEqual(build_positive_pairs(values), original_pairs(values))
        # At the first document boundary, the section index repeats the next
        # neighbour. That duplicate consumes the second candidate slot.
        values = chunks_for_test(4)
        pairs = build_positive_pairs(values)
        first_positives = [p.positive_chunk_id for p in pairs if p.anchor_chunk_id == values[0].chunk_id]
        self.assertEqual(first_positives, [values[0].chunk_id, values[1].chunk_id])

    def test_section_lookup_has_linear_field_accesses(self):
        class CountedChunk:
            reads = 0

            def __init__(self, index):
                self.ordinal = index
                self.chunk_id = f"synthetic_{index}"
                self.text = "synthetic authorized sample passage"

            @property
            def document_id(self):
                type(self).reads += 1
                return "same_document"

            @property
            def section(self):
                type(self).reads += 1
                return "same_section"

        count = 10_000
        pairs = build_positive_pairs(CountedChunk(index) for index in range(count))
        self.assertGreater(len(pairs), count)
        self.assertLess(CountedChunk.reads, count * 15)

    def test_pair_preparation_honours_cancellation(self):
        cancellation = threading.Event()
        cancellation.set()
        with self.assertRaises(TrainingCancelled):
            build_positive_pairs(chunks_for_test(2), cancellation_event=cancellation)


class ContrastiveScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def trainer(self, count=9, *, weighted=False, dropout=0.15):
        chunks = chunks_for_test(count)
        tokenizer = JarvisTokenizer(vocab_size=80)
        tokenizer.train("\n\n".join(chunk.text for chunk in chunks))
        encoder = RetrieverEncoder(RetrieverConfig(
            vocab_size=tokenizer.vocab_actual_size, context_len=8,
            embed_dim=8, projection_dim=4, num_heads=2, num_layers=1,
            ff_dim=16, dropout=dropout,
        ), seed=23)
        payload = None
        if weighted:
            total = sum(range(1, count + 1))
            payload = {
                "format_version": 1, "corpus_sha256": corpus_sha256(chunks),
                "signal_version": 1, "chunk_ids": [c.chunk_id for c in chunks],
                "weights": [(index + 1) / total for index in range(count)],
            }
        return ContrastiveTrainer(encoder, tokenizer, chunks, seed=31, sampling_payload=payload)

    def test_small_uncertainty_preserves_original_all_corpus_scores(self):
        trainer = self.trainer(5, dropout=0.0)
        trainer.encoder.eval()
        views = [module._deterministic_views(chunk.text) for chunk in trainer.chunks]
        with torch.no_grad():
            logits = trainer._encode([v[0] for v in views]) @ trainer._encode([v[1] for v in views]).T
            expected = 1.0 - torch.softmax(logits / trainer.temperature, dim=-1).diagonal()
        actual = trainer.measure_retrieval_uncertainty(batch_size=8)
        for chunk, value in zip(trainer.chunks, expected.tolist()):
            self.assertAlmostEqual(actual[chunk.chunk_id], value, places=6)
        self.assertEqual(trainer.uncertainty_metadata["negative_scope"], "full_corpus")

    def test_large_uncertainty_encodes_only_bounded_batches_and_non_singleton_tail(self):
        trainer = self.trainer(13)
        calls, progress = [], []
        original = trainer._encode

        def measured_encode(texts):
            calls.append(len(texts))
            return original(texts)

        with patch.object(trainer, "_encode", side_effect=measured_encode):
            result = trainer.measure_retrieval_uncertainty(
                batch_size=4, progress_callback=lambda step, total: progress.append((step, total)),
            )
        self.assertEqual(set(result), {c.chunk_id for c in trainer.chunks})
        self.assertTrue(all(2 <= size <= 4 for size in calls))
        self.assertEqual(calls[-2:], [2, 2])
        self.assertEqual(progress[-1], (13, 13))
        self.assertEqual(trainer.uncertainty_metadata["negative_scope"], "local_batch")
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_uncertainty_cancellation_restores_training_mode(self):
        trainer = self.trainer(8)
        event = threading.Event()
        trainer.encoder.train()
        with self.assertRaises(TrainingCancelled):
            trainer.measure_retrieval_uncertainty(
                batch_size=2, cancellation_event=event,
                progress_callback=lambda *_: event.set(),
            )
        self.assertTrue(trainer.encoder.training)

    def test_nonfinite_uncertainty_is_not_reported_as_zero(self):
        trainer = self.trainer(3)
        with patch.object(trainer, "_encode", side_effect=lambda texts: torch.full((len(texts), 4), float("nan"))):
            with self.assertRaises(FloatingPointError):
                trainer.measure_retrieval_uncertainty()

    def test_progress_reports_initial_first_interval_and_last_steps(self):
        trainer = self.trainer(6)
        progress = []
        trainer.train(epochs=1, batch_size=3, progress_interval=3,
                      progress_callback=lambda step, total: progress.append((step, total)))
        total = progress[-1][1]
        self.assertEqual(progress[0], (0, total))
        self.assertEqual(progress[1], (1, total))
        self.assertIn((3, total), progress)
        self.assertEqual(progress[-1], (total, total))

    def test_batch_resume_matches_uninterrupted_training_with_dropout_and_sampling(self):
        for weighted in (False, True):
            with self.subTest(weighted=weighted), tempfile.TemporaryDirectory() as directory:
                complete = self.trainer(weighted=weighted)
                expected_history = complete.train(epochs=2, batch_size=3)
                expected_weights = {k: v.clone() for k, v in complete.encoder.state_dict().items()}
                interrupted = self.trainer(weighted=weighted)
                event = threading.Event()

                def cancel_at_three(step, total):
                    if step == 3:
                        event.set()

                with self.assertRaises(TrainingCancelled):
                    interrupted.train(
                        epochs=2, batch_size=3, checkpoint_dir=directory,
                        checkpoint_interval=2, progress_interval=1,
                        cancellation_event=event, progress_callback=cancel_at_three,
                    )
                state_path = Path(directory) / "retriever_training_state.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                self.assertEqual(state["steps"], 3)
                self.assertEqual(state["batch_in_epoch"], 3)
                resumed = self.trainer(weighted=weighted)
                history = resumed.train(epochs=2, batch_size=3, checkpoint_dir=directory)
                self.assertEqual(history, expected_history)
                self.assertTrue(resumed.resume_metadata["resumed"])
                self.assertEqual(resumed.resume_metadata["steps"], 3)
                for name, tensor in resumed.encoder.state_dict().items():
                    self.assertTrue(torch.equal(tensor, expected_weights[name]), name)
                done = self.trainer(weighted=weighted)
                with patch.object(done, "_encode", side_effect=AssertionError("completed model retrained")):
                    self.assertEqual(done.train(epochs=2, batch_size=3, checkpoint_dir=directory), history)
                self.assertTrue(done.encoder.trained_on_corpus)

    def test_checkpoint_refuses_changed_sampling_or_training_config(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.trainer(weighted=True)
            original.train(epochs=1, batch_size=3, checkpoint_dir=directory)
            changed = self.trainer(weighted=False)
            with self.assertRaisesRegex(ValueError, "lineage or configuration"):
                changed.train(epochs=1, batch_size=3, checkpoint_dir=directory)
            with self.assertRaisesRegex(ValueError, "lineage or configuration"):
                self.trainer(weighted=True).train(epochs=2, batch_size=3, checkpoint_dir=directory)

    def test_failed_snapshot_write_leaves_last_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer()
            trainer.train(epochs=1, batch_size=3, checkpoint_dir=directory)
            path = Path(directory) / "retriever_training_state.pt"
            original_bytes = path.read_bytes()
            with patch.object(torch, "save", side_effect=OSError("synthetic storage failure")):
                with self.assertRaises(OSError):
                    trainer._save_training_state(path, {})
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_oversized_checkpoint_is_refused_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.trainer(3)
            path = Path(directory) / "retriever_training_state.pt"
            with path.open("wb") as handle:
                handle.truncate(trainer._training_state_byte_limit(1) + 1)
            with patch.object(torch, "load", side_effect=AssertionError("oversized file loaded")):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    trainer.train(epochs=1, batch_size=3, checkpoint_dir=directory)

    def test_corrupt_checkpoint_values_are_refused_before_model_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.trainer(3)
            original.train(epochs=1, batch_size=3, checkpoint_dir=directory)
            path = Path(directory) / "retriever_training_state.pt"
            original_bytes = path.read_bytes()
            for fault in ("model_nan", "moment_inf", "optimizer_lr_inf", "missing_moments", "model_shape", "loss_nan", "extra_history", "negative_step"):
                with self.subTest(fault=fault):
                    path.write_bytes(original_bytes)
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    if fault == "model_nan":
                        next(iter(state["model_state_dict"].values())).reshape(-1)[0] = float("nan")
                    elif fault == "moment_inf":
                        next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"].reshape(-1)[0] = float("inf")
                    elif fault == "optimizer_lr_inf":
                        state["optimizer_state_dict"]["param_groups"][0]["lr"] = float("inf")
                    elif fault == "missing_moments":
                        state["optimizer_state_dict"]["state"].clear()
                    elif fault == "model_shape":
                        key = next(iter(state["model_state_dict"]))
                        state["model_state_dict"][key] = torch.zeros(1)
                    elif fault == "loss_nan":
                        state["epoch_loss_sum"] = float("nan")
                    elif fault == "extra_history":
                        state["history"].append(dict(state["history"][0]))
                    else:
                        state["steps"] = -1
                    torch.save(state, path)
                    fresh = self.trainer(3)
                    before = {k: v.clone() for k, v in fresh.encoder.state_dict().items()}
                    with self.assertRaisesRegex(ValueError, "retriever checkpoint"):
                        fresh.train(epochs=1, batch_size=3, checkpoint_dir=directory)
                    for key, tensor in fresh.encoder.state_dict().items():
                        self.assertTrue(torch.equal(tensor, before[key]), key)


if __name__ == "__main__":
    unittest.main()
