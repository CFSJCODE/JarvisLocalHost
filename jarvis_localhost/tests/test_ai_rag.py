from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jarvis_localhost.ai.language_model import JarvisConfig, JarvisTransformer
from jarvis_localhost.ai.dataset import TextDataset, validate_sampling_payload
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import JarvisTrainer, TrainConfig
from jarvis_localhost.corpus.provenance import (
    CanonicalChunk,
    DocumentIdentity,
    corpus_sha256,
)
from jarvis_localhost.rag.citations import Citation, citations_from_results
from jarvis_localhost.rag.engine import RAGEngine, RAGMode
from jarvis_localhost.rag.grounding import verify_grounding
from jarvis_localhost.retrieval.contrastive import (
    ContrastiveTrainer,
    build_positive_pairs,
)
from jarvis_localhost.retrieval.encoder import RetrieverConfig, RetrieverEncoder
from jarvis_localhost.retrieval.retriever import SovereignRetriever
from jarvis_localhost.retrieval.vector_store import VectorStore


def sample_chunks() -> list[CanonicalChunk]:
    first = DocumentIdentity("doc_a", "a" * 64, "manual.pdf", 100)
    second = DocumentIdentity("doc_b", "b" * 64, "guia.pdf", 100)
    return [
        CanonicalChunk.build(
            first,
            page=7,
            section="Temporizadores",
            bbox=(1, 2, 3, 4),
            text="O temporizador reinicia o contador quando recebe um pulso de controle.",
            ordinal=0,
        ),
        CanonicalChunk.build(
            second,
            page=3,
            section="Energia",
            bbox=(0, 0, 10, 10),
            text="A fonte de alimentação fornece cinco volts ao circuito principal.",
            ordinal=0,
        ),
    ]


class TokenizerModelTests(unittest.TestCase):
    def test_tokenizer_is_deterministic_and_corpus_bound(self) -> None:
        corpus = "alpha beta beta gamma; temporizador controle contador"
        first = JarvisTokenizer(vocab_size=64)
        second = JarvisTokenizer(vocab_size=64)
        first.train(corpus)
        second.train(corpus)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.encode("beta gamma"), second.encode("beta gamma"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            first.save(path)
            loaded = JarvisTokenizer.load(
                path, expected_corpus_sha256=first.corpus_sha256
            )
            self.assertEqual(first.fingerprint(), loaded.fingerprint())

    def test_citation_markers_have_a_stable_round_trip_contract(self) -> None:
        tokenizer = JarvisTokenizer(vocab_size=64)
        tokenizer.train("corpus autorizado sem marcadores de citação")
        ids = tokenizer.encode("Resposta [e1] e [E27].", add_special=False)
        self.assertNotIn(tokenizer.UNK_ID, tokenizer.encode("[E1024]", False))
        decoded = tokenizer.decode(ids)
        self.assertIn("[E1]", decoded)
        self.assertIn("[E27]", decoded)

    def test_exact_context_plus_one_produces_one_training_window(self) -> None:
        dataset = TextDataset(list(range(9)), context_len=8)
        self.assertEqual(len(dataset), 1)
        inputs, targets = dataset[0]
        self.assertEqual(inputs.tolist(), list(range(8)))
        self.assertEqual(targets.tolist(), list(range(1, 9)))

    def test_sampling_weights_are_versioned_and_corpus_bound(self) -> None:
        payload = {
            "format_version": 1,
            "corpus_sha256": "a" * 64,
            "signal_version": 2,
            "chunk_ids": ["chk_a", "chk_b"],
            "weights": [0.6, 0.4],
        }
        weights, metadata = validate_sampling_payload(
            payload,
            expected_corpus_sha256="a" * 64,
            expected_chunk_ids=["chk_a", "chk_b"],
        )
        self.assertEqual(weights, {"chk_a": 0.6, "chk_b": 0.4})
        self.assertEqual(metadata["signal_version"], 2)
        with self.assertRaisesRegex(ValueError, "different corpus"):
            validate_sampling_payload(
                payload,
                expected_corpus_sha256="b" * 64,
                expected_chunk_ids=["chk_a", "chk_b"],
            )

    def test_safe_checkpoint_round_trip_and_lineage(self) -> None:
        tokenizer = JarvisTokenizer(vocab_size=48)
        tokenizer.train("um corpus local pequeno com palavras repetidas corpus local")
        config = JarvisConfig(
            vocab_size=tokenizer.vocab_actual_size,
            context_len=32,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            dropout=0.0,
        )
        model = JarvisTransformer(config, seed=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(
                path,
                corpus_sha256=tokenizer.corpus_sha256,
                tokenizer_sha256=tokenizer.fingerprint(),
            )
            loaded = JarvisTransformer.load(
                path,
                expected_corpus_sha256=tokenizer.corpus_sha256,
                expected_tokenizer_sha256=tokenizer.fingerprint(),
            )
            inputs = torch.tensor([[2, 3]], dtype=torch.long)
            self.assertTrue(torch.equal(model(inputs)[0], loaded(inputs)[0]))
            manifest = json.loads(
                path.with_suffix(".pt.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["checkpoint_sha256"]), 64)
            payload = bytearray(path.read_bytes())
            payload[len(payload) // 2] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                JarvisTransformer.load(path)

    def test_trainer_encodes_boundaries_controls_and_caps_warmup(self) -> None:
        corpus = (
            "O motor aciona o eixo e registra a velocidade de rotação.\n\n"
            "O sensor mede cada pulso e preserva a amostra autorizada."
        )
        tokenizer = JarvisTokenizer(vocab_size=96)
        tokenizer.train(corpus)
        model = JarvisTransformer(
            JarvisConfig(
                vocab_size=tokenizer.vocab_actual_size,
                context_len=16,
                embed_dim=16,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                dropout=0.0,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            chunk_ids = ("chk_measure_a", "chk_measure_b")
            canonical_digest = "c" * 64
            trainer = JarvisTrainer(
                model,
                tokenizer,
                corpus,
                TrainConfig(
                    max_steps=50,
                    warmup_steps=100,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=directory,
                ),
                chunk_ids=chunk_ids,
                canonical_corpus_sha256=canonical_digest,
                sampling_payload={
                    "format_version": 1,
                    "corpus_sha256": canonical_digest,
                    "signal_version": 3,
                    "chunk_ids": list(chunk_ids),
                    "weights": [0.75, 0.25],
                },
            )
            ids = trainer.training_token_ids
            self.assertIn(tokenizer.BOS_ID, ids)
            self.assertIn(tokenizer.EOS_ID, ids)
            self.assertIn(tokenizer.SEP_ID, ids)
            self.assertIn(tokenizer.STRUCTURAL_TOKENS["[E"], ids)
            self.assertEqual(trainer.control_training_metadata["evidence_examples"], 2)
            self.assertEqual(trainer.control_training_metadata["abstention_examples"], 1)
            self.assertFalse(trainer.control_training_metadata["invented_facts"])
            self.assertEqual(trainer.effective_warmup_steps, 5)
            self.assertEqual(trainer.learning_rate_at(5), trainer.cfg.learning_rate)
            measured_losses = trainer.measure_chunk_losses()
            self.assertEqual(set(measured_losses), set(chunk_ids))
            self.assertTrue(all(math.isfinite(value) for value in measured_losses.values()))
            self.assertTrue(trainer.sampling_metadata["weighted"])
            self.assertEqual(trainer.sampling_metadata["signal_version"], 3)
            trainer._checkpoint("metadata", 1.0)
            lineage = json.loads(
                (Path(directory) / "jarvis_metadata.pt.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                lineage["training"]["control_training"]["evidence_origin"],
                "exact_provided_corpus_segments",
            )
            canonical_path = Path(directory) / "canonical.pt"
            model.save(canonical_path, training={"objective": "causal-lm"})
            canonical_lineage = json.loads(
                canonical_path.with_suffix(".pt.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                canonical_lineage["training"]["control_training"],
                trainer.control_training_metadata,
            )
            self.assertTrue(canonical_lineage["training"]["sampling"]["weighted"])


class RetrievalRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = sample_chunks()
        corpus = "\n".join(chunk.text for chunk in self.chunks)
        self.tokenizer = JarvisTokenizer(vocab_size=128)
        self.tokenizer.train(corpus)
        self.retriever = SovereignRetriever(self.tokenizer)
        self.retriever.index(self.chunks)

    def test_lexical_retrieval_preserves_page_and_hash(self) -> None:
        results = self.retriever.retrieve("Como o temporizador reinicia o contador?", 2)
        self.assertEqual(results[0].chunk.page, 7)
        self.assertEqual(results[0].chunk.document_sha256, "a" * 64)
        self.assertGreater(results[0].score, 0.5)

    def test_reindex_publishes_a_complete_snapshot(self) -> None:
        self.retriever.index([self.chunks[0]])
        self.assertEqual(set(self.retriever.chunks), {self.chunks[0].chunk_id})
        self.assertEqual(self.retriever.retrieve("fonte alimentação", 2), [])
        self.retriever.index([])
        self.assertFalse(self.retriever.chunks)
        self.assertEqual(self.retriever.retrieve("temporizador", 2), [])

    def test_stale_dense_ids_are_filtered_without_key_error(self) -> None:
        encoder = RetrieverEncoder(
            RetrieverConfig(
                vocab_size=self.tokenizer.vocab_actual_size,
                context_len=32,
                embed_dim=16,
                projection_dim=8,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                dropout=0.0,
            )
        )
        encoder.trained_on_corpus = True
        retriever = SovereignRetriever(self.tokenizer)
        retriever.index(self.chunks)
        retriever.encoder = encoder
        retriever.store.add(
            np.ones((1, 8), dtype=np.float32),
            [{"chunk_id": "chk_stale_generation"}],
        )
        self.assertEqual(retriever.retrieve("fotossíntese", 2), [])

    def test_strict_rag_is_extractive_cited_and_grounded(self) -> None:
        engine = RAGEngine(
            None,
            self.tokenizer,
            retriever=self.retriever,
            mode=RAGMode.STRICT,
        )
        response = engine.answer("O que reinicia o contador?", top_k=2)
        self.assertFalse(response["abstained"])
        self.assertEqual(response["method"], "extractive")
        self.assertIn("[E1]", response["answer"])
        self.assertEqual(response["sources"][0]["page"], 7)
        self.assertTrue(response["grounding"]["grounded"])

    def test_strict_rag_abstains_for_dense_only_irrelevant_evidence(self) -> None:
        encoder = RetrieverEncoder(
            RetrieverConfig(
                vocab_size=self.tokenizer.vocab_actual_size,
                context_len=32,
                embed_dim=16,
                projection_dim=8,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                dropout=0.0,
            ),
            seed=23,
        )
        encoder.trained_on_corpus = True
        retriever = SovereignRetriever(self.tokenizer, encoder=encoder)
        retriever.index(self.chunks)
        results = retriever.retrieve("fotossíntese", top_k=2)
        self.assertTrue(results)
        self.assertTrue(all(result.lexical_score == 0.0 for result in results))
        self.assertEqual(results[0].score, 0.0)
        response = RAGEngine(
            None,
            self.tokenizer,
            retriever=retriever,
            mode=RAGMode.STRICT,
        ).answer("fotossíntese", top_k=2)
        self.assertTrue(response["abstained"])
        self.assertEqual(response["reason"], "retrieval_below_threshold")

    def test_prompt_budget_keeps_question(self) -> None:
        config = JarvisConfig(
            vocab_size=self.tokenizer.vocab_actual_size,
            context_len=128,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            dropout=0.0,
        )
        model = JarvisTransformer(config)
        engine = RAGEngine(
            model,
            self.tokenizer,
            retriever=self.retriever,
            mode=RAGMode.RAG,
        )
        results = self.retriever.retrieve("temporizador contador", 2)
        prompt_ids = engine._prompt_ids(
            "temporizador contador", citations_from_results(results), max_new=32
        )
        decoded = self.tokenizer.decode(prompt_ids)
        self.assertIn("temporizador", decoded)
        self.assertIn("contador", decoded)
        self.assertLessEqual(len(prompt_ids), config.context_len - 32)

    def test_default_generation_budget_preserves_question_and_evidence(self) -> None:
        model = JarvisTransformer(
            JarvisConfig(
                vocab_size=self.tokenizer.vocab_actual_size,
                context_len=128,
                embed_dim=16,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                dropout=0.0,
            )
        )
        engine = RAGEngine(model, self.tokenizer, retriever=self.retriever)
        results = self.retriever.retrieve("temporizador contador", 2)
        effective = engine._effective_generation_tokens(96)
        prompt_ids = engine._prompt_ids(
            "temporizador contador", citations_from_results(results), max_new=96
        )
        decoded = self.tokenizer.decode(prompt_ids)
        self.assertLessEqual(effective, 64)
        self.assertLessEqual(len(prompt_ids), 128 - effective)
        self.assertIn("temporizador", decoded)
        self.assertIn("contador", decoded)
        self.assertIn("[E1]", decoded)

    def test_grounding_rejects_changed_literals_and_polarity(self) -> None:
        citation = Citation(
            evidence_id="E1",
            document_id="doc_a",
            document_sha256="a" * 64,
            filename="limites.pdf",
            page=1,
            section="",
            chunk_id="chk_test",
            bbox=(0, 0, 1, 1),
            quote="O limite permitido e 10 unidades.",
            score=1.0,
        )
        changed = verify_grounding(
            "O limite nao permitido e 20 unidades [E1].", [citation]
        )
        self.assertFalse(changed.grounded)
        self.assertEqual(changed.supported_claims, 0)
        exact = verify_grounding(
            "O limite permitido e 10 unidades [e1].", [citation]
        )
        self.assertTrue(exact.grounded)

    def test_two_single_chunk_documents_train_contrastively(self) -> None:
        pairs = build_positive_pairs(self.chunks)
        self.assertGreaterEqual(len(pairs), 2)
        self.assertEqual(
            {pair.anchor_chunk_id for pair in pairs if pair.anchor_chunk_id == pair.positive_chunk_id},
            {chunk.chunk_id for chunk in self.chunks},
        )
        encoder = RetrieverEncoder(
            RetrieverConfig(
                vocab_size=self.tokenizer.vocab_actual_size,
                context_len=32,
                embed_dim=16,
                projection_dim=8,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                dropout=0.0,
            )
        )
        trainer = ContrastiveTrainer(
            encoder,
            self.tokenizer,
            self.chunks,
            device="cpu",
            seed=31,
            sampling_payload={
                "format_version": 1,
                "corpus_sha256": corpus_sha256(self.chunks),
                "signal_version": 4,
                "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
                "weights": [0.8, 0.2],
            },
        )
        history = trainer.train(epochs=1, batch_size=2)
        self.assertTrue(math.isfinite(history[0]["loss"]))
        self.assertTrue(encoder.trained_on_corpus)
        uncertainty = trainer.measure_retrieval_uncertainty()
        self.assertEqual(set(uncertainty), {chunk.chunk_id for chunk in self.chunks})
        self.assertTrue(all(0.0 <= value <= 1.0 for value in uncertainty.values()))
        self.assertTrue(trainer.sampling_metadata["weighted"])

    def test_vector_store_checks_dimensions_and_upserts(self) -> None:
        store = VectorStore()
        metadata = [self.chunks[0].to_dict(), self.chunks[1].to_dict()]
        store.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), metadata)
        store.add(np.asarray([[0.8, 0.2]], dtype=np.float32), [metadata[0]])
        self.assertEqual(len(store), 2)
        self.assertEqual(store.search(np.asarray([1.0, 0.0]), 1)[0]["chunk_id"], metadata[0]["chunk_id"])
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "index"
            store.save(prefix)
            loaded = VectorStore()
            loaded.load(prefix)
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
