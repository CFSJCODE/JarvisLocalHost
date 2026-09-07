from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jarvis_localhost.retrieval.encoder import RetrieverConfig, RetrieverEncoder
from jarvis_localhost.retrieval.vector_store import VectorStore, _sha256
from jarvis_localhost.sovereign import SovereignPolicy


class RetrieverArtifactTests(unittest.TestCase):
    def _encoder(self, seed: int = 7) -> RetrieverEncoder:
        encoder = RetrieverEncoder(
            RetrieverConfig(
                vocab_size=16,
                context_len=8,
                embed_dim=8,
                projection_dim=4,
                num_heads=2,
                num_layers=1,
                ff_dim=16,
                dropout=0.0,
            ),
            seed=seed,
            policy=SovereignPolicy(enabled=True),
        )
        encoder.trained_on_corpus = True
        return encoder

    def test_retriever_checkpoint_is_bound_to_manifest_and_expected_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retriever.pt"
            encoder = self._encoder()
            encoder.save(path, corpus_sha256="c" * 64, tokenizer_sha256="t" * 64)

            loaded = RetrieverEncoder.load(
                path,
                policy=SovereignPolicy(enabled=True),
                expected_corpus_sha256="c" * 64,
                expected_tokenizer_sha256="t" * 64,
            )
            self.assertTrue(loaded.trained_on_corpus)

            replacement = self._encoder(seed=99)
            torch.save(
                {
                    "format_version": replacement.CHECKPOINT_VERSION,
                    "config": replacement.cfg.__dict__,
                    "seed": replacement.seed,
                    "trained_on_corpus": True,
                    "state_dict": replacement.state_dict(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "size|checksum"):
                RetrieverEncoder.load(
                    path,
                    policy=SovereignPolicy(enabled=True),
                    expected_corpus_sha256="c" * 64,
                    expected_tokenizer_sha256="t" * 64,
                )

    def test_sovereign_retriever_load_requires_explicit_expected_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retriever.pt"
            self._encoder().save(
                path, corpus_sha256="c" * 64, tokenizer_sha256="t" * 64
            )
            with self.assertRaisesRegex(ValueError, "requires expected lineage"):
                RetrieverEncoder.load(path, policy=SovereignPolicy(enabled=True))


class VectorArtifactTests(unittest.TestCase):
    def test_vector_generation_is_finite_unique_normalized_and_lineage_bound(self) -> None:
        store = VectorStore()
        store.add(
            np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32),
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "vectors"
            store.save(
                prefix,
                corpus_sha256="c" * 64,
                tokenizer_sha256="t" * 64,
                encoder_sha256="e" * 64,
            )
            loaded = VectorStore()
            loaded.load(
                prefix,
                expected_corpus_sha256="c" * 64,
                expected_tokenizer_sha256="t" * 64,
                expected_encoder_sha256="e" * 64,
            )
            self.assertEqual(2, len(loaded))
            self.assertTrue(
                np.allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0)
            )
            with self.assertRaisesRegex(ValueError, "corpus_sha256"):
                VectorStore().load(prefix, expected_corpus_sha256="x" * 64)

    def test_vector_store_rejects_nonfinite_duplicate_and_incomplete_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            VectorStore().add(
                np.asarray([[np.nan, 1.0]], dtype=np.float32),
                [{"chunk_id": "bad"}],
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            VectorStore().add(
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                [{"chunk_id": "same"}, {"chunk_id": "same"}],
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                VectorStore().load(Path(directory) / "missing")

    def test_semantically_forged_metadata_is_rejected_even_with_updated_hash(self) -> None:
        store = VectorStore()
        store.add(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "vectors"
            store.save(prefix)
            _, metadata_path, manifest_path = VectorStore._paths(prefix)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata[1]["chunk_id"] = "a"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata_sha256"] = _sha256(metadata_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                VectorStore().load(prefix)


if __name__ == "__main__":
    unittest.main()
