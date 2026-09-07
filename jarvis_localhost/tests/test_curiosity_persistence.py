from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from jarvis_localhost.corpus.provenance import (
    CanonicalChunk,
    DocumentIdentity,
)
from jarvis_localhost.curiosity.engine import CuriosityEngine, Insight
from jarvis_localhost.tests.corpus_fixture import write_closed_corpus


def make_chunks(label: str, count: int = 4) -> list[CanonicalChunk]:
    identities = [
        DocumentIdentity(
            f"doc_{label}_{index}",
            hashlib.sha256(f"{label}-{index}".encode("utf-8")).hexdigest(),
            f"{label}-{index}.pdf",
            1_000 + index,
        )
        for index in range(2)
    ]
    return [
        CanonicalChunk.build(
            identities[index % 2],
            page=index + 1,
            section="",
            bbox=(0, 0, 1, 1),
            text=(
                f"Corpus {label} trecho {index} registra uma medição exclusiva "
                f"componente_{label}_{index} e preserva sua proveniência local."
            ),
            ordinal=index // 2,
        )
        for index in range(count)
    ]


def replace_corpus(corpus_dir: Path, chunks: list[CanonicalChunk]) -> None:
    write_closed_corpus(corpus_dir, chunks)


def optimizer_steps(optimizer: torch.optim.Optimizer) -> list[float]:
    steps: list[float] = []
    for state in optimizer.state.values():
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            steps.append(float(step.detach().cpu().item()))
        elif step is not None:
            steps.append(float(step))
    return sorted(steps)


class CuriosityPersistenceTests(unittest.TestCase):
    def test_generation_resumes_networks_optimizers_and_bound_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_dir = root / "curiosity"
            corpus_dir.mkdir()
            chunks = make_chunks("resume")
            replace_corpus(corpus_dir, chunks)

            engine = CuriosityEngine(
                corpus_dir, output_dir, device="cpu", top_k=3, seed=19
            )
            first_metrics = engine.run_cycle()
            self.assertFalse(first_metrics.get("skipped", False))
            self.assertIsNotNone(engine._agent)
            assert engine._agent is not None
            policy_before = {
                name: tensor.detach().cpu().clone()
                for name, tensor in engine._agent.policy.state_dict().items()
            }
            icm_steps_before = optimizer_steps(engine._agent.icm_optimizer)
            policy_steps_before = optimizer_steps(engine._agent.policy_optimizer)
            self.assertTrue(icm_steps_before)
            self.assertTrue(policy_steps_before)
            cycle_before = engine.get_stats()["cycle"]
            current = (
                output_dir
                / "corpora"
                / first_metrics["corpus_sha256"]
                / "CURRENT"
            )
            generation = current.read_text(encoding="utf-8").strip()
            generation_dir = current.parent / "generations" / generation
            manifest = json.loads(
                (generation_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(set(manifest["files"]), {
                "agent.pt",
                "curriculum.json",
                "insights.json",
                "sampling_weights.json",
                "state.json",
            })

            resumed = CuriosityEngine(
                corpus_dir, output_dir, device="cpu", top_k=3, seed=19
            )
            resumed.get_sampling_weights()
            self.assertEqual(resumed.get_stats()["cycle"], cycle_before)
            self.assertEqual(resumed.get_stats()["generation"], generation)
            assert resumed._agent is not None
            for name, tensor in resumed._agent.policy.state_dict().items():
                self.assertTrue(torch.equal(policy_before[name], tensor.detach().cpu()))
            self.assertEqual(
                optimizer_steps(resumed._agent.icm_optimizer), icm_steps_before
            )
            self.assertEqual(
                optimizer_steps(resumed._agent.policy_optimizer),
                policy_steps_before,
            )

            wrong_seed = CuriosityEngine(
                corpus_dir, output_dir, device="cpu", top_k=3, seed=20
            )
            wrong_seed.get_sampling_weights()
            self.assertEqual(wrong_seed.get_stats()["cycle"], 0)
            self.assertIn(
                "configuration mismatch",
                wrong_seed.get_stats()["last_metrics"]["checkpoint_load_error"],
            )

            agent_path = generation_dir / "agent.pt"
            agent_path.write_bytes(agent_path.read_bytes() + b"tamper")
            corrupted = CuriosityEngine(
                corpus_dir, output_dir, device="cpu", top_k=3, seed=19
            )
            corrupted.get_sampling_weights()
            self.assertEqual(corrupted.get_stats()["cycle"], 0)
            self.assertEqual(corrupted.insights, {})
            assert corrupted._agent is not None
            self.assertEqual(optimizer_steps(corrupted._agent.icm_optimizer), [])
            self.assertIn(
                "checksum failed",
                corrupted.get_stats()["last_metrics"]["checkpoint_load_error"],
            )

    def test_corpus_switch_never_leaks_and_switch_back_resumes_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_dir = root / "curiosity"
            corpus_dir.mkdir()
            chunks_a = make_chunks("alpha", 5)
            chunks_b = make_chunks("beta", 5)
            replace_corpus(corpus_dir, chunks_a)
            engine = CuriosityEngine(
                corpus_dir, output_dir, device="cpu", top_k=5, seed=23
            )
            metrics_a = engine.run_cycle()
            ids_a = set(engine.insights)
            self.assertTrue(ids_a)

            replace_corpus(corpus_dir, chunks_b)
            metrics_b = engine.run_cycle()
            ids_b = set(engine.insights)
            self.assertNotEqual(
                metrics_a["corpus_sha256"], metrics_b["corpus_sha256"]
            )
            self.assertTrue(ids_b)
            self.assertTrue(ids_a.isdisjoint(ids_b))
            chunk_ids_b = {chunk.chunk_id for chunk in chunks_b}
            self.assertTrue(
                all(insight.chunk_id in chunk_ids_b for insight in engine.insights.values())
            )
            self.assertEqual(
                len(engine._seen_vectors),
                len({insight.chunk_id for insight in engine.insights.values()}),
            )

            replace_corpus(corpus_dir, chunks_a)
            engine.get_sampling_weights()
            self.assertEqual(engine.state.corpus_sha256, metrics_a["corpus_sha256"])
            self.assertEqual(set(engine.insights), ids_a)

    def test_signals_are_measured_versioned_and_persisted_without_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_dir = root / "curiosity"
            corpus_dir.mkdir()
            chunks = make_chunks("signals")
            replace_corpus(corpus_dir, chunks)
            engine = CuriosityEngine(corpus_dir, output_dir, device="cpu", seed=29)

            initial = engine.get_sampling_weights()
            self.assertEqual(initial["signal_version"], 0)
            self.assertTrue(
                np.allclose(initial["weights"], [1.0 / len(chunks)] * len(chunks))
            )
            updated = engine.update_learning_signals(
                lm_loss_by_chunk={chunks[0].chunk_id: 3.0},
                retrieval_uncertainty_by_chunk={chunks[1].chunk_id: 2.0},
            )
            self.assertGreater(updated["signal_version"], 0)
            self.assertAlmostEqual(sum(updated["weights"]), 1.0)
            assert engine._agent is not None and engine._curriculum is not None
            untouched = engine._curriculum.signals[chunks[2].chunk_id]
            self.assertEqual(untouched.lm_loss, 0.0)
            self.assertEqual(untouched.retrieval_uncertainty, 0.0)
            untouched_index = engine._agent.environment._index_by_id[
                chunks[2].chunk_id
            ]
            self.assertFalse(engine._agent.environment._lm_loss_known[untouched_index])
            self.assertFalse(
                engine._agent.environment._retrieval_uncertainty_known[
                    untouched_index
                ]
            )

            resumed = CuriosityEngine(corpus_dir, output_dir, device="cpu", seed=29)
            persisted = resumed.get_sampling_weights()
            self.assertEqual(
                persisted["signal_version"], updated["signal_version"]
            )
            self.assertTrue(np.allclose(persisted["weights"], updated["weights"]))
            assert resumed._curriculum is not None
            self.assertEqual(
                resumed._curriculum.signals[chunks[0].chunk_id].lm_loss, 3.0
            )
            self.assertEqual(
                resumed._curriculum.signals[
                    chunks[1].chunk_id
                ].retrieval_uncertainty,
                2.0,
            )

    def test_cycles_serialize_lifecycle_stops_and_memory_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_dir = root / "curiosity"
            corpus_dir.mkdir()
            replace_corpus(corpus_dir, make_chunks("mutex"))
            engine = CuriosityEngine(
                corpus_dir,
                output_dir,
                device="cpu",
                cycle_interval=0.05,
                seed=31,
            )
            engine.get_sampling_weights()
            assert engine._agent is not None
            original = engine._agent.train_cycle
            counter_lock = threading.Lock()
            active = 0
            maximum = 0

            def measured_cycle():
                nonlocal active, maximum
                with counter_lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.03)
                    return original()
                finally:
                    with counter_lock:
                        active -= 1

            engine._agent.train_cycle = measured_cycle  # type: ignore[method-assign]
            workers = [threading.Thread(target=engine.run_cycle) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10.0)
                self.assertFalse(worker.is_alive())
            self.assertEqual(maximum, 1)

            self.assertTrue(engine.start())
            first_thread = engine._thread
            self.assertFalse(engine.start())
            self.assertIs(engine._thread, first_thread)
            self.assertTrue(engine.stop(timeout=10.0))
            self.assertEqual(engine.get_stats()["lifecycle"], "stopped")
            self.assertIsNone(engine._thread)
            self.assertTrue(engine.start())
            self.assertTrue(engine.stop(timeout=10.0))

            engine.MAX_INSIGHTS = 3
            engine.insights = {
                f"ins_{index}": Insight(
                    id=f"ins_{index}",
                    source="bounded.pdf",
                    chunk_text=f"vetor limitado {index}",
                    summary=f"resumo {index}",
                    tags=[f"tag_{index % 2}"],
                    curiosity_score=float(index),
                    novelty_score=1.0,
                    entropy_score=1.0,
                    surprise_score=1.0,
                    connections=[f"ins_{value}" for value in range(8)],
                    timestamp=float(index),
                    chunk_id=f"chunk_{index}",
                    document_sha256="a" * 64,
                )
                for index in range(8)
            }
            engine._prune_insights()
            self.assertEqual(len(engine.insights), 3)
            self.assertEqual(len(engine._seen_vectors), 3)
            self.assertEqual(engine._vector_matrix.shape[0], 3)
            active_ids = set(engine.insights)
            self.assertTrue(
                all(
                    set(identifiers).issubset(active_ids)
                    for identifiers in engine.state.topic_index.values()
                )
            )
            self.assertTrue(
                all(
                    set(insight.connections).issubset(active_ids)
                    for insight in engine.insights.values()
                )
            )


if __name__ == "__main__":
    unittest.main()
