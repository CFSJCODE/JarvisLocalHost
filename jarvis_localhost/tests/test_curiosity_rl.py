from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jarvis_localhost.corpus.provenance import CanonicalChunk, DocumentIdentity
from jarvis_localhost.curiosity.curriculum import CurriculumManager
from jarvis_localhost.curiosity.engine import CuriosityEngine
from jarvis_localhost.curiosity.environment import (
    CorpusAction,
    CorpusEnv,
    hashed_corpus_vector,
)
from jarvis_localhost.curiosity.icm import IntrinsicCuriosityModule
from jarvis_localhost.curiosity.policy import ActorCriticPolicy
from jarvis_localhost.curiosity.ppo import CuriosityAgent, PPOConfig
from jarvis_localhost.tests.corpus_fixture import write_closed_corpus


def curiosity_chunks() -> list[CanonicalChunk]:
    first = DocumentIdentity("doc_a", "a" * 64, "a.pdf", 100)
    second = DocumentIdentity("doc_b", "b" * 64, "b.pdf", 100)
    texts = [
        (first, 1, "O motor recebe energia e aciona o eixo principal."),
        (first, 2, "O eixo principal transmite movimento ao conjunto mecânico."),
        (second, 1, "O sensor mede a rotação do eixo e registra cada amostra."),
        (second, 2, "As amostras registradas permitem calcular a velocidade de rotação."),
    ]
    return [
        CanonicalChunk.build(
            document,
            page=page,
            section="",
            bbox=(0, 0, 1, 1),
            text=text,
            ordinal=index % 2,
        )
        for index, (document, page, text) in enumerate(texts)
    ]


class CorpusEnvironmentTests(unittest.TestCase):
    def test_feature_hashing_and_transitions_are_stable(self) -> None:
        text = "eixo rotação eixo sensor"
        self.assertTrue(
            np.array_equal(
                hashed_corpus_vector(text, 32), hashed_corpus_vector(text, 32)
            )
        )
        environment = CorpusEnv(curiosity_chunks(), text_dimension=32, seed=4)
        observation = environment.reset(start_index=0)
        self.assertEqual(observation.shape, (environment.observation_dim,))
        next_observation, reward, done, info = environment.step(CorpusAction.NEXT)
        self.assertEqual(info["transition"].next_chunk_id, environment.chunks[1].chunk_id)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(next_observation.dtype, np.float32)

    def test_icm_reward_and_losses_are_real_tensors(self) -> None:
        module = IntrinsicCuriosityModule(24, 7, feature_dim=12, hidden_dim=16)
        states = torch.randn(5, 24)
        next_states = torch.randn(5, 24)
        actions = torch.tensor([0, 1, 2, 3, 4])
        output = module.calculate(states, actions, next_states)
        self.assertEqual(tuple(output.intrinsic_reward.shape), (5,))
        self.assertTrue(bool((output.intrinsic_reward >= 0).all().item()))
        output.total_loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))

    def test_selected_policy_log_probabilities_backpropagate(self) -> None:
        policy = ActorCriticPolicy(12, 7, hidden_dim=16)
        features = torch.randn(5, 12)
        actions = torch.tensor([0, 1, 2, 3, 4])
        selected, values, entropy = policy.evaluate_actions(features, actions)
        logits, _ = policy(features)
        expected = torch.log_softmax(logits, dim=-1)[
            torch.arange(actions.shape[0]), actions
        ]
        self.assertTrue(torch.allclose(selected, expected))
        loss = -selected.mean() + values.pow(2).mean() - 0.01 * entropy.mean()
        loss.backward()
        self.assertIsNotNone(policy.actor.weight.grad)
        self.assertTrue(bool(torch.isfinite(policy.actor.weight.grad).all().item()))

    def test_ppo_cycle_updates_policy_and_icm(self) -> None:
        environment = CorpusEnv(
            curiosity_chunks(), text_dimension=24, max_episode_steps=8, seed=5
        )
        agent = CuriosityAgent(
            environment,
            feature_dim=16,
            config=PPOConfig(
                rollout_steps=12,
                update_epochs=1,
                minibatch_size=6,
            ),
            seed=5,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in agent.policy.named_parameters()
        }
        cycle = agent.train_cycle()
        self.assertEqual(cycle.metrics["algorithm"], "ICM+PPO")
        self.assertEqual(len(cycle.records), 12)
        self.assertTrue(math.isfinite(cycle.metrics["icm_forward_loss"]))
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in agent.policy.named_parameters()
            )
        )


class CurriculumEngineTests(unittest.TestCase):
    def test_curriculum_uses_reward_loss_uncertainty_and_visit_penalty(self) -> None:
        chunks = curiosity_chunks()
        manager = CurriculumManager(chunks)
        first = manager.update(
            chunks[0].chunk_id,
            intrinsic_reward=2.0,
            lm_loss=1.0,
            retrieval_uncertainty=0.5,
        )
        initial = first.priority
        for _ in range(50):
            manager.update(chunks[0].chunk_id, visit_increment=1)
        self.assertLess(manager.signals[chunks[0].chunk_id].priority, initial)

    def test_engine_emits_lineage_bound_insights_and_manifest(self) -> None:
        chunks = curiosity_chunks()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "corpus"
            output_dir = root / "curiosity"
            corpus_dir.mkdir()
            write_closed_corpus(corpus_dir, chunks)
            emitted = []
            engine = CuriosityEngine(
                corpus_dir,
                output_dir,
                on_insight=emitted.append,
                device="cpu",
                top_k=2,
                seed=7,
            )
            metrics = engine.run_cycle()
            self.assertEqual(metrics["algorithm"], "ICM+PPO")
            self.assertTrue((output_dir / "icm_ppo.pt").is_file())
            manifest = json.loads(
                (output_dir / "icm_ppo.pt.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["initialized_from"], "random")
            self.assertFalse(manifest["external_weights"])
            self.assertEqual(manifest["corpus_sha256"], metrics["corpus_sha256"])
            self.assertTrue(emitted)
            self.assertTrue(all(insight.chunk_id.startswith("chk_") for insight in emitted))
            self.assertTrue(all(insight.document_sha256 for insight in emitted))


if __name__ == "__main__":
    unittest.main()
