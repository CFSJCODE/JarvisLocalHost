"""Exercise the actual Jarvis neural stack on the selected Windows backend."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jarvis_localhost.ai.language_model import JarvisConfig, JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.corpus.provenance import (
    CanonicalChunk,
    DocumentIdentity,
    corpus_sha256,
)
from jarvis_localhost.curiosity.environment import CorpusEnv
from jarvis_localhost.curiosity.icm import IntrinsicCuriosityModule
from jarvis_localhost.curiosity.policy import ActorCriticPolicy
from jarvis_localhost.curiosity.ppo import CuriosityAgent, PPOConfig
from jarvis_localhost.hardware import detect_hardware, select_compute_device
from jarvis_localhost.retrieval.contrastive import ContrastiveTrainer, info_nce
from jarvis_localhost.retrieval.encoder import RetrieverConfig, RetrieverEncoder


def _finite(value: torch.Tensor) -> float:
    number = float(value.detach().cpu().item())
    if not math.isfinite(number):
        raise FloatingPointError("non-finite smoke-test value")
    return number


def run(
    *,
    require_backend: str | None = None,
    require_accelerator: bool = False,
) -> dict:
    hardware = detect_hardware()
    descriptor = select_compute_device(hardware)
    if require_backend and descriptor.backend.casefold() != require_backend.casefold():
        raise RuntimeError(
            f"required backend {require_backend!r}, selected {descriptor.backend!r}"
        )
    if require_accelerator and not descriptor.accelerated:
        raise RuntimeError("an accelerator was required but the selected device is CPU")
    device = descriptor.torch_device
    corpus = (
        "motor eixo sensor rotação energia controle amostra velocidade "
        "motor eixo sensor rotação energia controle amostra velocidade"
    )
    tokenizer = JarvisTokenizer(vocab_size=96)
    tokenizer.train(corpus)

    language_model = JarvisTransformer(
        JarvisConfig(
            vocab_size=tokenizer.vocab_actual_size,
            context_len=32,
            embed_dim=24,
            num_heads=4,
            num_layers=1,
            ff_dim=48,
            dropout=0.0,
        ),
        seed=17,
    ).to(device)
    inputs = torch.tensor(
        [[tokenizer.BOS_ID, *tokenizer.encode("motor eixo", False)][:16]],
        dtype=torch.long,
        device=device,
    )
    targets = torch.cat((inputs[:, 1:], inputs[:, -1:]), dim=1)
    optimizer = torch.optim.AdamW(
        language_model.parameters(), lr=1e-3, foreach=False
    )
    optimizer.zero_grad(set_to_none=True)
    _, language_loss = language_model(inputs, targets)
    assert language_loss is not None
    language_loss.backward()
    optimizer.step()
    generated = language_model.generate(inputs[:, :2], max_new=2, top_k=8)

    retriever = RetrieverEncoder(
        RetrieverConfig(
            vocab_size=tokenizer.vocab_actual_size,
            context_len=24,
            embed_dim=24,
            projection_dim=12,
            num_heads=4,
            num_layers=1,
            ff_dim=48,
            dropout=0.0,
        ),
        seed=19,
    ).to(device)
    ids, mask = retriever.tokenize(
        tokenizer,
        ["motor eixo sensor", "eixo sensor rotação"],
        device,
    )
    dense_optimizer = torch.optim.AdamW(
        retriever.parameters(), lr=1e-3, foreach=False
    )
    dense_optimizer.zero_grad(set_to_none=True)
    embeddings = retriever(ids, mask)
    dense_loss = info_nce(embeddings, embeddings.roll(1, 0), 0.1)
    dense_loss.backward()
    dense_optimizer.step()

    icm = IntrinsicCuriosityModule(20, 7, feature_dim=12, hidden_dim=24).to(device)
    policy = ActorCriticPolicy(12, 7, hidden_dim=24).to(device)
    states = torch.randn(4, 20, device=device)
    next_states = torch.randn(4, 20, device=device)
    actions = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    curiosity = icm.calculate(states, actions, next_states)
    features = icm.encoder(states)
    log_probability, values, entropy = policy.evaluate_actions(features, actions)
    old_log_probability = log_probability.detach() - 0.01
    ratio = torch.exp(log_probability - old_log_probability)
    # Positive, non-uniform advantages avoid a numerically cancelling surrogate
    # and prove the selected-log-probability branch contributes to backward.
    advantages = torch.linspace(0.5, 1.5, steps=4, device=device)
    ppo_loss = -torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 0.8, 1.2) * advantages,
    ).mean()
    curiosity_loss = (
        curiosity.total_loss
        + ppo_loss
        + values.pow(2).mean()
        - 0.01 * entropy.mean()
    )
    curiosity_loss.backward()

    identity = DocumentIdentity(
        "doc_directml_smoke",
        "d" * 64,
        "directml-smoke.pdf",
        128,
    )
    smoke_chunks = [
        CanonicalChunk.build(
            identity,
            page=index + 1,
            section="",
            bbox=(0, 0, 1, 1),
            text=text,
            ordinal=index,
        )
        for index, text in enumerate(
            (
                "O motor aciona o eixo principal.",
                "O sensor mede a rotação do eixo.",
                "A amostra registra a velocidade local.",
            )
        )
    ]
    environment = CorpusEnv(
        smoke_chunks,
        text_dimension=20,
        max_episode_steps=4,
        seed=29,
    )
    ppo_agent = CuriosityAgent(
        environment,
        feature_dim=12,
        device=device,
        config=PPOConfig(
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=4,
        ),
        seed=29,
    )
    ppo_cycle = ppo_agent.train_cycle()
    contrastive = ContrastiveTrainer(
        retriever,
        tokenizer,
        smoke_chunks,
        device=device,
        temperature=0.1,
        seed=37,
        sampling_payload={
            "format_version": 1,
            "corpus_sha256": corpus_sha256(smoke_chunks),
            "signal_version": 1,
            "chunk_ids": [chunk.chunk_id for chunk in smoke_chunks],
            "weights": [0.5, 0.3, 0.2],
        },
    )
    contrastive_history = contrastive.train(epochs=1, batch_size=3)
    retrieval_uncertainty = contrastive.measure_retrieval_uncertainty()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "jarvis.pt"
        language_model.save(
            checkpoint,
            corpus_sha256=tokenizer.corpus_sha256,
            tokenizer_sha256=tokenizer.fingerprint(),
        )
        loaded = JarvisTransformer.load(
            checkpoint,
            device="cpu",
            expected_corpus_sha256=tokenizer.corpus_sha256,
            expected_tokenizer_sha256=tokenizer.fingerprint(),
        )
        checkpoint_parameters = loaded.count_params()

    return {
        "backend": descriptor.backend,
        "accelerated": descriptor.accelerated,
        "smoke_tested": descriptor.smoke_tested,
        "device": descriptor.to_dict().get("device"),
        "language_model": {
            "forward_loss": _finite(language_loss),
            "generated_tokens": int(generated.shape[1] - 2),
            "checkpoint_parameters": checkpoint_parameters,
        },
        "retriever": {
            "contrastive_loss": _finite(dense_loss),
            "embedding_shape": list(embeddings.shape),
            "trainer_loss": float(contrastive_history[-1]["loss"]),
            "uncertainty_count": len(retrieval_uncertainty),
        },
        "curiosity": {
            "total_loss": _finite(curiosity.total_loss),
            "mean_reward": _finite(curiosity.intrinsic_reward.mean()),
            "policy_log_probability": _finite(log_probability.mean()),
            "ppo_loss": _finite(ppo_loss),
            "cycle_policy_loss": float(ppo_cycle.metrics["policy_loss"]),
            "cycle_icm_forward_loss": float(
                ppo_cycle.metrics["icm_forward_loss"]
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-backend")
    parser.add_argument("--require-accelerator", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                require_backend=arguments.require_backend,
                require_accelerator=arguments.require_accelerator,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
