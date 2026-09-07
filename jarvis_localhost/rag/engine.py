"""Strict/extractive and generative RAG with mandatory grounding gates."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

import torch

from jarvis_localhost.ai.language_model import JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.corpus.chunker import PageSpan, canonicalize_spans
from jarvis_localhost.corpus.provenance import CanonicalChunk, DocumentIdentity
from jarvis_localhost.planning import ResponsePlanner, ResponseOutline
from jarvis_localhost.rag.citations import Citation, citations_from_results
from jarvis_localhost.rag.grounding import (
    GroundingReport,
    extractive_answer,
    verify_grounding,
)
from jarvis_localhost.retrieval.retriever import SovereignRetriever
from jarvis_localhost.retrieval.vector_store import VectorStore


class RAGMode(str, Enum):
    STRICT = "strict"
    RAG = "rag"


def _coerce_mode(value: RAGMode | str) -> RAGMode:
    return value if isinstance(value, RAGMode) else RAGMode(str(value).lower())


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    mode: str
    method: str
    abstained: bool
    citations: tuple[Citation, ...]
    grounding: GroundingReport | None
    retrieval_confidence: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "method": self.method,
            "abstained": self.abstained,
            "sources": [citation.to_dict() for citation in self.citations],
            "citations": [citation.to_dict() for citation in self.citations],
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "retrieval_confidence": self.retrieval_confidence,
            "reason": self.reason,
        }


class RAGEngine:
    """Evidence-first RAG. Generative output never bypasses grounding checks."""

    def __init__(
        self,
        model: JarvisTransformer | None,
        tokenizer: JarvisTokenizer,
        store: VectorStore | None = None,
        device: Any = "cpu",
        *,
        retriever: SovereignRetriever | None = None,
        mode: RAGMode | str | None = None,
        minimum_retrieval_confidence: float = 0.18,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.retriever = retriever or SovereignRetriever(
            tokenizer, encoder=None, store=store, device=device
        )
        configured_mode = mode or os.getenv("JARVIS_RAG_MODE", RAGMode.STRICT.value)
        self.mode = _coerce_mode(configured_mode)
        self.minimum_retrieval_confidence = minimum_retrieval_confidence
        self.planner = ResponsePlanner()

    def index_chunks(self, chunks: Iterable[CanonicalChunk]) -> int:
        return self.retriever.index(chunks)

    def index_document(self, text: str, source: str) -> int:
        """Compatibility entrypoint; new callers should pass canonical chunks."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = DocumentIdentity(
            document_id=f"doc_{digest[:24]}",
            document_sha256=digest,
            filename=source,
            byte_size=len(text.encode("utf-8")),
        )
        chunks = canonicalize_spans(
            identity,
            [PageSpan(page=1, text=text, bbox=(0, 0, 0, 0))],
            target_words=220,
            overlap_words=40,
            minimum_words=3,
        )
        return self.index_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return [result.to_dict() for result in self.retriever.retrieve(query, top_k)]

    def _effective_generation_tokens(self, requested: int) -> int:
        if self.model is None:
            return max(0, int(requested))
        context_len = self.model.cfg.context_len
        if context_len < 8:
            raise ValueError("RAG requires a language-model context of at least 8 tokens")
        # Keep at least half of compact contexts (and at least 24 tokens when
        # possible) for the question, an evidence marker and quoted evidence.
        minimum_prompt = min(context_len - 1, max(24, context_len // 2))
        return min(max(1, int(requested)), context_len - minimum_prompt)

    def _prompt_ids(
        self,
        question: str,
        citations: list[Citation],
        *,
        max_new: int,
    ) -> list[int]:
        if self.model is None:
            return []
        effective_max_new = self._effective_generation_tokens(max_new)
        budget = self.model.cfg.context_len - effective_max_new
        question_ids = self.tokenizer.encode(question.strip(), add_special=False)
        evidence_ids_full: list[int] = []
        for citation in citations:
            # Provenance labels stay in the response metadata. The generation
            # prompt needs only the stable marker and exact authorized quote.
            evidence_ids_full.extend(
                self.tokenizer.encode(
                    f"{citation.marker} {citation.quote}\n", add_special=False
                )
            )

        structures = (
            ("Pergunta:\n", "\nEvidências:\n", "\nResposta:"),
            ("Q:\n", "\nE:\n", "\nA:"),
        )
        question_header: list[int] = []
        evidence_header: list[int] = []
        suffix_ids: list[int] = []
        payload_budget = -1
        for question_label, evidence_label, response_label in structures:
            question_header = self.tokenizer.encode(
                question_label, add_special=False
            )
            evidence_header = self.tokenizer.encode(
                evidence_label, add_special=False
            )
            suffix_ids = self.tokenizer.encode(response_label, add_special=False)
            fixed_count = (
                len(question_header) + len(evidence_header) + len(suffix_ids) + 2
            )
            payload_budget = budget - fixed_count
            if payload_budget >= (5 if evidence_ids_full else 1):
                break
        if payload_budget < (5 if evidence_ids_full else 1):
            raise ValueError("model context cannot preserve question and useful evidence")

        minimum_evidence = (
            min(len(evidence_ids_full), max(4, payload_budget // 3))
            if evidence_ids_full
            else 0
        )
        question_budget = max(1, payload_budget - minimum_evidence)
        selected_question = question_ids[:question_budget]
        evidence_budget = payload_budget - len(selected_question)
        selected_evidence = evidence_ids_full[:evidence_budget]
        prefix_ids = [
            *question_header,
            *selected_question,
            *evidence_header,
        ]
        return [
            self.tokenizer.BOS_ID,
            *prefix_ids,
            *selected_evidence,
            *suffix_ids,
            self.tokenizer.EOS_ID,
        ][:budget]

    @torch.no_grad()
    def answer(
        self,
        query: str,
        top_k: int = 5,
        max_new: int = 96,
        temperature: float = 0.65,
        *,
        mode: RAGMode | str | None = None,
    ) -> dict:
        active_mode = _coerce_mode(mode or self.mode)
        results = self.retriever.retrieve(query, top_k=top_k)
        q_lower = query.lower()
        if " e " in q_lower or " ou " in q_lower or " vs " in q_lower or "versus" in q_lower:
            parts = re.split(r"\b(?:e|ou|versus|vs)\b", q_lower)
            seen_ids = {r.chunk.chunk_id for r in results}
            for part in parts:
                part = part.strip()
                if len(part) > 3:
                    sub_results = self.retriever.retrieve(part, top_k=3)
                    for sr in sub_results:
                        if sr.chunk.chunk_id not in seen_ids:
                            seen_ids.add(sr.chunk.chunk_id)
                            results.append(sr)
        confidence = results[0].score if results else 0.0
        citations = citations_from_results(results)
        if not citations or confidence < self.minimum_retrieval_confidence:
            response = RAGResponse(
                answer=(
                    "Não encontrei evidência suficiente nos documentos autorizados "
                    "para responder com segurança."
                ),
                mode=active_mode.value,
                method="abstention",
                abstained=True,
                citations=tuple(citations),
                grounding=None,
                retrieval_confidence=confidence,
                reason="retrieval_below_threshold",
            )
            return response.to_dict()

        fallback = extractive_answer(query, citations)
        if not fallback:
            response = RAGResponse(
                answer="As fontes recuperadas não contêm uma passagem diretamente responsiva.",
                mode=active_mode.value,
                method="abstention",
                abstained=True,
                citations=tuple(citations),
                grounding=None,
                retrieval_confidence=confidence,
                reason="no_supported_passage",
            )
            return response.to_dict()

        if active_mode is RAGMode.STRICT or self.model is None:
            report = verify_grounding(fallback, citations)
            return RAGResponse(
                answer=fallback,
                mode=active_mode.value,
                method="extractive",
                abstained=False,
                citations=tuple(citations),
                grounding=report,
                retrieval_confidence=confidence,
            ).to_dict()

        effective_max_new = self._effective_generation_tokens(max_new)
        prompt_ids = self._prompt_ids(
            query, citations, max_new=effective_max_new
        )
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            prompt,
            max_new=effective_max_new,
            temperature=temperature,
            top_k=40,
        )
        answer_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
        candidate = self.tokenizer.decode(answer_ids).strip()
        report = verify_grounding(candidate, citations)
        if not candidate or not report.grounded:
            fallback_report = verify_grounding(fallback, citations)
            return RAGResponse(
                answer=fallback,
                mode=active_mode.value,
                method="extractive_fallback",
                abstained=False,
                citations=tuple(citations),
                grounding=fallback_report,
                retrieval_confidence=confidence,
                reason="generated_answer_failed_grounding",
            ).to_dict()
        return RAGResponse(
            answer=candidate,
            mode=active_mode.value,
            method="grounded_generation",
            abstained=False,
            citations=tuple(citations),
            grounding=report,
            retrieval_confidence=confidence,
        ).to_dict()


__all__ = ["RAGEngine", "RAGMode", "RAGResponse"]
