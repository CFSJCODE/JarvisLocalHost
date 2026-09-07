"""Hybrid BM25 + corpus-trained dense retrieval with stable provenance."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np

from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.corpus.provenance import CanonicalChunk
from jarvis_localhost.linguistics import LinguisticExtractor, LinguisticProfile
from jarvis_localhost.retrieval.encoder import RetrieverEncoder
from jarvis_localhost.retrieval.vector_store import VectorStore


def lexical_tokens(text: str) -> list[str]:
    # No external stop-word list or domain lexicon: all terms originate in the
    # question/corpus and IDF is learned from the indexed chunks.
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


@dataclass(frozen=True)
class RetrieverResult:
    chunk: CanonicalChunk
    score: float
    lexical_score: float
    dense_score: float | None
    rank: int

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": self.score,
            "lexical_score": self.lexical_score,
            "dense_score": self.dense_score,
            "rank": self.rank,
        }


class SovereignRetriever:
    """Retriever whose learned statistics and weights come from the corpus."""

    def __init__(
        self,
        tokenizer: JarvisTokenizer,
        encoder: RetrieverEncoder | None = None,
        store: VectorStore | None = None,
        *,
        device: Any = "cpu",
        lexical_weight: float = 0.65,
        dense_weight: float = 0.35,
        dense_confidence_calibrator: Callable[[float], float] | None = None,
    ) -> None:
        if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
            raise ValueError("retriever weights must be non-negative and not both zero")
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.store = store or VectorStore()
        self.device = device
        total = lexical_weight + dense_weight
        self.lexical_weight = lexical_weight / total
        self.dense_weight = dense_weight / total
        # ``trained_on_corpus`` is a lineage claim, not a confidence
        # calibration. Dense-only evidence may cross the strict gate only when
        # the caller supplies an explicit calibrator learned/validated for that
        # retriever's score distribution.
        self.dense_confidence_calibrator = dense_confidence_calibrator
        self.chunks: dict[str, CanonicalChunk] = {}
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.document_frequencies: Counter[str] = Counter()
        self.term_postings: dict[str, list[tuple[str, int]]] = {}
        self.average_length = 0.0
        self._lock = threading.RLock()

    def index(self, chunks: Iterable[CanonicalChunk], *, batch_size: int = 32) -> int:
        incoming = list(chunks)
        incoming_by_id: dict[str, CanonicalChunk] = {}
        for chunk in incoming:
            if chunk.chunk_id in incoming_by_id:
                raise ValueError(f"duplicate chunk in corpus snapshot: {chunk.chunk_id}")
            incoming_by_id[chunk.chunk_id] = chunk
        term_frequencies, document_frequencies, average_length, term_postings = (
            self._build_lexical_index(incoming_by_id)
        )
        vectors: np.ndarray | None = None
        metadata: list[dict] = []
        staged_store: VectorStore | None = None

        if self.encoder is not None:
            extractor = LinguisticExtractor()
            profiles = {chunk.chunk_id: extractor.extract(chunk.text) for chunk in incoming}
            vectors = self.encoder.embed_texts(
                self.tokenizer,
                [chunk.text for chunk in incoming],
                device=self.device,
                batch_size=batch_size,
            ).numpy()
            
            style_vecs = []
            rhet_vecs = []
            for chunk in incoming:
                prof = profiles[chunk.chunk_id]
                d = chunk.to_dict()
                d["rhetorical_role"] = prof.rhetorical.primary_role.value
                d["linguistic"] = prof.to_dict()
                metadata.append(d)
                style_vecs.append(prof.style_vector)
                rhet_vecs.append(prof.rhetorical_vector)

            staged_store = VectorStore()
            if metadata:
                staged_store.add(
                    vectors,
                    metadata,
                    style_vectors=np.array(style_vecs, dtype=np.float32),
                    rhetorical_vectors=np.array(rhet_vecs, dtype=np.float32),
                )

        with self._lock:
            if not incoming_by_id:
                self.store.clear()
            elif self.encoder is None and len(self.store):
                store_ids = {
                    str(entry.get("chunk_id", "")) for entry in self.store.metadata
                }
                if store_ids != set(incoming_by_id):
                    raise ValueError(
                        "precomputed vector store does not match the complete corpus snapshot"
                    )
            elif self.encoder is not None:
                # Publish a complete dense generation while holding the same
                # lock used by search. Keeping the VectorStore object preserves
                # Brain/save call sites while stale entries are removed.
                self.store.clear()
                if staged_store is not None and staged_store.vectors is not None:
                    self.store.add(
                        staged_store.vectors,
                        staged_store.metadata,
                        style_vectors=staged_store.style_vectors,
                        rhetorical_vectors=staged_store.rhetorical_vectors,
                    )
            else:
                self.store.clear()
            self.chunks = incoming_by_id
            self.term_frequencies = term_frequencies
            self.document_frequencies = document_frequencies
            self.term_postings = term_postings
            self.average_length = average_length
        return len(incoming)

    @staticmethod
    def _build_lexical_index(
        chunks: dict[str, CanonicalChunk],
    ) -> tuple[dict[str, Counter[str]], Counter[str], float, dict[str, list[tuple[str, int]]]]:
        term_frequencies: dict[str, Counter[str]] = {}
        document_frequencies: Counter[str] = Counter()
        lengths: list[int] = []
        term_postings: dict[str, list[tuple[str, int]]] = {}
        for chunk_id, chunk in chunks.items():
            frequencies = Counter(lexical_tokens(chunk.text))
            term_frequencies[chunk_id] = frequencies
            document_frequencies.update(frequencies.keys())
            lengths.append(sum(frequencies.values()))
            for term, freq in frequencies.items():
                if term not in term_postings:
                    term_postings[term] = []
                term_postings[term].append((chunk_id, freq))
        average_length = sum(lengths) / len(lengths) if lengths else 0.0
        return term_frequencies, document_frequencies, average_length, term_postings

    @staticmethod
    def _query_terms(query: str) -> Counter[str]:
        tokens = lexical_tokens(query)
        content = [token for token in tokens if len(token) > 2]
        return Counter(content or tokens)

    def _idf(self, term: str) -> float:
        total_chunks = len(self.chunks)
        docs_with_term = self.document_frequencies.get(term, 0)
        return math.log(
            1.0
            + (total_chunks - docs_with_term + 0.5)
            / (docs_with_term + 0.5)
        )

    def _bm25(self, query: str) -> dict[str, float]:
        query_terms = self._query_terms(query)
        total_chunks = len(self.chunks)
        if not query_terms or not total_chunks:
            return {}
        k1, b = 1.5, 0.75
        avg_len = max(self.average_length, 1.0)
        scores: dict[str, float] = {}
        for term, query_frequency in query_terms.items():
            postings = self.term_postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            weight = idf * (k1 + 1.0) * min(query_frequency, 2)
            for chunk_id, frequency in postings:
                length = sum(self.term_frequencies[chunk_id].values())
                denominator = frequency + k1 * (1.0 - b + b * length / avg_len)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (weight * frequency) / denominator
        return scores

    def _lexical_confidence(
        self, query: str, chunk_id: str, bm25_score: float
    ) -> float:
        """Return an absolute confidence based on query evidence coverage."""

        if bm25_score <= 0.0:
            return 0.0
        query_terms = self._query_terms(query)
        frequencies = self.term_frequencies.get(chunk_id, Counter())
        total_weight = sum(
            self._idf(term) * min(frequency, 2)
            for term, frequency in query_terms.items()
        )
        if total_weight <= 0.0:
            return 0.0
        supported_weight = sum(
            self._idf(term) * min(frequency, 2)
            for term, frequency in query_terms.items()
            if frequencies.get(term, 0) > 0
        )
        coverage = supported_weight / total_weight
        saturation = 1.0 - math.exp(-bm25_score / total_weight)
        return min(1.0, max(0.0, 0.85 * coverage + 0.15 * saturation))

    def _dense_confidence(self, score: float | None) -> float | None:
        if score is None or self.dense_confidence_calibrator is None:
            return None
        confidence = float(self.dense_confidence_calibrator(score))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("dense confidence calibrator must return a finite [0,1] value")
        return confidence

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrieverResult]:
        with self._lock:
            if top_k <= 0 or not query.strip() or not self.chunks:
                return []
            lexical = self._bm25(query)
            lexical_rank = sorted(lexical, key=lambda key: (-lexical[key], key))
            dense_entries: list[dict] = []
            dense_available = (
                self.encoder is not None
                and self.encoder.trained_on_corpus
                and len(self.store)
            )
            if dense_available and self.encoder is not None:
                vector = self.encoder.embed_texts(
                    self.tokenizer, [query], device=self.device, batch_size=1
                ).numpy()[0]
                dense_entries = [
                    entry
                    for entry in self.store.search(
                        vector, top_k=max(top_k * 5, 20)
                    )
                    if str(entry.get("chunk_id", "")) in self.chunks
                ]
            dense_by_id = {
                str(entry["chunk_id"]): float(entry["score"])
                for entry in dense_entries
            }
            dense_rank = [str(entry["chunk_id"]) for entry in dense_entries]

            # Reciprocal-rank fusion avoids pretending BM25 and cosine scores
            # share a scale. Absolute lexical coverage independently controls
            # the strict-answer confidence gate.
            fused: defaultdict[str, float] = defaultdict(float)
            for rank, chunk_id in enumerate(lexical_rank):
                if lexical[chunk_id] > 0:
                    fused[chunk_id] += self.lexical_weight / (60.0 + rank + 1)
            if dense_available:
                for rank, chunk_id in enumerate(dense_rank):
                    fused[chunk_id] += self.dense_weight / (60.0 + rank + 1)
            if not fused:
                return []
            ranked = sorted(fused, key=lambda key: (-fused[key], key))[:top_k]
            output: list[RetrieverResult] = []
            for rank, chunk_id in enumerate(ranked, start=1):
                lexical_confidence = self._lexical_confidence(
                    query, chunk_id, lexical.get(chunk_id, 0.0)
                )
                dense_score = dense_by_id.get(chunk_id)
                dense_confidence = self._dense_confidence(dense_score)
                if dense_confidence is None:
                    confidence = (
                        lexical_confidence if self.lexical_weight > 0 else 0.0
                    )
                elif self.lexical_weight <= 0:
                    confidence = dense_confidence
                elif self.dense_weight <= 0:
                    confidence = lexical_confidence
                else:
                    confidence = (
                        self.lexical_weight * lexical_confidence
                        + self.dense_weight * dense_confidence
                    )
                output.append(
                    RetrieverResult(
                        chunk=self.chunks[chunk_id],
                        score=confidence,
                        lexical_score=lexical.get(chunk_id, 0.0),
                        dense_score=dense_score,
                        rank=rank,
                    )
                )
            return output

    search = retrieve


Retriever = SovereignRetriever


__all__ = ["Retriever", "RetrieverResult", "SovereignRetriever", "lexical_tokens"]
