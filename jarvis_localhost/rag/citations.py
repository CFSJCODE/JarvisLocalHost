"""Citation records derived directly from canonical retrieval results."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from jarvis_localhost.retrieval.retriever import RetrieverResult


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    document_id: str
    document_sha256: str
    filename: str
    page: int
    section: str
    chunk_id: str
    bbox: tuple[float, float, float, float]
    quote: str
    score: float

    @property
    def marker(self) -> str:
        return f"[{self.evidence_id}]"

    @property
    def label(self) -> str:
        return f"[{self.filename} — página {self.page} — {self.chunk_id}]"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["marker"] = self.marker
        payload["label"] = self.label
        return payload


def citations_from_results(
    results: list[RetrieverResult], *, quote_limit: int = 900
) -> list[Citation]:
    citations: list[Citation] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        quote = chunk.text.strip()
        if len(quote) > quote_limit:
            quote = quote[: quote_limit - 1].rstrip() + "…"
        citations.append(
            Citation(
                evidence_id=f"E{index}",
                document_id=chunk.document_id,
                document_sha256=chunk.document_sha256,
                filename=chunk.filename,
                page=chunk.page,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                bbox=chunk.bbox,
                quote=quote,
                score=result.score,
            )
        )
    return citations


__all__ = ["Citation", "citations_from_results"]
