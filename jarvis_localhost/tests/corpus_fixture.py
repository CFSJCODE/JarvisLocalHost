"""Closed-world corpus fixtures shared by integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis_localhost.corpus.manifest import (
    CorpusManifest,
    DocumentManifest,
    write_chunks_jsonl,
)
from jarvis_localhost.corpus.provenance import CanonicalChunk, sha256_file


def write_closed_corpus(root: Path, chunks: list[CanonicalChunk]) -> CorpusManifest:
    """Replace a temporary test corpus with a manifest-authorized snapshot."""

    root.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_chunks.jsonl", "*_corpus.txt", "*_meta.json"):
        for path in root.glob(pattern):
            path.unlink()
    (root / "corpus_manifest.json").unlink(missing_ok=True)

    by_document: dict[str, list[CanonicalChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    manifest = CorpusManifest()
    for document_id, document_chunks in sorted(by_document.items()):
        ordered = sorted(
            document_chunks,
            key=lambda item: (item.ordinal, item.page, item.chunk_id),
        )
        first = ordered[0]
        chunks_path = root / f"{document_id}_chunks.jsonl"
        corpus_path = root / f"{document_id}_corpus.txt"
        metadata_path = root / f"{document_id}_meta.json"
        write_chunks_jsonl(chunks_path, ordered)
        corpus_path.write_text(
            "\n".join(chunk.text for chunk in ordered) + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "document_sha256": first.document_sha256,
                    "chunks_file": chunks_path.name,
                    "corpus_file": corpus_path.name,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record = DocumentManifest(
            document_id=document_id,
            document_sha256=first.document_sha256,
            filename=first.filename,
            byte_size=0,
            pages=max(chunk.page for chunk in ordered),
            words=sum(len(chunk.text.split()) for chunk in ordered),
            chunks=len(ordered),
            language="pt-BR",
            indexed_at="2026-08-26T00:00:00+00:00",
            chunks_file=chunks_path.name,
            chunks_sha256=sha256_file(chunks_path),
            chunks_bytes=chunks_path.stat().st_size,
            chunks_records=len(ordered),
            corpus_file=corpus_path.name,
            corpus_file_sha256=sha256_file(corpus_path),
            corpus_bytes=corpus_path.stat().st_size,
            metadata_file=metadata_path.name,
            metadata_sha256=sha256_file(metadata_path),
            metadata_bytes=metadata_path.stat().st_size,
        )
        manifest.upsert_document(record, chunks)

    manifest.save(root / "corpus_manifest.json")
    return manifest


__all__ = ["write_closed_corpus"]
