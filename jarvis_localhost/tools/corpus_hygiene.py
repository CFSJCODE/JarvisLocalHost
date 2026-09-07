"""Detect and, optionally, remove zero-chunk "ghost" documents.

(audit fix / maintenance tool) Before ``EmptyDocumentError`` existed in
``processing/pdf_processor.py``, a PDF ingested while sovereign mode was off
and that produced no extractable text, tables or OCR output (for example a
100%-rasterized-image PDF saved from a browser print dialog) was still
recorded as a valid, permanent ``DocumentManifest`` entry with zero chunks
and zero words. Such a document inflates the corpus document/page counts,
contributes nothing to retrieval, and raises no error anywhere -- it can
only be found by reading every document's chunk count.

This script is read-only by default: it lists any zero-chunk documents it
finds in ``corpus_manifest.json`` without changing anything. Pass ``--fix``
to actually remove them (source PDFs under ``uploads/`` are never touched,
only the corpus artifacts: ``*_chunks.jsonl`` / ``*_corpus.txt`` /
``*_meta.json`` and the manifest entry), so the original files remain
available for re-ingestion later (e.g. with OCR enabled).

Usage:
    python -m jarvis_localhost.tools.corpus_hygiene              # report only
    python -m jarvis_localhost.tools.corpus_hygiene --fix         # remove them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jarvis_localhost.corpus.manifest import CorpusManifest, corpus_update_lock
from jarvis_localhost.paths import CORPUS_ROOT


def find_ghost_documents(manifest: CorpusManifest) -> list[dict[str, object]]:
    # (audit fix) A document can also be a "ghost" with chunks > 0: e.g. a
    # vector-graphic PDF where pdfplumber misreads decorative lines as an
    # empty table grid, producing one or more canonical chunks whose text is
    # just empty table syntax ("| | | --- |") -- zero real words, same as the
    # chunks == 0 case, just a different extraction failure mode. Checking
    # words == 0 in addition to chunks == 0 catches both without depending on
    # any document-specific heuristic.
    return [
        {
            "document_id": document.document_id,
            "filename": document.filename,
            "pages": document.pages,
            "chunks": document.chunks,
            "words": document.words,
            "indexed_at": document.indexed_at,
        }
        for document in manifest.documents.values()
        if document.chunks == 0 or document.words == 0
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_ROOT,
        help="Directory containing corpus_manifest.json (default: the "
        "package's own data/embeddings directory).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove the zero-chunk documents found (default: report only).",
    )
    args = parser.parse_args(argv)

    manifest_path = args.corpus_dir / "corpus_manifest.json"
    if not manifest_path.exists():
        print(json.dumps({"status": "no_manifest", "path": str(manifest_path)}))
        return 0

    manifest = CorpusManifest.load(manifest_path, verify_artifacts=not args.fix)
    ghosts = find_ghost_documents(manifest)
    if not ghosts:
        print(json.dumps({"status": "clean", "documents_checked": len(manifest.documents)}))
        return 0

    if not args.fix:
        print(
            json.dumps(
                {
                    "status": "ghosts_found",
                    "count": len(ghosts),
                    "documents": ghosts,
                    "hint": "Re-run with --fix to remove these manifest "
                    "entries and their empty corpus artifacts. Source PDFs "
                    "under uploads/ are left untouched for re-ingestion.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    lock_path = args.corpus_dir / ".corpus_manifest.lock"
    removed: list[dict[str, object]] = []
    with corpus_update_lock(lock_path):
        # Re-load inside the lock: another process may have changed the
        # manifest between the report above and taking the lock.
        manifest = CorpusManifest.load(manifest_path, verify_artifacts=True)
        for ghost in find_ghost_documents(manifest):
            manifest.remove_document(str(ghost["document_id"]), args.corpus_dir)
            removed.append(ghost)
        if removed:
            manifest.save(manifest_path)
    print(
        json.dumps(
            {"status": "fixed", "removed": removed}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
