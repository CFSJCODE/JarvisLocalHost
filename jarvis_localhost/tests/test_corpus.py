from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from jarvis_localhost.corpus.chunker import PageSpan, canonicalize_spans
from jarvis_localhost.corpus.manifest import (
    CorpusManifest,
    DocumentManifest,
    corpus_update_lock,
    read_chunks_jsonl,
    write_chunks_jsonl,
)
from jarvis_localhost.corpus.provenance import (
    CanonicalChunk,
    DocumentIdentity,
    canonical_document_id,
    sha256_file,
    stable_sha256,
)
from jarvis_localhost.sovereign import SovereignModeViolation, SovereignPolicy


def identity(fill: str, filename: str = "manual.pdf", size: int = 42) -> DocumentIdentity:
    digest = fill * 64
    return DocumentIdentity(
        canonical_document_id(digest), digest, filename, size
    )


def create_closed_manifest(
    root: Path,
    document: DocumentIdentity,
    chunks: list[CanonicalChunk],
) -> tuple[CorpusManifest, Path]:
    chunks_path = root / f"{document.document_id}_chunks.jsonl"
    corpus_path = root / f"{document.document_id}_corpus.txt"
    metadata_path = root / f"{document.document_id}_meta.json"
    manifest_path = root / "corpus_manifest.json"
    write_chunks_jsonl(chunks_path, chunks)
    corpus_path.write_text(
        "\n".join(chunk.text for chunk in chunks) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "document_id": document.document_id,
                "document_sha256": document.document_sha256,
                "chunks_file": chunks_path.name,
                "corpus_file": corpus_path.name,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    document_manifest = DocumentManifest(
        document_id=document.document_id,
        document_sha256=document.document_sha256,
        filename=document.filename,
        byte_size=document.byte_size,
        pages=max(chunk.page for chunk in chunks),
        words=sum(len(chunk.text.split()) for chunk in chunks),
        chunks=len(chunks),
        language="pt-BR",
        indexed_at="2026-08-26T00:00:00+00:00",
        chunks_file=chunks_path.name,
        chunks_sha256=sha256_file(chunks_path),
        chunks_bytes=chunks_path.stat().st_size,
        chunks_records=len(chunks),
        corpus_file=corpus_path.name,
        corpus_file_sha256=sha256_file(corpus_path),
        corpus_bytes=corpus_path.stat().st_size,
        metadata_file=metadata_path.name,
        metadata_sha256=sha256_file(metadata_path),
        metadata_bytes=metadata_path.stat().st_size,
    )
    manifest = CorpusManifest()
    manifest.upsert_document(document_manifest, chunks)
    manifest.save(manifest_path)
    return manifest, manifest_path


class CorpusTests(unittest.TestCase):
    def test_chunk_ids_are_stable_and_sections_use_only_contributing_bboxes(self) -> None:
        document = identity("a")
        spans = [
            PageSpan(1, "Secao Alpha", (0, 0, 100, 10), True),
            PageSpan(1, " ".join(f"alpha{i}" for i in range(18)), (0, 20, 100, 30)),
            PageSpan(1, "Secao Beta", (0, 100, 100, 110), True),
            PageSpan(1, " ".join(f"beta{i}" for i in range(18)), (0, 120, 100, 130)),
            PageSpan(2, "rotulo", (10, 10, 20, 20)),
        ]
        first = canonicalize_spans(
            document, spans, target_words=16, overlap_words=4
        )
        second = canonicalize_spans(
            document, spans, target_words=16, overlap_words=4
        )
        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])
        self.assertEqual({item.page for item in first}, {1, 2})
        beta = [
            item
            for item in first
            if item.section == "Secao Beta" and item.page == 1
        ]
        self.assertTrue(beta)
        self.assertTrue(all(item.bbox[1] >= 100 for item in beta))
        self.assertEqual(first[-1].text, "rotulo")
        self.assertEqual(first[-1].section, "Secao Beta")
        self.assertEqual(first[-1].bbox, (10.0, 10.0, 20.0, 20.0))
        self.assertTrue(first[0].citation().startswith("[manual.pdf"))

    def test_jsonl_round_trip_rejects_forged_content_and_identity(self) -> None:
        document = identity("b", "book.pdf", 99)
        chunks = canonicalize_spans(
            document,
            [PageSpan(1, "conteudo curto", (0, 0, 1, 1))],
            target_words=32,
            overlap_words=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "book_chunks.jsonl"
            write_chunks_jsonl(chunks_path, chunks)
            self.assertEqual(read_chunks_jsonl(chunks_path), chunks)
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            payload["bbox"] = [0, 0, 9, 9]
            chunks_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chunk_id"):
                read_chunks_jsonl(chunks_path)
            payload = chunks[0].to_dict()
            payload["document_id"] = "doc_000000000000000000000000"
            chunks_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "document_id"):
                read_chunks_jsonl(chunks_path)

    def test_manifest_binds_artifacts_and_rejects_orphan_glob_injection(self) -> None:
        document = identity("c", "closed.pdf", 123)
        chunks = canonicalize_spans(
            document,
            [PageSpan(1, "formula E igual a m c dois", (1, 2, 3, 4))],
            target_words=32,
            overlap_words=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, manifest_path = create_closed_manifest(root, document, chunks)
            loaded = CorpusManifest.load(manifest_path)
            self.assertEqual(loaded.corpus_sha256, manifest.corpus_sha256)
            self.assertFalse(loaded.needs_reindex(document.document_sha256))
            document_record = loaded.documents[document.document_id]
            self.assertEqual(document_record.chunks_records, len(chunks))
            self.assertEqual(document_record.chunks_sha256, sha256_file(root / document_record.chunks_file))

            (root / "injected_chunks.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed-world chunks mismatch"):
                CorpusManifest.load(manifest_path)

    def test_manifest_rejects_artifact_tampering_and_legacy_format(self) -> None:
        document = identity("d", "bound.pdf", 12)
        chunk = CanonicalChunk.build(
            document,
            page=1,
            section="",
            bbox=(0, 0, 1, 1),
            text="x",
            ordinal=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest_path = create_closed_manifest(root, document, [chunk])
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks_path = root / record["documents"][document.document_id]["chunks_file"]
            chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte count mismatch"):
                CorpusManifest.load(manifest_path)

            record["format_version"] = 1
            manifest_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Re-ingest"):
                CorpusManifest.load(manifest_path)

    def test_corpus_lock_serializes_independent_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / ".corpus.lock"
            code = (
                "import sys,time; "
                "from jarvis_localhost.corpus.manifest import corpus_update_lock; "
                "\nwith corpus_update_lock(sys.argv[1]):"
                "\n print('locked', flush=True); time.sleep(0.35)"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", code, str(lock_path)],
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                started = time.monotonic()
                with corpus_update_lock(lock_path, timeout_seconds=2):
                    pass
                self.assertGreaterEqual(time.monotonic() - started, 0.20)
                _, stderr = child.communicate(timeout=2)
                self.assertEqual(child.returncode, 0, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=2)

    def test_sovereign_policy_blocks_ocr_and_egress(self) -> None:
        policy = SovereignPolicy(enabled=True)
        with self.assertRaises(SovereignModeViolation):
            policy.require_text_layer(3, False)
        with self.assertRaises(SovereignModeViolation):
            policy.assert_url_allowed("https://example.com/api")
        policy.assert_url_allowed("http://127.0.0.1:8000")

    def test_stable_sha_is_not_python_hash(self) -> None:
        self.assertEqual(stable_sha256("abc"), stable_sha256(b"abc"))
        self.assertEqual(len(stable_sha256("abc")), 64)


if __name__ == "__main__":
    unittest.main()
