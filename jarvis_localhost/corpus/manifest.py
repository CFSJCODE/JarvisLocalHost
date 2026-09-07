"""Closed-world, content-addressed corpus manifest and atomic persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping

from .provenance import (
    CanonicalChunk,
    canonical_document_id,
    corpus_sha256,
    sha256_file,
)


CORPUS_FORMAT_VERSION = 2
_EMPTY_CORPUS_SHA256 = corpus_sha256([])
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_FIELDS = {
    "document_id",
    "document_sha256",
    "filename",
    "byte_size",
    "pages",
    "words",
    "chunks",
    "language",
    "indexed_at",
    "chunks_file",
    "chunks_sha256",
    "chunks_bytes",
    "chunks_records",
    "corpus_file",
    "corpus_file_sha256",
    "corpus_bytes",
    "metadata_file",
    "metadata_sha256",
    "metadata_bytes",
}
_MANIFEST_FIELDS = {
    "format_version",
    "corpus_sha256",
    "generated_at",
    "documents",
    "aggregate",
}
_LOCAL_LOCKS: dict[str, threading.RLock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


class ConcurrentCorpusUpdateError(RuntimeError):
    """A manifest changed after the caller acquired its expected snapshot."""


def _verified_sha(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _verified_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _verified_artifact_name(value: Any, field_name: str, suffix: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{field_name} must be a single relative filename")
    if value in {".", ".."} or "\x00" in value or not value.endswith(suffix):
        raise ValueError(f"{field_name} must end with {suffix!r}")
    return value


def _verified_timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


def _safe_artifact(root: Path, name: str) -> Path:
    root_resolved = root.resolve()
    candidate = root / name
    if candidate.is_symlink() or candidate.resolve().parent != root_resolved:
        raise ValueError(f"corpus artifact escapes or aliases the corpus root: {name}")
    if not candidate.is_file():
        raise ValueError(f"authorized corpus artifact is missing: {name}")
    return candidate


@dataclass(frozen=True)
class DocumentManifest:
    document_id: str
    document_sha256: str
    filename: str
    byte_size: int
    pages: int
    words: int
    chunks: int
    language: str
    indexed_at: str
    chunks_file: str
    chunks_sha256: str
    chunks_bytes: int
    chunks_records: int
    corpus_file: str
    corpus_file_sha256: str
    corpus_bytes: int
    metadata_file: str
    metadata_sha256: str
    metadata_bytes: int

    @classmethod
    def from_dict_verified(cls, value: Mapping[str, Any]) -> "DocumentManifest":
        if not isinstance(value, Mapping) or set(value) != _DOCUMENT_FIELDS:
            fields = set(value) if isinstance(value, Mapping) else set()
            raise ValueError(
                "document manifest field mismatch; "
                f"missing={sorted(_DOCUMENT_FIELDS - fields)}, "
                f"extra={sorted(fields - _DOCUMENT_FIELDS)}"
            )
        digest = _verified_sha(value["document_sha256"], "document_sha256")
        expected_id = canonical_document_id(digest)
        if value["document_id"] != expected_id:
            raise ValueError("document manifest id does not match its source SHA-256")
        filename = value["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or "\x00" in filename
        ):
            raise ValueError("document manifest filename must be a basename")
        language = value["language"]
        if not isinstance(language, str) or not language or len(language) > 64:
            raise ValueError("document manifest language is invalid")
        manifest = cls(
            document_id=expected_id,
            document_sha256=digest,
            filename=filename,
            byte_size=_verified_int(value["byte_size"], "byte_size"),
            pages=_verified_int(value["pages"], "pages", minimum=1),
            words=_verified_int(value["words"], "words"),
            chunks=_verified_int(value["chunks"], "chunks"),
            language=language,
            indexed_at=_verified_timestamp(value["indexed_at"], "indexed_at"),
            chunks_file=_verified_artifact_name(
                value["chunks_file"], "chunks_file", "_chunks.jsonl"
            ),
            chunks_sha256=_verified_sha(value["chunks_sha256"], "chunks_sha256"),
            chunks_bytes=_verified_int(value["chunks_bytes"], "chunks_bytes"),
            chunks_records=_verified_int(
                value["chunks_records"], "chunks_records"
            ),
            corpus_file=_verified_artifact_name(
                value["corpus_file"], "corpus_file", "_corpus.txt"
            ),
            corpus_file_sha256=_verified_sha(
                value["corpus_file_sha256"], "corpus_file_sha256"
            ),
            corpus_bytes=_verified_int(value["corpus_bytes"], "corpus_bytes"),
            metadata_file=_verified_artifact_name(
                value["metadata_file"], "metadata_file", "_meta.json"
            ),
            metadata_sha256=_verified_sha(
                value["metadata_sha256"], "metadata_sha256"
            ),
            metadata_bytes=_verified_int(
                value["metadata_bytes"], "metadata_bytes"
            ),
        )
        if manifest.chunks != manifest.chunks_records:
            raise ValueError("chunks and chunks_records must agree")
        return manifest


@dataclass
class CorpusManifest:
    format_version: int = CORPUS_FORMAT_VERSION
    corpus_sha256: str = _EMPTY_CORPUS_SHA256
    generated_at: str = ""
    documents: Dict[str, DocumentManifest] = field(default_factory=dict)

    @classmethod
    def load(
        cls, path: str | Path, *, verify_artifacts: bool = True
    ) -> "CorpusManifest":
        source = Path(path)
        if not source.exists():
            return cls()
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != _MANIFEST_FIELDS:
            fields = set(data) if isinstance(data, dict) else set()
            raise ValueError(
                "corpus manifest field mismatch; "
                f"missing={sorted(_MANIFEST_FIELDS - fields)}, "
                f"extra={sorted(fields - _MANIFEST_FIELDS)}"
            )
        version = _verified_int(data["format_version"], "format_version")
        if version != CORPUS_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported corpus format {version}; expected {CORPUS_FORMAT_VERSION}. "
                "Re-ingest the authorized source PDFs to migrate safely."
            )
        raw_documents = data["documents"]
        if not isinstance(raw_documents, dict):
            raise ValueError("documents must be a JSON object")
        documents: dict[str, DocumentManifest] = {}
        for key, value in raw_documents.items():
            document = DocumentManifest.from_dict_verified(value)
            if key != document.document_id:
                raise ValueError("document manifest key does not match document_id")
            documents[key] = document
        manifest = cls(
            format_version=version,
            corpus_sha256=_verified_sha(data["corpus_sha256"], "corpus_sha256"),
            generated_at=_verified_timestamp(data["generated_at"], "generated_at"),
            documents=documents,
        )
        if data["aggregate"] != manifest._aggregate():
            raise ValueError("manifest aggregate does not match authorized documents")
        if verify_artifacts:
            manifest.validate_artifacts(source.parent)
        return manifest

    def upsert_document(
        self,
        document: DocumentManifest,
        all_chunks: Iterable[CanonicalChunk],
    ) -> None:
        verified = DocumentManifest.from_dict_verified(asdict(document))
        chunks = list(all_chunks)
        self.documents[verified.document_id] = verified
        self.corpus_sha256 = corpus_sha256(chunks)
        self.generated_at = datetime.now(timezone.utc).isoformat()

    def needs_reindex(self, document_sha256: str) -> bool:
        return all(
            item.document_sha256 != document_sha256
            for item in self.documents.values()
        )

    def remove_document(self, document_id: str, root: str | Path) -> DocumentManifest:
        """Remove a document and its on-disk artifacts from the corpus.

        (audit fix / maintenance) Added to hygienically retire "ghost"
        documents that were persisted with zero chunks before
        ``EmptyDocumentError`` existed (see ``processing/pdf_processor.py``),
        and more generally to support deliberate corpus curation. Deletes the
        document's three artifact files, drops its manifest entry, and
        recomputes ``corpus_sha256`` from the remaining authorized chunks so
        the manifest stays internally consistent and a subsequent ``save()``
        (which calls ``validate_artifacts``) still passes the closed-world
        check: every file on disk must be listed, and vice versa.

        Returns the removed ``DocumentManifest`` entry. Raises ``KeyError``
        if ``document_id`` is not present. Does not call ``save()``; the
        caller persists the mutated manifest (typically under
        ``corpus_update_lock()``), mirroring ``upsert_document``'s contract.
        """

        removed = self.documents.pop(document_id)
        corpus_root = Path(root)
        for name in (removed.chunks_file, removed.corpus_file, removed.metadata_file):
            _safe_artifact(corpus_root, name).unlink(missing_ok=True)
        remaining_chunks: list[CanonicalChunk] = []
        for document in self.documents.values():
            remaining_chunks.extend(
                read_chunks_jsonl(_safe_artifact(corpus_root, document.chunks_file))
            )
        self.corpus_sha256 = corpus_sha256(remaining_chunks)
        self.generated_at = datetime.now(timezone.utc).isoformat()
        return removed

    def _aggregate(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "bytes": sum(item.byte_size for item in self.documents.values()),
            "pages": sum(item.pages for item in self.documents.values()),
            "words": sum(item.words for item in self.documents.values()),
            "chunks": sum(item.chunks for item in self.documents.values()),
        }

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "corpus_sha256": self.corpus_sha256,
            "generated_at": self.generated_at,
            "documents": {
                key: asdict(value)
                for key, value in sorted(self.documents.items())
            },
            "aggregate": self._aggregate(),
        }

    def validate_artifacts(self, root: str | Path) -> list[CanonicalChunk]:
        corpus_root = Path(root)
        listed_chunks: set[str] = set()
        listed_corpora: set[str] = set()
        listed_metadata: set[str] = set()
        all_chunks: list[CanonicalChunk] = []
        for key, document in sorted(self.documents.items()):
            verified = DocumentManifest.from_dict_verified(asdict(document))
            if key != verified.document_id:
                raise ValueError("document map key does not match document identity")
            if verified.chunks_file in listed_chunks:
                raise ValueError("multiple documents authorize the same chunks file")
            if verified.corpus_file in listed_corpora:
                raise ValueError("multiple documents authorize the same corpus file")
            if verified.metadata_file in listed_metadata:
                raise ValueError("multiple documents authorize the same metadata file")
            listed_chunks.add(verified.chunks_file)
            listed_corpora.add(verified.corpus_file)
            listed_metadata.add(verified.metadata_file)

            chunks_path = _safe_artifact(corpus_root, verified.chunks_file)
            corpus_path = _safe_artifact(corpus_root, verified.corpus_file)
            metadata_path = _safe_artifact(corpus_root, verified.metadata_file)
            checks = (
                (chunks_path, verified.chunks_sha256, verified.chunks_bytes),
                (corpus_path, verified.corpus_file_sha256, verified.corpus_bytes),
                (metadata_path, verified.metadata_sha256, verified.metadata_bytes),
            )
            for artifact, expected_sha, expected_bytes in checks:
                if artifact.stat().st_size != expected_bytes:
                    raise ValueError(f"artifact byte count mismatch: {artifact.name}")
                if sha256_file(artifact) != expected_sha:
                    raise ValueError(f"artifact SHA-256 mismatch: {artifact.name}")

            document_chunks = read_chunks_jsonl(chunks_path)
            if len(document_chunks) != verified.chunks_records:
                raise ValueError(f"chunk record count mismatch: {chunks_path.name}")
            for chunk in document_chunks:
                if (
                    chunk.document_id != verified.document_id
                    or chunk.document_sha256 != verified.document_sha256
                    or chunk.filename != verified.filename
                ):
                    raise ValueError(
                        f"chunk provenance escapes document authority: {chunk.chunk_id}"
                    )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata artifact is not an object: {metadata_path.name}")
            bound_metadata = {
                "document_id": verified.document_id,
                "document_sha256": verified.document_sha256,
                "chunks_file": verified.chunks_file,
                "corpus_file": verified.corpus_file,
            }
            for field_name, expected in bound_metadata.items():
                if metadata.get(field_name) != expected:
                    raise ValueError(
                        f"metadata field {field_name!r} is not bound to its manifest"
                    )
            all_chunks.extend(document_chunks)

        actual_chunks = {path.name for path in corpus_root.glob("*_chunks.jsonl")}
        actual_corpora = {path.name for path in corpus_root.glob("*_corpus.txt")}
        actual_metadata = {path.name for path in corpus_root.glob("*_meta.json")}
        if actual_chunks != listed_chunks:
            raise ValueError(
                "closed-world chunks mismatch; "
                f"orphan={sorted(actual_chunks - listed_chunks)}, "
                f"missing={sorted(listed_chunks - actual_chunks)}"
            )
        if actual_corpora != listed_corpora:
            raise ValueError(
                "closed-world corpus artifacts mismatch; "
                f"orphan={sorted(actual_corpora - listed_corpora)}, "
                f"missing={sorted(listed_corpora - actual_corpora)}"
            )
        if actual_metadata != listed_metadata:
            raise ValueError(
                "closed-world metadata artifacts mismatch; "
                f"orphan={sorted(actual_metadata - listed_metadata)}, "
                f"missing={sorted(listed_metadata - actual_metadata)}"
            )
        expected_corpus_sha = corpus_sha256(all_chunks)
        if self.corpus_sha256 != expected_corpus_sha:
            raise ValueError("global corpus SHA-256 does not match authorized chunks")
        return all_chunks

    def save(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.format_version != CORPUS_FORMAT_VERSION:
            raise ValueError("cannot save an unsupported corpus format")
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        _verified_timestamp(self.generated_at, "generated_at")
        _verified_sha(self.corpus_sha256, "corpus_sha256")
        self.validate_artifacts(destination.parent)
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            json.loads(temporary.read_text(encoding="utf-8"))
            if expected_sha256 is not None:
                actual = sha256_file(destination) if destination.exists() else ""
                if actual != expected_sha256:
                    raise ConcurrentCorpusUpdateError(
                        "corpus manifest changed before atomic publication"
                    )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def load_authorized_chunks(root: str | Path) -> list[CanonicalChunk]:
    """Load exactly the closed-world corpus authorized by its manifest.

    An absent manifest is valid only for a genuinely empty corpus directory.
    ``validate_artifacts`` rejects orphan canonical artifacts, altered files,
    missing records and aggregate-lineage mismatches before returning data.
    """

    corpus_root = Path(root)
    with corpus_update_lock(corpus_root / ".corpus_manifest.lock"):
        manifest = CorpusManifest.load(
            corpus_root / "corpus_manifest.json",
            verify_artifacts=False,
        )
        return manifest.validate_artifacts(corpus_root)


def write_chunks_jsonl(path: str | Path, chunks: Iterable[CanonicalChunk]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    verified_chunks: list[CanonicalChunk] = []
    seen_ids: set[str] = set()
    seen_ordinals: set[tuple[str, int]] = set()
    for chunk in chunks:
        verified = CanonicalChunk.from_dict_verified(chunk.to_dict())
        if verified.chunk_id in seen_ids:
            raise ValueError(f"duplicate canonical chunk id: {verified.chunk_id}")
        ordinal_key = (verified.document_id, verified.ordinal)
        if ordinal_key in seen_ordinals:
            raise ValueError(
                f"duplicate canonical chunk ordinal: {verified.document_id}/{verified.ordinal}"
            )
        seen_ids.add(verified.chunk_id)
        seen_ordinals.add(ordinal_key)
        verified_chunks.append(verified)
    rows = [
        json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True)
        for chunk in verified_chunks
    ]
    payload = "\n".join(rows) + ("\n" if rows else "")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        read_chunks_jsonl(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_chunks_jsonl(path: str | Path) -> List[CanonicalChunk]:
    source = Path(path)
    if not source.exists():
        return []
    chunks: list[CanonicalChunk] = []
    seen_ids: set[str] = set()
    seen_ordinals: set[tuple[str, int]] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            chunk = CanonicalChunk.from_dict_verified(payload)
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk id {chunk.chunk_id}")
            ordinal_key = (chunk.document_id, chunk.ordinal)
            if ordinal_key in seen_ordinals:
                raise ValueError(
                    f"duplicate ordinal {chunk.document_id}/{chunk.ordinal}"
                )
            seen_ids.add(chunk.chunk_id)
            seen_ordinals.add(ordinal_key)
            chunks.append(chunk)
        except Exception as exc:
            raise ValueError(f"Invalid chunk JSONL line {line_number}: {exc}") from exc
    return chunks


def _local_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve()).casefold()
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def corpus_update_lock(
    path: str | Path, *, timeout_seconds: float = 60.0
) -> Iterator[None]:
    """Serialize manifest updates across threads and operating-system processes."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    local_lock = _local_lock_for(lock_path)
    if not local_lock.acquire(timeout=timeout_seconds):
        raise TimeoutError(f"timed out acquiring corpus lock: {lock_path}")
    handle = None
    acquired = False
    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring corpus lock: {lock_path}")
                time.sleep(0.05)
        yield
    finally:
        if handle is not None:
            if acquired:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()
        local_lock.release()
