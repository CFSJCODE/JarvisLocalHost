"""Stable, self-verifying provenance records for retrievable corpus units."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_CHUNK_FIELDS = {
    "document_id",
    "document_sha256",
    "filename",
    "page",
    "section",
    "chunk_id",
    "bbox",
    "text",
    "ordinal",
    "token_count",
    "extraction_method",
    "content_type",
    "source_ref",
}


def stable_sha256(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verified_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def canonical_document_id(document_sha256: str) -> str:
    digest = _verified_sha256(document_sha256, "document_sha256")
    return f"doc_{digest[:24]}"


def _normalized_bbox(bbox: Sequence[float] | None) -> Tuple[float, float, float, float]:
    if bbox is None:
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly four numeric coordinates")
    values: list[float] = []
    for coordinate in bbox:
        if isinstance(coordinate, bool):
            raise ValueError("bbox coordinates cannot be booleans")
        try:
            numeric = float(coordinate)
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox coordinates must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError("bbox coordinates must be finite")
        values.append(round(numeric, 3))
    x0, y0, x1, y1 = values
    if x0 > x1 or y0 > y1:
        raise ValueError("bbox coordinates must satisfy x0 <= x1 and y0 <= y1")
    return (x0, y0, x1, y1)


def _verified_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("filename must be a non-empty string")
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("filename must be a basename, not a path")
    return value


def _verified_label(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a lowercase provenance label using letters, digits, ._:-"
        )
    return value


def _chunk_payload(
    *,
    document_sha256: str,
    page: int,
    section: str,
    bbox: Tuple[float, float, float, float],
    text: str,
    ordinal: int,
    token_count: Optional[int],
    extraction_method: str,
    content_type: str,
    source_ref: str,
) -> str:
    return json.dumps(
        {
            "bbox": list(bbox),
            "content_type": content_type,
            "document_sha256": document_sha256,
            "extraction_method": extraction_method,
            "ordinal": ordinal,
            "page": page,
            "section": section,
            "source_ref": source_ref,
            "text": text,
            "token_count": token_count,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_chunk_id(**payload: Any) -> str:
    return f"chk_{stable_sha256(_chunk_payload(**payload))[:32]}"


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    document_sha256: str
    filename: str
    byte_size: int

    @classmethod
    def from_path(cls, path: str | Path) -> "DocumentIdentity":
        source = Path(path)
        digest = sha256_file(source)
        return cls(
            document_id=canonical_document_id(digest),
            document_sha256=digest,
            filename=source.name,
            byte_size=source.stat().st_size,
        )


@dataclass(frozen=True)
class CanonicalChunk:
    document_id: str
    document_sha256: str
    filename: str
    page: int
    section: str
    chunk_id: str
    bbox: Tuple[float, float, float, float]
    text: str
    ordinal: int
    token_count: Optional[int] = None
    extraction_method: str = "text_layer"
    content_type: str = "text"
    source_ref: str = ""

    @classmethod
    def build(
        cls,
        document: DocumentIdentity,
        *,
        page: int,
        section: str,
        bbox: Sequence[float] | None,
        text: str,
        ordinal: int,
        token_count: Optional[int] = None,
        extraction_method: str = "text_layer",
        content_type: str = "text",
        source_ref: str = "",
    ) -> "CanonicalChunk":
        document_sha256 = _verified_sha256(
            document.document_sha256, "document_sha256"
        )
        filename = _verified_filename(document.filename)
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("PDF page numbers are one-based positive integers")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if token_count is not None and (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("token_count must be null or a non-negative integer")
        if not isinstance(section, str):
            raise ValueError("section must be a string")
        if not isinstance(source_ref, str):
            raise ValueError("source_ref must be a string")
        if len(source_ref) > 512 or "\x00" in source_ref:
            raise ValueError("source_ref is invalid or too long")
        cleaned = " ".join(str(text).split()) if isinstance(text, str) else ""
        if not cleaned:
            raise ValueError("canonical chunk text cannot be empty")
        normalized_section = " ".join(section.split())
        normalized_bbox = _normalized_bbox(bbox)
        normalized_extraction = _verified_label(
            extraction_method, "extraction_method"
        )
        normalized_content_type = _verified_label(content_type, "content_type")
        normalized_source_ref = source_ref.strip()
        identity = canonical_document_id(document_sha256)
        id_payload = {
            "document_sha256": document_sha256,
            "page": page,
            "section": normalized_section,
            "bbox": normalized_bbox,
            "text": cleaned,
            "ordinal": ordinal,
            "token_count": token_count,
            "extraction_method": normalized_extraction,
            "content_type": normalized_content_type,
            "source_ref": normalized_source_ref,
        }
        return cls(
            document_id=identity,
            document_sha256=document_sha256,
            filename=filename,
            page=page,
            section=normalized_section,
            chunk_id=_canonical_chunk_id(**id_payload),
            bbox=normalized_bbox,
            text=cleaned,
            ordinal=ordinal,
            token_count=token_count,
            extraction_method=normalized_extraction,
            content_type=normalized_content_type,
            source_ref=normalized_source_ref,
        )

    @classmethod
    def from_dict_verified(cls, value: Mapping[str, Any]) -> "CanonicalChunk":
        """Deserialize only records whose entire identity can be recomputed.

        Older JSONL records did not bind bbox/extraction metadata into
        ``chunk_id`` and cannot be migrated without re-ingesting the authorized
        source PDF, so they intentionally fail closed here.
        """

        if not isinstance(value, Mapping):
            raise ValueError("canonical chunk must be a JSON object")
        fields = set(value)
        if fields != _CHUNK_FIELDS:
            missing = sorted(_CHUNK_FIELDS - fields)
            extra = sorted(fields - _CHUNK_FIELDS)
            raise ValueError(
                f"canonical chunk field mismatch; missing={missing}, extra={extra}"
            )
        digest = _verified_sha256(value["document_sha256"], "document_sha256")
        expected_document_id = canonical_document_id(digest)
        if value["document_id"] != expected_document_id:
            raise ValueError("document_id does not match document_sha256")
        chunk = cls.build(
            DocumentIdentity(
                document_id=expected_document_id,
                document_sha256=digest,
                filename=_verified_filename(value["filename"]),
                byte_size=0,
            ),
            page=value["page"],
            section=value["section"],
            bbox=value["bbox"],
            text=value["text"],
            ordinal=value["ordinal"],
            token_count=value["token_count"],
            extraction_method=value["extraction_method"],
            content_type=value["content_type"],
            source_ref=value["source_ref"],
        )
        if value["chunk_id"] != chunk.chunk_id:
            raise ValueError("chunk_id does not match canonical chunk content")
        if value["text"] != chunk.text:
            raise ValueError("canonical chunk text is not normalized")
        if value["section"] != chunk.section:
            raise ValueError("canonical chunk section is not normalized")
        if value["source_ref"] != chunk.source_ref:
            raise ValueError("canonical chunk source_ref is not normalized")
        if tuple(value["bbox"]) != chunk.bbox:
            raise ValueError("canonical chunk bbox is not normalized")
        return chunk

    def citation(self) -> str:
        return f"[{self.filename} — pagina {self.page} — {self.chunk_id}]"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


def corpus_sha256(chunks: Iterable[CanonicalChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(
        chunks, key=lambda item: (item.document_id, item.ordinal, item.chunk_id)
    ):
        verified = CanonicalChunk.from_dict_verified(chunk.to_dict())
        digest.update(
            json.dumps(
                verified.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
