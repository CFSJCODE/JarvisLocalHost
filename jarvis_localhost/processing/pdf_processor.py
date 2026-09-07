"""
pdf_processor.py — J.A.R.V.I.S Multimodal Document Processor
Extracts text, images, tables and metadata from PDF documents.
"""

import io
import os
import re
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import fitz          # PyMuPDF
import pdfplumber
from PIL import Image

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
    sha256_file,
)
from jarvis_localhost.logging_config import get_logger
from jarvis_localhost.paths import CORPUS_ROOT, IMAGES_ROOT
from jarvis_localhost.sovereign import POLICY, SovereignPolicy

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = get_logger(__name__)


class EmptyDocumentError(RuntimeError):
    """Raised when a PDF yields zero canonical chunks.

    (audit fix) Before this check existed, a PDF with no extractable text
    layer, no tables and no OCR output (e.g. a 100%-rasterized-image PDF
    ingested while sovereign mode was off, so ``SovereignPolicy.require_text_
    layer`` never fired) still passed through ``_save_corpus`` and was
    recorded as a permanently valid ``DocumentManifest`` entry with zero
    chunks/zero words: a "ghost document" that inflates the document/page
    count, is unrecoverable, and raises no error anywhere. This is the
    independent, sovereign-mode-agnostic invariant: never persist a document
    that would contribute nothing to retrieval.
    """


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    page: int
    text: str
    bbox: tuple        # (x0, y0, x1, y1)
    font_size: float
    is_heading: bool


@dataclass
class TableData:
    page: int
    headers: List[str]
    rows: List[List[str]]
    markdown: str       # table as markdown for training
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    extraction_method: str = "pdfplumber_table"


@dataclass
class ImageData:
    page: int
    index: int
    width: int
    height: int
    ocr_text: str       # text extracted via OCR
    path: str           # saved image path
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    extraction_method: str = "pytesseract_pretrained_non_sovereign"


@dataclass
class DocumentResult:
    filename: str
    total_pages: int
    metadata: Dict[str, Any]
    full_text: str
    chunks: List[TextChunk]
    tables: List[TableData]
    images: List[ImageData]
    word_count: int
    language_hint: str
    training_corpus: str   # clean text ready for model training
    stats: Dict[str, int] = field(default_factory=dict)
    document_id: str = ""
    document_sha256: str = ""
    canonical_chunks: List[CanonicalChunk] = field(default_factory=list)
    corpus_path: str = ""
    chunks_path: str = ""


# ─── Processor ────────────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Multimodal PDF extractor for J.A.R.V.I.S.
    Handles text, tables, images (with OCR) from PDF documents.
    """

    MIN_CHUNK_WORDS   = 15
    HEADING_MIN_SIZE  = 13.0
    OCR_MIN_DIM       = 50   # pixels — skip tiny images

    def __init__(
        self,
        output_dir: str | Path | None = None,
        image_dir: str | Path | None = None,
        policy: SovereignPolicy | None = None,
        target_words: int = 220,
        overlap_words: int = 40,
        minimum_words: int = 12,
    ):
        self.output_dir = Path(output_dir) if output_dir else CORPUS_ROOT
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = Path(image_dir) if image_dir else IMAGES_ROOT
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy or POLICY
        self.policy.validate()
        self.target_words = target_words
        self.overlap_words = overlap_words
        # (audit fix) previously hardcoded to 1 at the canonicalize_spans call
        # below, which silently disabled the size floor for every document
        # ever ingested: any fragment as short as a single stray word (a page
        # number, an isolated table cell) became its own permanent chunk.
        # target_words/overlap_words were already configurable here; this
        # brings minimum_words in line with them, defaulting to the same
        # floor canonicalize_spans itself defaults to.
        self.minimum_words = minimum_words

    # ─── Public API ───────────────────────────────────────────────────────────

    def process(self, pdf_path: str) -> DocumentResult:
        """Full pipeline: extract everything from a PDF."""
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        logger.info("[PDFProcessor] Processing: %s", path.name)

        identity    = DocumentIdentity.from_path(path)
        metadata    = self._extract_metadata(pdf_path)
        pages_with_text = self._pages_with_text_layer(pdf_path)
        chunks      = self._extract_text_chunks(pdf_path)
        for page_number in range(1, int(metadata.get("pages", 0)) + 1):
            self.policy.require_text_layer(page_number, page_number in pages_with_text)
        tables      = self._extract_tables(pdf_path)
        images      = []
        if self.policy.allow_pretrained_ocr:
            images = self._extract_images(pdf_path, identity.document_id)
        canonical_spans = [
            PageSpan(
                page=chunk.page,
                text=chunk.text,
                bbox=chunk.bbox,
                is_heading=chunk.is_heading,
                extraction_method="pymupdf_text_layer",
                content_type="text",
            )
            for chunk in chunks
        ]
        canonical_spans.extend(
            PageSpan(
                page=table.page,
                text=table.markdown,
                bbox=table.bbox,
                extraction_method=table.extraction_method,
                content_type="table",
                source_ref=f"table:{index}",
            )
            for index, table in enumerate(tables)
            if table.markdown.strip()
        )
        canonical_spans.extend(
            PageSpan(
                page=image.page,
                text=image.ocr_text,
                bbox=image.bbox,
                extraction_method=image.extraction_method,
                content_type="ocr",
                source_ref=f"image:{image.index}",
            )
            for image in images
            if image.ocr_text.strip()
        )
        # Interleave extraction modalities by their source geometry.  This
        # prevents a table near the top of a page from inheriting a heading
        # that actually appears below it merely because table extraction ran
        # after text extraction.
        canonical_spans.sort(
            key=lambda span: (
                span.page,
                float(span.bbox[1]) if len(span.bbox) == 4 else float("inf"),
                float(span.bbox[0]) if len(span.bbox) == 4 else float("inf"),
                {"text": 0, "table": 1, "ocr": 2}.get(span.content_type, 3),
            )
        )
        canonical_chunks = canonicalize_spans(
            identity,
            canonical_spans,
            target_words=self.target_words,
            overlap_words=self.overlap_words,
            minimum_words=self.minimum_words,
        )
        if not canonical_chunks:
            raise EmptyDocumentError(
                f"'{path.name}' produced zero extractable chunks (no text "
                "layer, no tables, no OCR output). Nothing was indexed; the "
                "document was not saved. If this is a scanned/image-only "
                "PDF, enable OCR (JARVIS_ALLOW_PRETRAINED_OCR=1) or provide "
                "a text-layer version of the file."
            )
        full_text   = self._build_full_text(chunks)
        corpus      = self._build_training_corpus(canonical_chunks, tables, images)
        word_count  = len(full_text.split())

        result = DocumentResult(
            filename      = path.name,
            total_pages   = metadata.get("pages", 0),
            metadata      = metadata,
            full_text     = full_text,
            chunks        = chunks,
            tables        = tables,
            images        = images,
            word_count    = word_count,
            language_hint = self._detect_language(full_text[:500]),
            training_corpus = corpus,
            stats = {
                "chunks": len(chunks),
                "tables": len(tables),
                "images": len(images),
                "words":  word_count,
                "pages":  metadata.get("pages", 0),
                "canonical_chunks": len(canonical_chunks),
                "ocr_enabled": int(self.policy.allow_pretrained_ocr),
            }
            ,
            document_id=identity.document_id,
            document_sha256=identity.document_sha256,
            canonical_chunks=canonical_chunks,
        )

        revalidated = DocumentIdentity.from_path(path)
        if revalidated != identity:
            raise RuntimeError("source PDF changed while it was being ingested")
        self._save_corpus(result, identity, path)
        logger.info(
            "[PDFProcessor] Done - %d words, %d chunks, %d tables, %d images",
            word_count, len(chunks), len(tables), len(images),
        )
        return result

    def process_many(self, pdf_paths: List[str]) -> List[DocumentResult]:
        return [self.process(p) for p in pdf_paths]

    # ─── Metadata ─────────────────────────────────────────────────────────────

    def _extract_metadata(self, path: str) -> Dict[str, Any]:
        doc = fitz.open(path)
        meta = doc.metadata or {}
        info = {
            "pages":    len(doc),
            "title":    meta.get("title", ""),
            "author":   meta.get("author", ""),
            "subject":  meta.get("subject", ""),
            "creator":  meta.get("creator", ""),
            "keywords": meta.get("keywords", ""),
        }
        doc.close()
        return info

    # ─── Text Chunks ──────────────────────────────────────────────────────────

    def _pages_with_text_layer(self, path: str) -> set[int]:
        """Detect a PDF text layer before any content-quality filtering."""

        document = fitz.open(path)
        pages: set[int] = set()
        try:
            for page_number, page in enumerate(document, start=1):
                blocks = page.get_text("dict").get("blocks", [])
                if any(
                    str(span.get("text", "")).strip()
                    for block in blocks
                    if block.get("type") == 0
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ):
                    pages.add(page_number)
        finally:
            document.close()
        return pages

    def _extract_text_chunks(self, path: str) -> List[TextChunk]:
        doc = fitz.open(path)
        chunks: List[TextChunk] = []

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        size = span.get("size", 11.0)
                        chunks.append(TextChunk(
                            page      = page_num,
                            text      = text,
                            bbox      = tuple(span.get("bbox", (0, 0, 0, 0))),
                            font_size = round(size, 1),
                            is_heading = size >= self.HEADING_MIN_SIZE,
                        ))
        doc.close()
        # Keep source spans intact: the canonical chunker needs their exact
        # order and geometry to bind each window to its own bbox and section.
        return chunks

    def _merge_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        if not chunks:
            return chunks
        merged = []
        buf = chunks[0]
        for c in chunks[1:]:
            same_page = c.page == buf.page
            small = len(buf.text.split()) < self.MIN_CHUNK_WORDS
            if same_page and small and not buf.is_heading and not c.is_heading:
                buf = TextChunk(
                    page=buf.page,
                    text=buf.text + " " + c.text,
                    bbox=(
                        min(buf.bbox[0], c.bbox[0]),
                        min(buf.bbox[1], c.bbox[1]),
                        max(buf.bbox[2], c.bbox[2]),
                        max(buf.bbox[3], c.bbox[3]),
                    ),
                    font_size=buf.font_size,
                    is_heading=False,
                )
            else:
                merged.append(buf)
                buf = c
        merged.append(buf)
        return merged

    # ─── Tables ───────────────────────────────────────────────────────────────

    def _extract_tables(self, path: str) -> List[TableData]:
        tables: List[TableData] = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    for located_table in page.find_tables():
                        tbl = located_table.extract()
                        if not tbl or len(tbl) < 2:
                            continue
                        headers = [str(h or "").strip() for h in tbl[0]]
                        rows    = [[str(c or "").strip() for c in row]
                                   for row in tbl[1:] if any(c for c in row)]
                        markdown = self._table_to_markdown(headers, rows)
                        bbox = tuple(float(value) for value in located_table.bbox)
                        tables.append(TableData(
                            page=page_num,
                            headers=headers,
                            rows=rows,
                            markdown=markdown,
                            bbox=bbox,
                        ))
        except Exception as e:
            logger.warning("[PDFProcessor] Table extraction warning: %s", e)
        return tables

    def _table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        if not headers:
            return ""
        def cell(value: str) -> str:
            return " ".join(value.split()).replace("|", "\\|")

        width = len(headers)
        normalized_rows = [
            (row + [""] * width)[:width]
            for row in rows
        ]
        sep  = "| " + " | ".join("---" for _ in headers) + " |"
        head = "| " + " | ".join(cell(value) for value in headers) + " |"
        body = "\n".join(
            "| " + " | ".join(cell(value) for value in row) + " |"
            for row in normalized_rows
        )
        return f"{head}\n{sep}\n{body}"

    # ─── Images / OCR ─────────────────────────────────────────────────────────

    def _extract_images(self, path: str, stem: str) -> List[ImageData]:
        images: List[ImageData] = []
        doc = fitz.open(path)

        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    w, h = pil_img.size

                    if w < self.OCR_MIN_DIM or h < self.OCR_MIN_DIM:
                        continue

                    # Save image
                    img_path = self.img_dir / f"{stem}_p{page_num}_i{img_idx}.png"
                    pil_img.save(str(img_path))

                    # OCR
                    ocr_text = ""
                    if OCR_AVAILABLE and w > 100 and h > 100:
                        try:
                            ocr_text = pytesseract.image_to_string(
                                pil_img, config="--psm 6"
                            ).strip()
                        except Exception:
                            pass

                    rectangles = page.get_image_rects(xref)
                    if rectangles:
                        bbox = (
                            min(rect.x0 for rect in rectangles),
                            min(rect.y0 for rect in rectangles),
                            max(rect.x1 for rect in rectangles),
                            max(rect.y1 for rect in rectangles),
                        )
                    else:
                        bbox = (0.0, 0.0, float(w), float(h))

                    images.append(ImageData(
                        page=page_num,
                        index=img_idx,
                        width=w,
                        height=h,
                        ocr_text=ocr_text,
                        path=str(img_path),
                        bbox=bbox,
                    ))
                except Exception as e:
                    logger.warning(
                        "[PDFProcessor] Image %s p%s: %s", img_idx, page_num, e
                    )

        doc.close()
        return images

    # ─── Text Building ────────────────────────────────────────────────────────

    def _build_full_text(self, chunks: List[TextChunk]) -> str:
        parts = []
        current_page = None
        for c in chunks:
            if c.page != current_page:
                parts.append(f"\n\n--- Página {c.page} ---\n")
                current_page = c.page
            parts.append(c.text)
        return "\n".join(parts)

    def _build_training_corpus(
        self,
        chunks:  List[CanonicalChunk],
        tables:  List[TableData],
        images:  List[ImageData],
    ) -> str:
        """
        Build training text exclusively from verified canonical records.

        Synthetic section/page headers are deliberately excluded: they are not
        present in the authorized PDF.  Table and optional OCR records are
        already explicit, tagged canonical chunks and are included exactly once.
        """
        del tables, images  # retained in the public compatibility signature
        return "\n".join(chunk.text for chunk in chunks)

    def _detect_language(self, sample: str) -> str:
        pt_markers = ["de", "do", "da", "em", "um", "uma", "para", "com", "que"]
        en_markers = ["the", "and", "for", "with", "that", "this", "are", "from"]
        pt_score = sum(1 for w in pt_markers if f" {w} " in sample.lower())
        en_score = sum(1 for w in en_markers if f" {w} " in sample.lower())
        if pt_score > en_score:
            return "pt-BR"
        elif en_score > 0:
            return "en"
        return "unknown"

    # ─── Persistence ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_staged_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _publish_staged(stage: Path, destination: Path) -> None:
        """Publish an immutable artifact, preserving conflicting recovery data."""

        if destination.exists():
            if (
                destination.stat().st_size == stage.stat().st_size
                and sha256_file(destination) == sha256_file(stage)
            ):
                return
            raise RuntimeError(
                f"refusing to overwrite conflicting corpus artifact: {destination.name}"
            )
        os.replace(stage, destination)

    def _save_corpus(
        self,
        result: DocumentResult,
        identity: DocumentIdentity,
        source_path: Path,
    ) -> None:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(result.filename).stem).strip("._")
        safe_stem = safe_stem[:80] or "document"
        prefix = f"{identity.document_id}_{safe_stem}"
        corpus_path = self.output_dir / f"{prefix}_corpus.txt"
        chunks_path = self.output_dir / f"{prefix}_chunks.jsonl"
        meta_path = self.output_dir / f"{prefix}_meta.json"
        transaction_dir = (
            self.output_dir
            / ".ingest-staging"
            / f"{identity.document_id}-{uuid.uuid4().hex}"
        )
        transaction_dir.mkdir(parents=True, exist_ok=False)
        staged_corpus = transaction_dir / corpus_path.name
        staged_chunks = transaction_dir / chunks_path.name
        staged_meta = transaction_dir / meta_path.name
        self._write_staged_text(
            staged_corpus,
            result.training_corpus + ("\n" if result.training_corpus else ""),
        )
        write_chunks_jsonl(staged_chunks, result.canonical_chunks)

        indexed_at = datetime.now(timezone.utc).isoformat()
        metadata_payload = {
            "document_id": result.document_id,
            "document_sha256": result.document_sha256,
            "filename": result.filename,
            "source_bytes": identity.byte_size,
            "pages": result.total_pages,
            "words": result.word_count,
            "stats": result.stats,
            "metadata": result.metadata,
            "language": result.language_hint,
            "corpus_file": corpus_path.name,
            "corpus_file_sha256": sha256_file(staged_corpus),
            "corpus_bytes": staged_corpus.stat().st_size,
            "chunks_file": chunks_path.name,
            "chunks_sha256": sha256_file(staged_chunks),
            "chunks_bytes": staged_chunks.stat().st_size,
            "chunks_records": len(result.canonical_chunks),
            "indexed_at": indexed_at,
            "sovereign_mode": self.policy.enabled,
            "ocr_provenance": (
                "pytesseract_pretrained_non_sovereign"
                if self.policy.allow_pretrained_ocr
                else "disabled"
            ),
        }
        self._write_staged_text(
            staged_meta,
            json.dumps(
                metadata_payload, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
        )

        document_manifest = DocumentManifest(
            document_id=identity.document_id,
            document_sha256=identity.document_sha256,
            filename=identity.filename,
            byte_size=identity.byte_size,
            pages=result.total_pages,
            words=result.word_count,
            chunks=len(result.canonical_chunks),
            language=result.language_hint,
            indexed_at=indexed_at,
            chunks_file=chunks_path.name,
            chunks_sha256=sha256_file(staged_chunks),
            chunks_bytes=staged_chunks.stat().st_size,
            chunks_records=len(result.canonical_chunks),
            corpus_file=corpus_path.name,
            corpus_file_sha256=sha256_file(staged_corpus),
            corpus_bytes=staged_corpus.stat().st_size,
            metadata_file=meta_path.name,
            metadata_sha256=sha256_file(staged_meta),
            metadata_bytes=staged_meta.stat().st_size,
        )

        manifest_path = self.output_dir / "corpus_manifest.json"
        lock_path = self.output_dir / ".corpus_manifest.lock"
        with corpus_update_lock(lock_path):
            current_identity = DocumentIdentity.from_path(source_path)
            if current_identity != identity:
                raise RuntimeError("source PDF changed before corpus publication")
            manifest_snapshot = (
                sha256_file(manifest_path) if manifest_path.exists() else ""
            )
            manifest = CorpusManifest.load(manifest_path)
            existing = manifest.documents.get(identity.document_id)
            if existing is not None:
                existing_chunks = read_chunks_jsonl(
                    self.output_dir / existing.chunks_file
                )
                existing_ids = [chunk.chunk_id for chunk in existing_chunks]
                new_ids = [chunk.chunk_id for chunk in result.canonical_chunks]
                if existing_ids != new_ids:
                    raise RuntimeError(
                        "the same source document already has a different canonical "
                        "chunking; explicit corpus migration is required"
                    )
                result.canonical_chunks = existing_chunks
                result.training_corpus = (
                    self.output_dir / existing.corpus_file
                ).read_text(encoding="utf-8").rstrip("\n")
                result.corpus_path = str(self.output_dir / existing.corpus_file)
                result.chunks_path = str(self.output_dir / existing.chunks_file)
                logger.info("[PDFProcessor] Reused corpus -> %s", result.corpus_path)
                return

            existing_chunks = manifest.validate_artifacts(self.output_dir)
            self._publish_staged(staged_corpus, corpus_path)
            self._publish_staged(staged_chunks, chunks_path)
            self._publish_staged(staged_meta, meta_path)
            if DocumentIdentity.from_path(source_path) != identity:
                raise RuntimeError("source PDF changed during corpus publication")
            manifest.upsert_document(
                document_manifest,
                [*existing_chunks, *result.canonical_chunks],
            )
            manifest.save(manifest_path, expected_sha256=manifest_snapshot)

        result.corpus_path = str(corpus_path)
        result.chunks_path = str(chunks_path)
        logger.info("[PDFProcessor] Saved corpus -> %s", corpus_path)
