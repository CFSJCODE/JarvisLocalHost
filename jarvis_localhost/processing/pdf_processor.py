"""
pdf_processor.py — J.A.R.V.I.S Multimodal Document Processor
Extracts text, images, tables and metadata from PDF documents.
"""

import io
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import fitz          # PyMuPDF
import pdfplumber
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


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


@dataclass
class ImageData:
    page: int
    index: int
    width: int
    height: int
    ocr_text: str       # text extracted via OCR
    path: str           # saved image path


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


# ─── Processor ────────────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Multimodal PDF extractor for J.A.R.V.I.S.
    Handles text, tables, images (with OCR) from PDF documents.
    """

    MIN_CHUNK_WORDS   = 15
    HEADING_MIN_SIZE  = 13.0
    OCR_MIN_DIM       = 50   # pixels — skip tiny images

    def __init__(self, output_dir: str = "data/embeddings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = Path("data/extracted_images")
        self.img_dir.mkdir(parents=True, exist_ok=True)

    # ─── Public API ───────────────────────────────────────────────────────────

    def process(self, pdf_path: str) -> DocumentResult:
        """Full pipeline: extract everything from a PDF."""
        path = Path(pdf_path)
        print(f"[PDFProcessor] Processing: {path.name}")

        metadata    = self._extract_metadata(pdf_path)
        chunks      = self._extract_text_chunks(pdf_path)
        tables      = self._extract_tables(pdf_path)
        images      = self._extract_images(pdf_path, path.stem)
        full_text   = self._build_full_text(chunks)
        corpus      = self._build_training_corpus(chunks, tables, images)
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
            }
        )

        self._save_corpus(result)
        print(f"[PDFProcessor] Done - {word_count} words, "
              f"{len(chunks)} chunks, {len(tables)} tables, {len(images)} images")
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
                        if not text or len(text.split()) < 2:
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
        # Merge small consecutive chunks on same page
        return self._merge_chunks(chunks)

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
                    bbox=buf.bbox,
                    font_size=buf.font_size,
                    is_heading=False,
                )
            else:
                if len(buf.text.split()) >= 3:
                    merged.append(buf)
                buf = c
        if len(buf.text.split()) >= 3:
            merged.append(buf)
        return merged

    # ─── Tables ───────────────────────────────────────────────────────────────

    def _extract_tables(self, path: str) -> List[TableData]:
        tables: List[TableData] = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    for tbl in page.extract_tables():
                        if not tbl or len(tbl) < 2:
                            continue
                        headers = [str(h or "").strip() for h in tbl[0]]
                        rows    = [[str(c or "").strip() for c in row]
                                   for row in tbl[1:] if any(c for c in row)]
                        markdown = self._table_to_markdown(headers, rows)
                        tables.append(TableData(
                            page=page_num,
                            headers=headers,
                            rows=rows,
                            markdown=markdown,
                        ))
        except Exception as e:
            print(f"[PDFProcessor] Table extraction warning: {e}")
        return tables

    def _table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        if not headers:
            return ""
        sep  = "| " + " | ".join("---" for _ in headers) + " |"
        head = "| " + " | ".join(headers) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
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

                    images.append(ImageData(
                        page=page_num,
                        index=img_idx,
                        width=w,
                        height=h,
                        ocr_text=ocr_text,
                        path=str(img_path),
                    ))
                except Exception as e:
                    print(f"[PDFProcessor] Image {img_idx} p{page_num}: {e}")

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
        chunks:  List[TextChunk],
        tables:  List[TableData],
        images:  List[ImageData],
    ) -> str:
        """
        Builds a clean corpus string ideal for model training.
        Combines text chunks, table markdown, and OCR from images.
        """
        parts = []

        # Text
        parts.append("=== CONTEÚDO TEXTUAL ===")
        for c in chunks:
            if len(c.text.split()) >= self.MIN_CHUNK_WORDS:
                parts.append(c.text)

        # Tables
        if tables:
            parts.append("\n=== TABELAS ===")
            for t in tables:
                parts.append(t.markdown)

        # OCR from images
        img_ocr = [img.ocr_text for img in images if len(img.ocr_text.split()) > 5]
        if img_ocr:
            parts.append("\n=== TEXTO EXTRAÍDO DE IMAGENS (OCR) ===")
            parts.extend(img_ocr)

        corpus = "\n".join(parts)
        # Clean up
        corpus = re.sub(r"\n{3,}", "\n\n", corpus)
        corpus = re.sub(r"[ \t]{2,}", " ", corpus)
        return corpus.strip()

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

    def _save_corpus(self, result: DocumentResult) -> None:
        stem = Path(result.filename).stem
        corpus_path = self.output_dir / f"{stem}_corpus.txt"
        corpus_path.write_text(result.training_corpus, encoding="utf-8")

        meta_path = self.output_dir / f"{stem}_meta.json"
        meta_path.write_text(
            json.dumps({
                "filename": result.filename,
                "pages":    result.total_pages,
                "words":    result.word_count,
                "stats":    result.stats,
                "metadata": result.metadata,
                "language": result.language_hint,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[PDFProcessor] Saved corpus -> {corpus_path}")
