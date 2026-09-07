from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import fitz

from jarvis_localhost.corpus.manifest import CorpusManifest
from jarvis_localhost.processing.pdf_processor import (
    ImageData,
    PDFProcessor,
    TableData,
)
from jarvis_localhost.sovereign import SovereignModeViolation, SovereignPolicy


def make_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_textbox(
                fitz.Rect(36, 36, 560, 780),
                text,
                fontsize=11,
            )
    document.save(path)
    document.close()


def make_table_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    columns = [50, 200, 350]
    rows = [50, 90, 130, 170]
    for x_coordinate in columns:
        page.draw_line((x_coordinate, rows[0]), (x_coordinate, rows[-1]))
    for y_coordinate in rows:
        page.draw_line((columns[0], y_coordinate), (columns[-1], y_coordinate))
    page.insert_text((60, 75), "Item")
    page.insert_text((210, 75), "Valor")
    page.insert_text((60, 115), "A")
    page.insert_text((210, 115), "10")
    page.insert_text((60, 155), "B")
    page.insert_text((210, 155), "20")
    document.save(path)
    document.close()


class PDFProcessorTests(unittest.TestCase):
    def test_text_pdf_preserves_page_bbox_hash_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "manual.pdf"
            make_pdf(
                pdf,
                [
                    "O sistema local utiliza um temporizador para controlar o contador principal.",
                    "A segunda pagina descreve o sensor e o circuito de alimentacao local.",
                ],
            )
            processor = PDFProcessor(
                output_dir=root / "corpus",
                image_dir=root / "images",
                policy=SovereignPolicy(),
                target_words=32,
                overlap_words=8,
            )
            result = processor.process(str(pdf))
            self.assertEqual(result.total_pages, 2)
            self.assertEqual({chunk.page for chunk in result.canonical_chunks}, {1, 2})
            self.assertTrue(result.document_id.startswith("doc_"))
            self.assertEqual(len(result.document_sha256), 64)
            self.assertTrue(all(len(chunk.bbox) == 4 for chunk in result.canonical_chunks))
            self.assertTrue(Path(result.chunks_path).is_file())
            manifest = json.loads(
                (root / "corpus" / "corpus_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            document = manifest["documents"][result.document_id]
            self.assertEqual(document["document_sha256"], result.document_sha256)
            self.assertEqual(document["chunks_records"], len(result.canonical_chunks))
            self.assertEqual(len(document["chunks_sha256"]), 64)
            self.assertEqual(len(document["metadata_sha256"]), 64)
            self.assertEqual(manifest["aggregate"]["chunks"], len(result.canonical_chunks))
            self.assertEqual(result.stats["ocr_enabled"], 0)
            CorpusManifest.load(root / "corpus" / "corpus_manifest.json")

    def test_short_text_is_canonical_and_training_has_no_synthetic_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "formula.pdf"
            make_pdf(pdf, ["E=mc2"])
            processor = PDFProcessor(
                output_dir=root / "corpus",
                image_dir=root / "images",
                policy=SovereignPolicy(),
                target_words=32,
                overlap_words=4,
            )
            result = processor.process(str(pdf))
            self.assertEqual(len(result.canonical_chunks), 1)
            self.assertEqual(result.canonical_chunks[0].text, "E=mc2")
            self.assertEqual(result.training_corpus, "E=mc2")
            self.assertNotIn("===", result.training_corpus)
            self.assertNotIn("Página", result.training_corpus)

    def test_table_and_optional_ocr_are_separate_tagged_canonical_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "mixed.pdf"
            make_pdf(pdf, ["Texto local"])
            processor = PDFProcessor(
                output_dir=root / "corpus",
                image_dir=root / "images",
                policy=SovereignPolicy(enabled=False, allow_pretrained_ocr=True),
                target_words=32,
                overlap_words=4,
            )
            table = TableData(
                page=1,
                headers=["Item", "Valor"],
                rows=[["A", "10"]],
                markdown="| Item | Valor |\n| --- | --- |\n| A | 10 |",
                bbox=(40, 100, 300, 180),
            )
            image = ImageData(
                page=1,
                index=0,
                width=200,
                height=100,
                ocr_text="rotulo lido por OCR opcional",
                path=str(root / "images" / "i.png"),
                bbox=(320, 100, 520, 200),
            )
            with mock.patch.object(processor, "_extract_tables", return_value=[table]), mock.patch.object(
                processor, "_extract_images", return_value=[image]
            ):
                result = processor.process(str(pdf))
            by_type = {chunk.content_type: chunk for chunk in result.canonical_chunks}
            self.assertIn("table", by_type)
            self.assertIn("ocr", by_type)
            self.assertEqual(by_type["table"].bbox, (40.0, 100.0, 300.0, 180.0))
            self.assertEqual(by_type["table"].extraction_method, "pdfplumber_table")
            self.assertEqual(
                by_type["ocr"].extraction_method,
                "pytesseract_pretrained_non_sovereign",
            )
            self.assertEqual(by_type["ocr"].source_ref, "image:0")
            self.assertIn("rotulo lido por OCR opcional", result.training_corpus)

    def test_real_pdf_table_is_citable_with_extraction_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "table.pdf"
            make_table_pdf(pdf)
            processor = PDFProcessor(
                output_dir=root / "corpus",
                image_dir=root / "images",
                policy=SovereignPolicy(),
                target_words=32,
                overlap_words=4,
            )
            result = processor.process(str(pdf))
            table_chunks = [
                chunk
                for chunk in result.canonical_chunks
                if chunk.content_type == "table"
            ]
            self.assertEqual(len(result.tables), 1)
            self.assertEqual(len(table_chunks), 1)
            self.assertEqual(table_chunks[0].page, 1)
            self.assertEqual(table_chunks[0].bbox, (50.0, 50.0, 350.0, 170.0))
            self.assertEqual(table_chunks[0].extraction_method, "pdfplumber_table")
            self.assertIn("Item", table_chunks[0].text)
            self.assertIn("20", table_chunks[0].text)

    def test_concurrent_ingest_merges_manifest_without_lost_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pdf"
            second = root / "second.pdf"
            make_pdf(first, ["primeiro documento curto e valido"])
            make_pdf(second, ["segundo documento curto e valido"])
            corpus_root = root / "corpus"

            def ingest(path: Path):
                return PDFProcessor(
                    output_dir=corpus_root,
                    image_dir=root / "images",
                    policy=SovereignPolicy(),
                    target_words=32,
                    overlap_words=4,
                ).process(str(path))

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(ingest, [first, second]))
            manifest = CorpusManifest.load(corpus_root / "corpus_manifest.json")
            self.assertEqual(set(manifest.documents), {item.document_id for item in results})
            self.assertEqual(manifest._aggregate()["documents"], 2)
            self.assertEqual(manifest._aggregate()["chunks"], 2)

    def test_blank_page_is_rejected_in_sovereign_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "scan.pdf"
            make_pdf(pdf, ["Esta pagina possui uma camada textual valida e local.", ""])
            processor = PDFProcessor(
                output_dir=root / "corpus",
                image_dir=root / "images",
                policy=SovereignPolicy(),
            )
            with self.assertRaises(SovereignModeViolation):
                processor.process(str(pdf))


if __name__ == "__main__":
    unittest.main()
