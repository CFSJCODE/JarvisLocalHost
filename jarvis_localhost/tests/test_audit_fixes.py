"""Regression tests for the 2026-08-30 technical audit fixes.

Covers:
  F1 - PDFProcessor must refuse to persist a zero-chunk ("ghost") document,
       and CorpusManifest.remove_document / tools.corpus_hygiene must be able
       to hygienically retire ghost entries that already exist on disk.
  F2 - rag.grounding._clean_passage / extractive_answer must no longer
       rewrite every "compu*" word to a fixed, unrelated phrase, and must no
       longer depend on hardcoded document vocabulary ("Robótica",
       "Capítulo").
  F4 - corpus.chunker.canonicalize_spans must actually enforce
       minimum_words instead of silently ignoring it, without dropping real
       content or exceeding minimum_words > target_words.
  F6 - rag.grounding._is_index_or_tabular_noise must not discard legitimate
       numeric technical facts (datasheet values such as "800mw", "8 MB")
       while still discarding index/table-of-contents/section-numbering
       noise. Real MQ-5/ESP32-C6 datasheet sentences from the audit's E2E
       test are used as regression fixtures, not just synthetic examples.
  F10 - A crashed training run (process killed or crashed outright) used to
        lose ALL progress: JarvisTrainer had no way to persist optimizer
        state, RNG state or its position in the step loop, and even a
        graceful cancel's own cleanup deleted the whole staging directory.
        JarvisTrainer._checkpoint now also writes a jarvis_{tag}.trainer_
        state.pt sidecar, JarvisTrainer.apply_resume_state restores it, and
        core.brain._find_resumable_training locates it automatically at the
        start of a new run -- but only ever resumes when the corpus and
        tokenizer digests are byte-identical to what the checkpoint was
        produced against, refusing (never silently degrading) otherwise.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz
import numpy as np
import torch

from jarvis_localhost.ai.language_model import JarvisConfig, JarvisTransformer
from jarvis_localhost.ai.tokenizer import JarvisTokenizer
from jarvis_localhost.ai.trainer import JarvisTrainer, TrainConfig
from jarvis_localhost.core.brain import _find_resumable_training
from jarvis_localhost.corpus.chunker import PageSpan, canonicalize_spans
from jarvis_localhost.corpus.manifest import CorpusManifest
from jarvis_localhost.corpus.provenance import CanonicalChunk, DocumentIdentity
from jarvis_localhost.processing.pdf_processor import EmptyDocumentError, PDFProcessor
from jarvis_localhost.rag.citations import Citation
from jarvis_localhost.rag.grounding import (
    _bare_number_digit_ratio,
    _is_index_or_tabular_noise,
    extractive_answer,
)
from jarvis_localhost.sovereign import SovereignPolicy
from jarvis_localhost.tools.corpus_hygiene import find_ghost_documents


def _blank_pdf(path: Path) -> None:
    """A one-page PDF with no text layer, no tables and no images."""

    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()


class EmptyDocumentRejectionTests(unittest.TestCase):
    def test_zero_chunk_document_is_rejected_not_silently_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "scanned_no_text_layer.pdf"
            _blank_pdf(pdf)
            processor = PDFProcessor(
                output_dir=root / "embeddings",
                image_dir=root / "images",
                # Reproduces the historical ghost-document trigger: sovereign
                # mode off (so require_text_layer is a no-op) and OCR
                # disabled (the default), on a page with zero extractable text.
                policy=SovereignPolicy(enabled=False, allow_pretrained_ocr=False),
            )
            with self.assertRaises(EmptyDocumentError):
                processor.process(str(pdf))
            # Nothing must have been persisted: no manifest, no artifacts.
            manifest_path = root / "embeddings" / "corpus_manifest.json"
            self.assertFalse(manifest_path.exists())

    def test_document_with_real_text_is_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "real.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_textbox(
                fitz.Rect(36, 36, 560, 780),
                "Este documento contem texto real extraivel para indexacao.",
                fontsize=11,
            )
            document.save(pdf)
            document.close()
            processor = PDFProcessor(
                output_dir=root / "embeddings",
                image_dir=root / "images",
                policy=SovereignPolicy(enabled=False, allow_pretrained_ocr=False),
            )
            result = processor.process(str(pdf))
            self.assertGreater(len(result.canonical_chunks), 0)


class CorpusHygieneTests(unittest.TestCase):
    def test_remove_document_drops_entry_and_deletes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            healthy_pdf = root / "healthy.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_textbox(
                fitz.Rect(36, 36, 560, 780),
                "Conteudo real suficiente para gerar pelo menos um chunk valido.",
                fontsize=11,
            )
            document.save(healthy_pdf)
            document.close()
            ghost_pdf = root / "ghost.pdf"
            _blank_pdf(ghost_pdf)

            embeddings = root / "embeddings"
            processor = PDFProcessor(
                output_dir=embeddings,
                image_dir=root / "images",
                policy=SovereignPolicy(enabled=False, allow_pretrained_ocr=False),
            )
            processor.process(str(healthy_pdf))
            # Simulate a ghost document that predates EmptyDocumentError by
            # writing its manifest entry directly (bypassing the new guard),
            # exactly matching the shape found in the real production corpus.
            manifest_path = embeddings / "corpus_manifest.json"
            manifest = CorpusManifest.load(manifest_path)
            from jarvis_localhost.corpus.manifest import DocumentManifest
            from jarvis_localhost.corpus.provenance import canonical_document_id

            empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ghost_sha = "b" * 64
            ghost_id = canonical_document_id(ghost_sha)
            for name, content in (
                ("ghost_chunks.jsonl", ""),
                ("ghost_corpus.txt", ""),
                (
                    "ghost_meta.json",
                    json.dumps(
                        {
                            "document_id": ghost_id,
                            "document_sha256": ghost_sha,
                            "chunks_file": "ghost_chunks.jsonl",
                            "corpus_file": "ghost_corpus.txt",
                        }
                    ),
                ),
            ):
                (embeddings / name).write_text(content, encoding="utf-8")

            ghost_entry = DocumentManifest(
                document_id=ghost_id,
                document_sha256=ghost_sha,
                filename="ghost.pdf",
                byte_size=ghost_pdf.stat().st_size,
                pages=1,
                words=0,
                chunks=0,
                language="unknown",
                indexed_at="2026-01-01T00:00:00+00:00",
                chunks_file="ghost_chunks.jsonl",
                chunks_sha256=empty_sha,
                chunks_bytes=0,
                chunks_records=0,
                corpus_file="ghost_corpus.txt",
                corpus_file_sha256=empty_sha,
                corpus_bytes=0,
                metadata_file="ghost_meta.json",
                metadata_sha256=__import__("hashlib")
                .sha256((embeddings / "ghost_meta.json").read_bytes())
                .hexdigest(),
                metadata_bytes=(embeddings / "ghost_meta.json").stat().st_size,
            )
            manifest.documents[ghost_id] = ghost_entry
            manifest.save(manifest_path)

            manifest = CorpusManifest.load(manifest_path)
            ghosts = find_ghost_documents(manifest)
            self.assertEqual([g["document_id"] for g in ghosts], [ghost_id])

            manifest.remove_document(ghost_id, embeddings)
            manifest.save(manifest_path)

            self.assertFalse((embeddings / "ghost_chunks.jsonl").exists())
            self.assertFalse((embeddings / "ghost_corpus.txt").exists())
            self.assertFalse((embeddings / "ghost_meta.json").exists())
            reloaded = CorpusManifest.load(manifest_path)
            self.assertNotIn(ghost_id, reloaded.documents)
            self.assertEqual(len(reloaded.documents), 1)

    def test_document_with_nonzero_chunks_but_zero_words_is_also_flagged(self) -> None:
        # (audit fix, found during the 2026-08-31 continuation) Real
        # production evidence: manual_rocket_A3.pdf (a vector-graphic PDF)
        # produced 2 canonical chunks via pdfplumber's table extractor, but
        # both chunks are just empty table-grid syntax ("| | | --- |") with
        # zero real words -- the same "contributes nothing to retrieval"
        # problem as chunks == 0, just a different extraction failure mode.
        # The original chunks == 0 check alone did not catch this;
        # find_ghost_documents must also check words == 0.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            healthy_pdf = root / "healthy.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_textbox(
                fitz.Rect(36, 36, 560, 780),
                "Conteudo real suficiente para gerar pelo menos um chunk valido.",
                fontsize=11,
            )
            document.save(healthy_pdf)
            document.close()

            embeddings = root / "embeddings"
            processor = PDFProcessor(
                output_dir=embeddings,
                image_dir=root / "images",
                policy=SovereignPolicy(enabled=False, allow_pretrained_ocr=False),
            )
            processor.process(str(healthy_pdf))

            manifest_path = embeddings / "corpus_manifest.json"
            manifest = CorpusManifest.load(manifest_path)
            # Captured before any ghost-document files are written below: the
            # closed-world check in validate_artifacts requires every file on
            # disk to be a manifest-authorized document, so this snapshot of
            # "just the healthy document" must be taken first.
            existing_chunks = manifest.validate_artifacts(embeddings)
            from jarvis_localhost.corpus.manifest import DocumentManifest
            from jarvis_localhost.corpus.provenance import canonical_document_id
            import hashlib

            table_sha = "c" * 64
            table_id = canonical_document_id(table_sha)
            table_document = DocumentIdentity(
                document_id=table_id,
                document_sha256=table_sha,
                filename="manual_rocket_A3.pdf",
                byte_size=1904154,
            )
            # Real production shape: pdfplumber's table extractor turns a
            # vector-graphic PDF's decorative lines into "cells" holding only
            # grid syntax -- non-empty text (CanonicalChunk.build's own
            # emptiness check only rejects whitespace-only text), but zero
            # real words once word-counted.
            table_chunks = [
                CanonicalChunk.build(
                    table_document,
                    page=1,
                    section="",
                    bbox=None,
                    text="| | | --- |",
                    ordinal=0,
                    extraction_method="table",
                    content_type="table",
                ),
                CanonicalChunk.build(
                    table_document,
                    page=2,
                    section="",
                    bbox=None,
                    text="| | | | --- | --- |",
                    ordinal=1,
                    extraction_method="table",
                    content_type="table",
                ),
            ]
            chunks_content = "".join(
                json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                for chunk in table_chunks
            )
            for name, content in (
                ("table_chunks.jsonl", chunks_content),
                ("table_corpus.txt", ""),
                (
                    "table_meta.json",
                    json.dumps(
                        {
                            "document_id": table_id,
                            "document_sha256": table_sha,
                            "chunks_file": "table_chunks.jsonl",
                            "corpus_file": "table_corpus.txt",
                        }
                    ),
                ),
            ):
                (embeddings / name).write_text(content, encoding="utf-8")

            def sha256_of(name: str) -> str:
                return hashlib.sha256((embeddings / name).read_bytes()).hexdigest()

            table_entry = DocumentManifest(
                document_id=table_id,
                document_sha256=table_sha,
                filename="manual_rocket_A3.pdf",
                byte_size=1904154,
                pages=2,
                words=0,
                chunks=2,
                language="unknown",
                indexed_at="2026-01-01T00:00:00+00:00",
                chunks_file="table_chunks.jsonl",
                chunks_sha256=sha256_of("table_chunks.jsonl"),
                chunks_bytes=(embeddings / "table_chunks.jsonl").stat().st_size,
                chunks_records=2,
                corpus_file="table_corpus.txt",
                corpus_file_sha256=sha256_of("table_corpus.txt"),
                corpus_bytes=0,
                metadata_file="table_meta.json",
                metadata_sha256=sha256_of("table_meta.json"),
                metadata_bytes=(embeddings / "table_meta.json").stat().st_size,
            )
            # upsert_document (not a bare dict assignment) is required here:
            # the manifest also carries a top-level corpus_sha256 digest over
            # every authorized chunk in the whole corpus, which save() itself
            # re-verifies -- it must be recomputed to include the new ghost
            # document's 2 chunks alongside the pre-existing healthy one.
            manifest.upsert_document(table_entry, [*existing_chunks, *table_chunks])
            manifest.save(manifest_path)

            reloaded = CorpusManifest.load(manifest_path)
            ghosts = find_ghost_documents(reloaded)
            ghost_ids = [g["document_id"] for g in ghosts]
            self.assertIn(table_id, ghost_ids)
            self.assertEqual(len(ghosts), 1)
            flagged = ghosts[0]
            self.assertEqual(flagged["chunks"], 2)
            self.assertEqual(flagged["words"], 0)

            reloaded.remove_document(table_id, embeddings)
            reloaded.save(manifest_path)
            self.assertFalse((embeddings / "table_chunks.jsonl").exists())
            final = CorpusManifest.load(manifest_path)
            self.assertNotIn(table_id, final.documents)


class GroundingCleanupTests(unittest.TestCase):
    def _citation(self, quote: str) -> Citation:
        return Citation(
            evidence_id="E1",
            document_id="doc_1",
            document_sha256="a" * 64,
            filename="arquitetura_de_computadores.pdf",
            page=1,
            section="",
            chunk_id="chunk_1",
            bbox=(0.0, 0.0, 1.0, 1.0),
            quote=quote,
            score=1.0,
        )

    def test_words_starting_with_compu_are_not_rewritten(self) -> None:
        quote = (
            "O computador moderno depende de arquitetura de computacao "
            "eficiente para processar instrucoes rapidamente."
        )
        answer = extractive_answer(
            "O que e um computador?", [self._citation(quote)]
        )
        self.assertNotIn("manipulador", answer.lower())
        self.assertIn("computador", answer.lower())

    def test_word_boundary_hyphenation_join_still_works(self) -> None:
        # The generic de-hyphenation rule (join "word-\nword" across a line
        # break) must still function after removing the compu*-specific hack.
        quote = "O sistema opera de forma inde- pendente do hardware instalado."
        answer = extractive_answer(
            "Como o sistema opera?", [self._citation(quote)]
        )
        self.assertIn("independente", answer.lower())


class ChunkerMinimumWordsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = DocumentIdentity("doc_test", "a" * 64, "test.pdf", 1)

    def test_consecutive_heading_only_segments_merge_forward(self) -> None:
        # Reproduces the real pattern found in a table of contents / numbered
        # section list (empirically observed in the ESP32-C6 datasheet real
        # E2E test): each heading immediately flushes the previous one as its
        # own 1-2 word segment before _segments_for_page can attach any body
        # text to it. These must not remain permanent orphan chunks.
        spans = [
            PageSpan(page=1, text="1", bbox=(0, 0, 1, 1), is_heading=True),
            PageSpan(page=1, text="1.1", bbox=(0, 0, 1, 1), is_heading=True),
            PageSpan(page=1, text="1.2", bbox=(0, 0, 1, 1), is_heading=True),
            PageSpan(
                page=1,
                text="Conteudo real desta subsecao com bastante texto util para o leitor.",
                bbox=(0, 0, 1, 1),
            ),
        ]
        chunks = canonicalize_spans(
            self.document, spans, target_words=220, overlap_words=40, minimum_words=12
        )
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("1 1.1 1.2 Conteudo real"))
        self.assertGreaterEqual(len(chunks[0].text.split()), 12)

    def test_orphan_heading_before_a_table_stays_separate_from_the_table(self) -> None:
        # A heading immediately followed by a *table* (different content_type)
        # must never be absorbed into the table's markdown text.
        spans = [
            PageSpan(page=1, text="3", bbox=(0, 0, 1, 1), is_heading=True),
            PageSpan(
                page=1,
                text="| A | B |",
                bbox=(0, 0, 1, 1),
                content_type="table",
                source_ref="table:0",
            ),
        ]
        chunks = canonicalize_spans(
            self.document, spans, target_words=220, overlap_words=40, minimum_words=12
        )
        self.assertEqual(len(chunks), 2)
        content_types = {c.content_type for c in chunks}
        self.assertEqual(content_types, {"text", "table"})
        text_chunk = next(c for c in chunks if c.content_type == "text")
        self.assertEqual(text_chunk.text, "3")

    def test_short_tail_window_merges_instead_of_staying_below_minimum(self) -> None:
        words = " ".join(f"w{i}" for i in range(228))
        spans = [PageSpan(page=1, text=words, bbox=(0, 0, 1, 1))]
        chunks = canonicalize_spans(
            self.document, spans, target_words=220, overlap_words=2, minimum_words=12
        )
        self.assertTrue(all(len(c.text.split()) >= 12 for c in chunks))
        # No content lost: every generated word appears in some chunk.
        covered = " ".join(c.text for c in chunks)
        for token in ("w0", "w150", "w227"):
            self.assertIn(token, covered)

    def test_lone_short_segment_is_kept_standalone(self) -> None:
        spans = [PageSpan(page=1, text="Tabela 3 valores", bbox=(0, 0, 1, 1))]
        chunks = canonicalize_spans(
            self.document, spans, target_words=220, overlap_words=40, minimum_words=12
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Tabela 3 valores")

    def test_minimum_words_cannot_exceed_target_words(self) -> None:
        spans = [PageSpan(page=1, text="qualquer texto", bbox=(0, 0, 1, 1))]
        with self.assertRaises(ValueError):
            canonicalize_spans(
                self.document, spans, target_words=50, overlap_words=10, minimum_words=100
            )

    def test_default_processor_minimum_words_is_no_longer_disabled(self) -> None:
        # (audit fix) PDFProcessor used to hardcode minimum_words=1 at its
        # canonicalize_spans call site, silently disabling the size floor for
        # every ingested document regardless of the function's own default.
        processor = PDFProcessor.__new__(PDFProcessor)  # bypass I/O in __init__
        self.assertEqual(getattr(processor, "minimum_words", None), None)
        real_processor = PDFProcessor(
            output_dir=tempfile.mkdtemp(), image_dir=tempfile.mkdtemp()
        )
        self.assertEqual(real_processor.minimum_words, 12)


class NoiseFilterTests(unittest.TestCase):
    """F6: _is_index_or_tabular_noise must keep legitimate numeric technical
    facts while still rejecting index/TOC/section-numbering noise.

    The real-sentence fixtures below are the exact chunk text extracted by
    this audit's E2E test from the user's own MQ-5 and ESP32-C6-WROOM-1
    datasheets (jarvis_localhost/tests fixtures do not embed the PDFs
    themselves, only the plain-text spans they produced), so this is a
    regression test against real production failure modes, not only
    hypothetical ones.
    """

    # Real chunk (chk_70f20f66fd746437ae3e1aba561306de, mq5.pdf p1). Before
    # the fix, its raw digit ratio (0.0843) was just over the old 0.08
    # cutoff, so this sentence -- the ONLY place the MQ-5's heating
    # consumption spec appears in the corpus -- was silently discarded as
    # "noise" on every query (audit E2E test, question Q7).
    REAL_MQ5_HEATING_SENTENCE = (
        "Vc Circuit voltage 5V±0.1 AC OR DC V Heating voltage 5V±0.1 "
        "ACOR DC H P Load resistance 20K Ω L R Heater resistance Room Tem "
        "31 10% ± H P Heating consumption less than 800mw H B."
    )

    # Real chunk (chk_a0bb2b3dc043246286fe1e4578af0058 vicinity, ESP32-C6
    # datasheet p28): an ordinary sentence with one embedded spec, already
    # correctly kept before this fix (raw ratio 0.0149) -- must stay kept.
    REAL_ESP32_CLOCK_SENTENCE = (
        "By default, the SPI flash on the module operates at a maximum "
        "clock frequency of 80 MHz and does not support the auto suspend "
        "feature."
    )

    # Real chunk (chk_33c5e02d25389e1f5518006786629547, ESP32-C6 datasheet
    # p4): a part-number/flash-size/dimensions table flattened into running
    # text. This is the source of the "8 MB" fact the audit's Q8 needed.
    # It stays filtered even after the fix -- not because of digit density,
    # but because of the independent, unchanged comma+dash>5 rule (13
    # dashes from the temperature range "-40 85" and the "18.0 x 19.2 x
    # 3.2" dimensions): a genuinely flattened multi-column table row, not a
    # sentence. This is a deliberate, documented limitation (see
    # findings.md, F6), not an oversight -- loosening the comma+dash rule
    # to admit it would reopen false negatives for real index/range noise
    # such as "Table 3.1, Table 3.2, 12-15, 22-28, 45-52".
    REAL_ESP32_FLASH_TABLE_ROW = (
        "Size 2,3 Part Number Flash (°C) (mm) ESP32-C6-WROOM-1U-N4 4 MB "
        "(Quad SPI) ESP32-C6-WROOM-1U-N8 8 MB (Quad SPI) –40 85 18.0 "
        "× 19.2 × 3.2 ∼ ESP32-C6-WROOM-1U-N16 16 MB (Quad SPI) 2 "
        "For specifications, refer to Section 6.5 Memory Specifications."
    )

    SYNTHETIC_NOISE = {
        "toc_page_list": (
            "1 Introducao 1 2 Trabalhos Relacionados 3 3 Metodologia 8 4 "
            "Resultados 15 5 Conclusao 22 Referencias 24"
        ),
        "bare_page_number_list": "12, 45, 67, 89, 102, 156, 203, 245, 301",
        "table_and_figure_index": (
            "Table 3.1, Table 3.2, Table 3.3, Figure 4.1, Figure 4.2, "
            "12-15, 22-28, 45-52"
        ),
        "loose_section_numbering": (
            "1.1 1.2 1.3 1.4 2.1 2.2 2.3 3.1 3.2 3.3 3.4 3.5 4.1 4.2"
        ),
        "footer_page_run": "23 24 25 26 27 28 29 30 31 32 33 34 35 36",
    }

    def test_real_legitimate_numeric_sentences_are_not_filtered(self) -> None:
        for label, sentence in (
            ("mq5_heating", self.REAL_MQ5_HEATING_SENTENCE),
            ("esp32_clock", self.REAL_ESP32_CLOCK_SENTENCE),
        ):
            with self.subTest(label=label):
                self.assertFalse(_is_index_or_tabular_noise(sentence))

    def test_flattened_table_row_is_still_filtered_as_a_documented_limitation(
        self,
    ) -> None:
        self.assertTrue(_is_index_or_tabular_noise(self.REAL_ESP32_FLASH_TABLE_ROW))

    def test_synthetic_index_and_toc_noise_stays_filtered(self) -> None:
        for label, text in self.SYNTHETIC_NOISE.items():
            with self.subTest(label=label):
                self.assertTrue(_is_index_or_tabular_noise(text))

    def test_bare_number_ratio_ignores_unit_suffixed_digits(self) -> None:
        # "800mw" has letters glued to its digits -> not a bare number.
        self.assertLess(
            _bare_number_digit_ratio(self.REAL_MQ5_HEATING_SENTENCE), 0.03
        )
        # A run of plain page numbers has no unit suffixes -> fully counted.
        self.assertGreater(
            _bare_number_digit_ratio(self.SYNTHETIC_NOISE["footer_page_run"]), 0.6
        )

    def test_extractive_answer_now_surfaces_the_real_heating_spec(self) -> None:
        # End-to-end regression for the audit's Q7 failure: the fact must
        # now reach the rendered extractive answer, not just survive the
        # noise filter in isolation.
        quote = (
            self.REAL_MQ5_HEATING_SENTENCE
            + " Environment condition Symbol Parameter name Technical "
            "condition Remarks"
        )
        citation = Citation(
            evidence_id="E1",
            document_id="doc_mq5",
            document_sha256="c" * 64,
            filename="mq5.pdf",
            page=1,
            section="",
            chunk_id="chk_mq5_heating",
            bbox=(0.0, 0.0, 1.0, 1.0),
            quote=quote,
            score=1.0,
        )
        answer = extractive_answer(
            "What is the maximum heating consumption (power) of the MQ-5 sensor?",
            [citation],
        )
        self.assertIn("800mw", answer.lower())


def _tiny_trainer_setup(
    corpus: str, *, tokenizer_path: Path | None = None
) -> tuple[JarvisTokenizer, JarvisTransformer]:
    """A minimal, fast-to-train tokenizer+model pair for resume tests.

    Mirrors the sizing already used by
    ``test_ai_rag.py::test_trainer_encodes_boundaries_controls_and_caps_warmup``
    so a handful of real gradient steps run near-instantly on CPU.
    """

    tokenizer = JarvisTokenizer(vocab_size=96)
    tokenizer.train(corpus)
    if tokenizer_path is not None:
        tokenizer.save(tokenizer_path)
    model = JarvisTransformer(
        JarvisConfig(
            vocab_size=tokenizer.vocab_actual_size,
            context_len=16,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            dropout=0.0,
        )
    )
    return tokenizer, model


class TrainingResumeTests(unittest.TestCase):
    """(F10, 2026-08-31) A crashed run used to lose all progress since it
    started -- there was no way to persist or restore optimizer state, RNG
    state, or step position, and a graceful cancel's own cleanup deleted the
    entire staging directory outright. These tests exercise a real (tiny)
    crash-and-resume cycle end to end, and the lineage guards that must
    refuse -- never silently approximate -- a resume against a corpus or
    tokenizer that no longer matches what the checkpoint was trained on.
    """

    CORPUS = (
        "O motor aciona o eixo e registra a velocidade de rotação.\n\n"
        "O sensor mede cada pulso e preserva a amostra autorizada."
    )

    def test_checkpoint_writes_and_rotates_trainer_state_sidecar(self) -> None:
        tokenizer, model = _tiny_trainer_setup(self.CORPUS)
        with tempfile.TemporaryDirectory() as directory:
            trainer = JarvisTrainer(
                model,
                tokenizer,
                self.CORPUS,
                TrainConfig(
                    max_steps=5,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=directory,
                ),
            )
            trainer._checkpoint("step_1", 5.0)
            first_state = Path(directory) / "jarvis_step_1.trainer_state.pt"
            self.assertTrue(first_state.exists())

            trainer.step = 2
            trainer._checkpoint("step_2", 4.0)
            second_state = Path(directory) / "jarvis_step_2.trainer_state.pt"
            self.assertTrue(second_state.exists())
            # Only the newest sidecar is kept: the previous one is deleted
            # as soon as the new one is safely on disk, since only the most
            # recent checkpoint of a run is ever useful to resume from.
            self.assertFalse(first_state.exists())

            payload = torch.load(second_state, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format_version"], 1)
            self.assertEqual(payload["step"], 2)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("torch_rng_state", payload)
            self.assertIn("numpy_rng_state", payload)
            self.assertIn("python_rng_state", payload)
            self.assertEqual(payload["corpus_sha256"], trainer.corpus_sha256)
            self.assertEqual(payload["tokenizer_sha256"], tokenizer.fingerprint())

    def test_resume_continues_instead_of_restarting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tokenizer_path = directory_path / "jarvis_tokenizer.json"
            tokenizer, model = _tiny_trainer_setup(
                self.CORPUS, tokenizer_path=tokenizer_path
            )

            run_a = JarvisTrainer(
                model,
                tokenizer,
                self.CORPUS,
                TrainConfig(
                    max_steps=4,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=str(directory_path),
                    log_interval=1,
                ),
            )
            run_a.train()
            self.assertEqual(run_a.step, 3)  # 0-indexed: 4 steps ran, 0..3
            interrupted_tokens_seen = run_a.tokens_seen
            self.assertGreater(interrupted_tokens_seen, 0)

            checkpoint_path = directory_path / "jarvis_final.pt"
            state_path = directory_path / "jarvis_final.trainer_state.pt"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(state_path.exists())

            # --- simulate the crash: a brand new process rebuilds
            # everything from what is on disk, exactly like
            # core.brain._run's resume branch does. ---
            resumed_tokenizer = JarvisTokenizer.load(
                tokenizer_path,
                expected_corpus_sha256=JarvisTokenizer.corpus_digest(self.CORPUS),
            )
            resumed_model = JarvisTransformer.load(
                checkpoint_path,
                expected_corpus_sha256=resumed_tokenizer.corpus_digest(self.CORPUS),
                expected_tokenizer_sha256=resumed_tokenizer.fingerprint(),
            )
            run_b = JarvisTrainer(
                resumed_model,
                resumed_tokenizer,
                self.CORPUS,
                TrainConfig(
                    max_steps=8,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=str(directory_path),
                    log_interval=1,
                ),
            )
            # A freshly built optimizer has no per-parameter state at all --
            # AdamW/DirectMLAdamW only initialize it lazily on the first
            # .step() call. Checking this is empty BEFORE resuming, then
            # non-empty immediately AFTER (and before any new training
            # happens), is a direct, mechanical proof that the restored
            # state genuinely came from the sidecar and not from training
            # having run again.
            self.assertEqual(len(run_b.optimizer.state), 0)

            resumed_step = run_b.apply_resume_state(state_path)
            self.assertEqual(resumed_step, run_a.step + 1)
            self.assertEqual(run_b.tokens_seen, interrupted_tokens_seen)
            self.assertGreater(len(run_b.optimizer.state), 0)
            # History carried over from train_history.json so the eventual
            # loss curve has no gap at the resume point.
            self.assertTrue(any("step" in entry for entry in run_b.history))

            history = run_b.train()
            # train()'s return is the trainer's full cumulative history, by
            # design: it includes the pre-crash entries carried over by
            # apply_resume_state (steps 0..3, from run_a) *and* the newly
            # trained ones (4..7) as one continuous, gap-free record -- not
            # just what this particular call trained.
            recorded_steps = sorted(
                entry["step"] for entry in history if "loss" in entry
            )
            self.assertEqual(recorded_steps, list(range(8)))
            self.assertGreater(run_b.tokens_seen, interrupted_tokens_seen)

    def test_resume_refuses_when_corpus_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tokenizer, model = _tiny_trainer_setup(self.CORPUS)
            trainer = JarvisTrainer(
                model,
                tokenizer,
                self.CORPUS,
                TrainConfig(
                    max_steps=3,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=str(directory_path),
                ),
            )
            trainer._checkpoint("step_2", 1.0)
            state_path = directory_path / "jarvis_step_2.trainer_state.pt"

            # A different trainer, built against a corpus with one extra
            # sentence appended -- exactly what happens in production when a
            # document is added or a ghost document is cleaned up between an
            # interrupted run and the next training start.
            changed_corpus = self.CORPUS + "\n\nUma nova frase foi adicionada."
            other_tokenizer, other_model = _tiny_trainer_setup(changed_corpus)
            other_trainer = JarvisTrainer(
                other_model,
                other_tokenizer,
                changed_corpus,
                TrainConfig(
                    max_steps=3,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=str(directory_path),
                ),
            )
            with self.assertRaisesRegex(ValueError, "corpus digest"):
                other_trainer.apply_resume_state(state_path)
            # Refused, not approximated: nothing about the resume state was
            # applied.
            self.assertEqual(other_trainer.step, 0)
            self.assertEqual(len(other_trainer.optimizer.state), 0)

    def test_resume_refuses_when_tokenizer_digest_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            tokenizer, model = _tiny_trainer_setup(self.CORPUS)
            trainer = JarvisTrainer(
                model,
                tokenizer,
                self.CORPUS,
                TrainConfig(
                    max_steps=3,
                    context_len=16,
                    validation_fraction=0.0,
                    checkpoint_dir=str(directory_path),
                ),
            )
            trainer._checkpoint("step_2", 1.0)
            state_path = directory_path / "jarvis_step_2.trainer_state.pt"

            # Corrupt only the recorded tokenizer digest -- everything else
            # (corpus digest, step, optimizer state) still matches, isolating
            # this one guard clause.
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            payload["tokenizer_sha256"] = "0" * 64
            torch.save(payload, state_path)

            with self.assertRaisesRegex(ValueError, "tokenizer digest"):
                trainer.apply_resume_state(state_path)

    def test_find_resumable_training_matches_current_corpus_and_ignores_stale(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            models_root = Path(directory)

            def _make_staging(name: str, *, corpus_sha256: str, canonical_sha256: str, step: int) -> None:
                staging = models_root / name
                staging.mkdir()
                tokenizer, model = _tiny_trainer_setup(
                    self.CORPUS, tokenizer_path=staging / "jarvis_tokenizer.json"
                )
                trainer = JarvisTrainer(
                    model, tokenizer, self.CORPUS,
                    TrainConfig(max_steps=1000, context_len=16, checkpoint_dir=str(staging)),
                    canonical_corpus_sha256=canonical_sha256,
                )
                trainer.corpus_sha256 = corpus_sha256
                trainer.step = step
                trainer.tokens_seen = 100
                trainer._checkpoint("step_1", loss=1.0)

            current_text_digest = JarvisTokenizer.corpus_digest(self.CORPUS)
            current_canonical_digest = "b" * 64
            _make_staging(
                ".training-stale-1111",
                corpus_sha256="f" * 64,  # a document was added/removed since
                canonical_sha256="e" * 64,
                step=798,
            )
            _make_staging(
                ".training-match-2222",
                corpus_sha256=current_text_digest,
                canonical_sha256=current_canonical_digest,
                step=41,
            )

            match = _find_resumable_training(
                text_digest=current_text_digest,
                canonical_digest=current_canonical_digest,
                models_root=models_root,
            )
            self.assertIsNotNone(match)
            staging_dir, tag, state_path = match
            self.assertEqual(staging_dir.name, ".training-match-2222")
            self.assertEqual(tag, "step_1")
            self.assertTrue(state_path.is_file())

            _make_staging(
                ".training-older-3333",
                corpus_sha256=current_text_digest,
                canonical_sha256=current_canonical_digest,
                step=10,
            )
            # A corrupt higher-step model must not hide an older valid pair.
            (staging_dir / "jarvis_step_1.pt").write_bytes(b"truncated")
            fallback = _find_resumable_training(
                text_digest=current_text_digest,
                canonical_digest=current_canonical_digest,
                models_root=models_root,
            )
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback[0].name, ".training-older-3333")

            # A corpus that matches neither staged run finds nothing --
            # exactly the real situation this audit hit: the step_798
            # checkpoint's corpus no longer matches after the 3 ghost
            # documents were cleaned up, so training must start fresh
            # rather than resume against changed data.
            no_match = _find_resumable_training(
                text_digest="z" * 64,
                canonical_digest="y" * 64,
                models_root=models_root,
            )
            self.assertIsNone(no_match)


if __name__ == "__main__":
    unittest.main()
