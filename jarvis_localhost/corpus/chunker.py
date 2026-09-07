"""Span-aware deterministic chunking with page, section and bbox fidelity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .provenance import CanonicalChunk, DocumentIdentity


@dataclass(frozen=True)
class PageSpan:
    page: int
    text: str
    bbox: Sequence[float]
    is_heading: bool = False
    extraction_method: str = "text_layer"
    content_type: str = "text"
    source_ref: str = ""


@dataclass(frozen=True)
class _Segment:
    page: int
    section: str
    extraction_method: str
    content_type: str
    source_ref: str
    spans: tuple[PageSpan, ...]


def _bbox_union(spans: Sequence[PageSpan]) -> tuple[float, float, float, float]:
    boxes = [span.bbox for span in spans if span.bbox and len(span.bbox) == 4]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _segments_for_page(
    page: int,
    spans: Sequence[PageSpan],
    active_section: str,
) -> tuple[list[_Segment], str]:
    """Split at headings and extraction boundaries before word windows.

    A chunk therefore cannot inherit a heading that appears after its content,
    nor silently mix a PDF text layer with a table or optional OCR output.
    """

    segments: list[_Segment] = []
    current: list[PageSpan] = []
    current_key: tuple[str, str, str, str] | None = None

    def flush() -> None:
        nonlocal current, current_key
        if current and current_key is not None:
            section, extraction_method, content_type, source_ref = current_key
            segments.append(
                _Segment(
                    page=page,
                    section=section,
                    extraction_method=extraction_method,
                    content_type=content_type,
                    source_ref=source_ref,
                    spans=tuple(current),
                )
            )
        current = []
        current_key = None

    for span in spans:
        if span.is_heading and span.content_type == "text":
            flush()
            active_section = span.text
        key = (
            active_section,
            span.extraction_method,
            span.content_type,
            span.source_ref,
        )
        if current_key is not None and key != current_key:
            flush()
        current_key = key
        current.append(span)
    flush()
    return segments, active_section


def canonicalize_spans(
    document: DocumentIdentity,
    spans: Iterable[PageSpan],
    *,
    target_words: int = 220,
    overlap_words: int = 40,
    minimum_words: int = 12,
) -> List[CanonicalChunk]:
    if target_words < 16:
        raise ValueError("target_words must be at least 16")
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words must be between zero and target_words - 1")
    if minimum_words < 1:
        raise ValueError("minimum_words must be positive")
    if minimum_words > target_words:
        raise ValueError("minimum_words cannot exceed target_words")

    by_page: dict[int, list[PageSpan]] = defaultdict(list)
    for span in spans:
        if isinstance(span.page, bool) or not isinstance(span.page, int) or span.page < 1:
            raise ValueError("span page numbers must be one-based positive integers")
        cleaned = " ".join(span.text.split())
        if cleaned:
            by_page[span.page].append(
                PageSpan(
                    page=span.page,
                    text=cleaned,
                    bbox=span.bbox,
                    is_heading=span.is_heading,
                    extraction_method=span.extraction_method,
                    content_type=span.content_type,
                    source_ref=span.source_ref,
                )
            )

    chunks: list[CanonicalChunk] = []
    active_section = ""
    ordinal = 0
    step = target_words - overlap_words
    for page in sorted(by_page):
        segments, active_section = _segments_for_page(
            page, by_page[page], active_section
        )

        # Pass 1: slide the target/overlap window within each segment and
        # merge any resulting window under ``minimum_words`` into a neighbour
        # from the *same* segment (a lone heading fragment or a short tail
        # left over at a window boundary carries little standalone retrieval
        # signal). A segment that produces exactly one window is left as-is
        # here even if short: there is nothing in-segment to merge it with.
        segment_windows: list[tuple[_Segment, list[tuple[list[str], list[PageSpan]]]]] = []
        for segment in segments:
            words: list[str] = []
            word_spans: list[PageSpan] = []
            for span in segment.spans:
                span_words = span.text.split()
                words.extend(span_words)
                word_spans.extend([span] * len(span_words))
            if not words:
                continue

            windows: list[tuple[list[str], list[PageSpan]]] = []
            for start in range(0, len(words), step):
                # A final tail that consists only of already-covered overlap
                # adds no source content.  All other short units are retained.
                if start and len(words) - start <= overlap_words:
                    break
                end = min(start + target_words, len(words))
                selected_words = words[start:end]
                selected_spans: list[PageSpan] = []
                seen_span_objects: set[int] = set()
                for selected in word_spans[start:end]:
                    marker = id(selected)
                    if marker not in seen_span_objects:
                        seen_span_objects.add(marker)
                        selected_spans.append(selected)
                windows.append((selected_words, selected_spans))
                if end >= len(words):
                    break

            merged: list[tuple[list[str], list[PageSpan]]] = []
            for window_words, window_spans in windows:
                if len(window_words) < minimum_words and merged:
                    previous_words, previous_spans = merged[-1]
                    previous_words.extend(window_words)
                    seen = {id(span) for span in previous_spans}
                    previous_spans.extend(
                        span for span in window_spans if id(span) not in seen
                    )
                else:
                    merged.append((list(window_words), list(window_spans)))
            if len(merged) > 1 and len(merged[0][0]) < minimum_words:
                first_words, first_spans = merged.pop(0)
                next_words, next_spans = merged[0]
                seen = {id(span) for span in first_spans}
                merged[0] = (
                    [*first_words, *next_words],
                    [*first_spans, *(span for span in next_spans if id(span) not in seen)],
                )
            segment_windows.append((segment, merged))

        # Pass 2: a segment whose *entire* content is still below
        # ``minimum_words`` after pass 1 is, in practice, almost always a
        # lone heading/section-number line (e.g. a table of contents entry
        # like "1.1") that ``_segments_for_page`` had to flush as its own
        # segment purely because another heading immediately follows it --
        # not a deliberate standalone unit. Reuniting it with whatever comes
        # next on the same page is more faithful to the source than indexing
        # a permanent 1-3 word orphan chunk, so it is carried forward into
        # the next segment -- as long as that segment shares the same
        # extraction method/content type/source ref, so text still never
        # merges into a table or OCR span. (Two adjacent segments always
        # differ in ``section`` by construction -- otherwise
        # ``_segments_for_page`` would already have kept them as one -- so
        # this check intentionally allows crossing a heading/section
        # boundary, which is exactly the boundary manufacturing the orphans;
        # the merged chunk keeps the *receiving* segment's section label.)
        carry: tuple[list[str], list[PageSpan]] | None = None
        for index, (segment, windows_list) in enumerate(segment_windows):
            windows_list = list(windows_list)
            if carry is not None:
                first_words, first_spans = windows_list[0]
                seen = {id(span) for span in carry[1]}
                windows_list[0] = (
                    [*carry[0], *first_words],
                    [*carry[1], *(span for span in first_spans if id(span) not in seen)],
                )
                carry = None
            has_next = index + 1 < len(segment_windows)
            if len(windows_list) == 1 and len(windows_list[0][0]) < minimum_words and has_next:
                next_segment = segment_windows[index + 1][0]
                if (
                    next_segment.extraction_method == segment.extraction_method
                    and next_segment.content_type == segment.content_type
                    and next_segment.source_ref == segment.source_ref
                ):
                    carry = windows_list[0]
                    continue
            for window_words, window_spans in windows_list:
                chunks.append(
                    CanonicalChunk.build(
                        document,
                        page=segment.page,
                        section=segment.section,
                        bbox=_bbox_union(window_spans),
                        text=" ".join(window_words),
                        ordinal=ordinal,
                        extraction_method=segment.extraction_method,
                        content_type=segment.content_type,
                        source_ref=segment.source_ref,
                    )
                )
                ordinal += 1
    return chunks
