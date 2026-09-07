"""Deterministic lexical and terminological analysis for corpus units."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

_TOKEN_RE = re.compile(r"\b[a-zA-ZÀ-ÖØ-öø-ÿ0-9_]+\b")
_ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,8}\b")
_TECHNICAL_TERM_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:-[a-z]+)+|\d+[a-zA-Z]+|[a-zA-Z]+\d+)\b"
)


@dataclass(frozen=True)
class LexicalProfile:
    token_count: int
    unique_token_count: int
    ttr: float
    root_ttr: float
    yule_k: float
    hapax_ratio: float
    mean_word_length: float
    technical_term_count: int
    acronym_count: int
    top_terms: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict:
        return {
            "token_count": self.token_count,
            "unique_token_count": self.unique_token_count,
            "ttr": round(self.ttr, 4),
            "root_ttr": round(self.root_ttr, 4),
            "yule_k": round(self.yule_k, 2),
            "hapax_ratio": round(self.hapax_ratio, 4),
            "mean_word_length": round(self.mean_word_length, 2),
            "technical_term_count": self.technical_term_count,
            "acronym_count": self.acronym_count,
            "top_terms": list(self.top_terms),
        }


def analyze_lexical(text: str) -> LexicalProfile:
    """Extract deterministic lexical statistics from text without external models."""
    raw_tokens = _TOKEN_RE.findall(text)
    token_count = len(raw_tokens)
    if token_count == 0:
        return LexicalProfile(
            token_count=0,
            unique_token_count=0,
            ttr=0.0,
            root_ttr=0.0,
            yule_k=0.0,
            hapax_ratio=0.0,
            mean_word_length=0.0,
            technical_term_count=0,
            acronym_count=0,
            top_terms=(),
        )

    lowered_tokens = [t.lower() for t in raw_tokens]
    counts = Counter(lowered_tokens)
    unique_count = len(counts)

    # Type-Token Ratio
    ttr = unique_count / token_count
    root_ttr = unique_count / math.sqrt(token_count)

    # Yule's K Characteristic
    # K = 10^4 * (sum(i^2 * V_i) - N) / N^2
    spectrum = Counter(counts.values())
    sum_spectrum = sum((freq**2) * count for freq, count in spectrum.items())
    yule_k = (
        10000.0 * (sum_spectrum - token_count) / (token_count**2)
        if token_count > 1
        else 0.0
    )

    # Hapax Legomena (terms occurring only once)
    hapax_count = spectrum.get(1, 0)
    hapax_ratio = hapax_count / unique_count if unique_count > 0 else 0.0

    # Mean word length
    total_chars = sum(len(t) for t in raw_tokens)
    mean_len = total_chars / token_count

    # Technical terms and acronyms
    acronyms = _ACRONYM_RE.findall(text)
    technical_terms = _TECHNICAL_TERM_RE.findall(text)

    # Top content terms (filtering short stop-like tokens)
    content_terms = [
        (term, count)
        for term, count in counts.most_common(10)
        if len(term) > 2 and not term.isdigit()
    ]

    return LexicalProfile(
        token_count=token_count,
        unique_token_count=unique_count,
        ttr=float(ttr),
        root_ttr=float(root_ttr),
        yule_k=float(max(0.0, yule_k)),
        hapax_ratio=float(hapax_ratio),
        mean_word_length=float(mean_len),
        technical_term_count=len(technical_terms),
        acronym_count=len(acronyms),
        top_terms=tuple(content_terms[:8]),
    )
