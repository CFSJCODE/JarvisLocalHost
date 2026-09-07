"""Deterministic syntactic and structural profile extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_MATH_SYMBOLS_RE = re.compile(r"[\=\+\-\*\/\^\_\{\}\\\<\>\≤\≥\±\∑\∏\∫\√\α\β\γ\θ\λ\μ\π\σ\ω\τ]")
_LIST_ITEM_RE = re.compile(r"(?:^|\n)\s*(?:[•\-\*]|\d+[\.\)])\s+")


@dataclass(frozen=True)
class SyntacticProfile:
    sentence_count: int
    mean_sentence_length: float
    std_sentence_length: float
    min_sentence_length: int
    max_sentence_length: int
    colon_density: float
    semicolon_density: float
    comma_density: float
    dash_density: float
    paren_density: float
    question_density: float
    math_density: float
    list_density: float

    def to_dict(self) -> dict:
        return {
            "sentence_count": self.sentence_count,
            "mean_sentence_length": round(self.mean_sentence_length, 2),
            "std_sentence_length": round(self.std_sentence_length, 2),
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "colon_density": round(self.colon_density, 4),
            "semicolon_density": round(self.semicolon_density, 4),
            "comma_density": round(self.comma_density, 4),
            "dash_density": round(self.dash_density, 4),
            "paren_density": round(self.paren_density, 4),
            "question_density": round(self.question_density, 4),
            "math_density": round(self.math_density, 4),
            "list_density": round(self.list_density, 4),
        }


def analyze_syntax(text: str) -> SyntacticProfile:
    """Analyze punctuation distribution, sentence lengths, and structural densities."""
    raw_sentences = [
        s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()
    ]
    sentence_count = len(raw_sentences)

    if sentence_count == 0:
        return SyntacticProfile(
            sentence_count=0,
            mean_sentence_length=0.0,
            std_sentence_length=0.0,
            min_sentence_length=0,
            max_sentence_length=0,
            colon_density=0.0,
            semicolon_density=0.0,
            comma_density=0.0,
            dash_density=0.0,
            paren_density=0.0,
            question_density=0.0,
            math_density=0.0,
            list_density=0.0,
        )

    # Word lengths per sentence
    sentence_word_lengths = [len(s.split()) for s in raw_sentences]
    mean_len = sum(sentence_word_lengths) / sentence_count
    var_len = (
        sum((l - mean_len) ** 2 for l in sentence_word_lengths) / sentence_count
    )
    std_len = math.sqrt(var_len)

    total_chars = max(1, len(text))
    total_words = max(1, sum(sentence_word_lengths))

    # Punctuation counts
    colon_count = text.count(":")
    semicolon_count = text.count(";")
    comma_count = text.count(",")
    dash_count = text.count("-") + text.count("–") + text.count("—")
    paren_count = text.count("(") + text.count(")")
    question_count = text.count("?")

    # Math and list metrics
    math_symbols = len(_MATH_SYMBOLS_RE.findall(text))
    list_items = len(_LIST_ITEM_RE.findall(text))

    return SyntacticProfile(
        sentence_count=sentence_count,
        mean_sentence_length=float(mean_len),
        std_sentence_length=float(std_len),
        min_sentence_length=min(sentence_word_lengths),
        max_sentence_length=max(sentence_word_lengths),
        colon_density=colon_count / total_words,
        semicolon_density=semicolon_count / total_words,
        comma_density=comma_count / total_words,
        dash_density=dash_count / total_words,
        paren_density=paren_count / total_words,
        question_density=question_count / total_words,
        math_density=math_symbols / total_words,
        list_density=list_items / sentence_count,
    )
