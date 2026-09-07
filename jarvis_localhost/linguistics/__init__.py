"""Linguistic analysis and multi-vector style/rhetoric extraction."""

from __future__ import annotations

from jarvis_localhost.linguistics.discourse import DiscourseProfile, analyze_discourse
from jarvis_localhost.linguistics.extractor import (
    RHETORICAL_VECTOR_DIM,
    STYLE_VECTOR_DIM,
    LinguisticExtractor,
    LinguisticProfile,
)
from jarvis_localhost.linguistics.lexical import LexicalProfile, analyze_lexical
from jarvis_localhost.linguistics.rhetoric import (
    RhetoricalProfile,
    RhetoricalRole,
    classify_rhetoric,
)
from jarvis_localhost.linguistics.syntax import SyntacticProfile, analyze_syntax

__all__ = [
    "RHETORICAL_VECTOR_DIM",
    "STYLE_VECTOR_DIM",
    "DiscourseProfile",
    "LexicalProfile",
    "LinguisticExtractor",
    "LinguisticProfile",
    "RhetoricalProfile",
    "RhetoricalRole",
    "SyntacticProfile",
    "analyze_discourse",
    "analyze_lexical",
    "analyze_syntax",
    "classify_rhetoric",
]
