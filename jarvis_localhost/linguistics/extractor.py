"""Unified linguistic feature extraction and multi-vector generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from jarvis_localhost.linguistics.discourse import DiscourseProfile, analyze_discourse
from jarvis_localhost.linguistics.lexical import LexicalProfile, analyze_lexical
from jarvis_localhost.linguistics.rhetoric import (
    RhetoricalProfile,
    RhetoricalRole,
    classify_rhetoric,
)
from jarvis_localhost.linguistics.syntax import SyntacticProfile, analyze_syntax

STYLE_VECTOR_DIM = 32
RHETORICAL_VECTOR_DIM = 16


@dataclass(frozen=True)
class LinguisticProfile:
    lexical: LexicalProfile
    syntactic: SyntacticProfile
    discourse: DiscourseProfile
    rhetorical: RhetoricalProfile
    style_vector: tuple[float, ...]
    rhetorical_vector: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "lexical": self.lexical.to_dict(),
            "syntactic": self.syntactic.to_dict(),
            "discourse": self.discourse.to_dict(),
            "rhetorical": self.rhetorical.to_dict(),
            "style_vector": [round(x, 4) for x in self.style_vector],
            "rhetorical_vector": [round(x, 4) for x in self.rhetorical_vector],
        }


def _build_style_vector(
    lex: LexicalProfile, syn: SyntacticProfile, disc: DiscourseProfile
) -> np.ndarray:
    """Build a fixed 32-dimensional normalized style feature vector."""
    vec = np.zeros(STYLE_VECTOR_DIM, dtype=np.float32)

    # 0-7: Lexical density features
    vec[0] = min(1.0, lex.ttr)
    vec[1] = min(1.0, lex.hapax_ratio)
    vec[2] = min(1.0, lex.yule_k / 250.0)
    vec[3] = min(1.0, lex.mean_word_length / 12.0)
    vec[4] = min(1.0, lex.technical_term_count / max(1, lex.token_count))
    vec[5] = min(1.0, lex.acronym_count / max(1, lex.token_count))
    vec[6] = min(1.0, lex.token_count / 300.0)
    vec[7] = min(1.0, lex.root_ttr / 10.0)

    # 8-19: Syntactic and punctuation features
    vec[8] = min(1.0, syn.mean_sentence_length / 40.0)
    vec[9] = min(1.0, syn.std_sentence_length / 20.0)
    vec[10] = min(1.0, syn.sentence_count / 10.0)
    vec[11] = min(1.0, syn.colon_density * 20.0)
    vec[12] = min(1.0, syn.semicolon_density * 30.0)
    vec[13] = min(1.0, syn.comma_density * 10.0)
    vec[14] = min(1.0, syn.dash_density * 15.0)
    vec[15] = min(1.0, syn.paren_density * 15.0)
    vec[16] = min(1.0, syn.question_density * 10.0)
    vec[17] = min(1.0, syn.math_density * 5.0)
    vec[18] = min(1.0, syn.list_density)
    vec[19] = min(1.0, (syn.max_sentence_length - syn.min_sentence_length) / 50.0)

    # 20-27: Discourse connective features
    vec[20] = min(1.0, disc.connective_density * 20.0)
    vec[21] = min(1.0, disc.additive_count / 4.0)
    vec[22] = min(1.0, disc.adversative_count / 4.0)
    vec[23] = min(1.0, disc.consecutive_count / 4.0)
    vec[24] = min(1.0, disc.sequential_count / 4.0)
    vec[25] = min(1.0, disc.exemplificative_count / 4.0)
    vec[26] = float(len(disc.transition_flow)) / 8.0
    vec[27] = 1.0 if disc.primary_discourse_function != "neutral_informative" else 0.0

    # 28-31: Reserved for structural balance
    vec[28] = (vec[0] + vec[8]) / 2.0  # Complexity composite
    vec[29] = (vec[17] + vec[4]) / 2.0  # Technical density composite
    vec[30] = (vec[11] + vec[18]) / 2.0 # Explanatory structure composite
    vec[31] = (vec[20] + vec[13]) / 2.0 # Transition richness composite

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _build_rhetorical_vector(rhet: RhetoricalProfile) -> np.ndarray:
    """Build a fixed 16-dimensional normalized rhetorical feature vector."""
    vec = np.zeros(RHETORICAL_VECTOR_DIM, dtype=np.float32)
    roles = list(RhetoricalRole)

    for i, role in enumerate(roles):
        score = rhet.role_scores.get(role.value, 0.0)
        vec[i] = score

    vec[len(roles)] = rhet.confidence
    vec[len(roles) + 1] = 1.0 if rhet.primary_role == RhetoricalRole.DEFINITION else 0.0
    vec[len(roles) + 2] = 1.0 if rhet.primary_role == RhetoricalRole.COMPARISON else 0.0
    vec[len(roles) + 3] = 1.0 if rhet.primary_role == RhetoricalRole.PROCEDURE else 0.0
    vec[len(roles) + 4] = 1.0 if rhet.primary_role == RhetoricalRole.MATHEMATICAL_FORMULATION else 0.0

    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


class LinguisticExtractor:
    """Unified extractor producing linguistic profiles and orthogonal feature vectors."""

    def extract(self, text: str) -> LinguisticProfile:
        lex = analyze_lexical(text)
        syn = analyze_syntax(text)
        disc = analyze_discourse(text)
        rhet = classify_rhetoric(text)

        style_v = _build_style_vector(lex, syn, disc)
        rhet_v = _build_rhetorical_vector(rhet)

        return LinguisticProfile(
            lexical=lex,
            syntactic=syn,
            discourse=disc,
            rhetorical=rhet,
            style_vector=tuple(float(x) for x in style_v),
            rhetorical_vector=tuple(float(x) for x in rhet_v),
        )
