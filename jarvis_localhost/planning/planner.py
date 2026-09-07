"""Hierarchical response planner producing structured argument blueprints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class QueryIntent(str, Enum):
    CONCEPTUAL_DEFINITION = "conceptual_definition"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    PROCEDURAL_METHOD = "procedural_method"
    MATHEMATICAL_FORMULATION = "mathematical_formulation"
    GENERAL_TECHNICAL = "general_technical"


@dataclass(frozen=True)
class OutlineSection:
    title: str
    target_role: str
    description: str
    key_terms: tuple[str, ...]


@dataclass(frozen=True)
class ResponseOutline:
    intent: QueryIntent
    topic: str
    subtopics: tuple[str, ...]
    sections: tuple[OutlineSection, ...]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "topic": self.topic,
            "subtopics": list(self.subtopics),
            "sections": [
                {
                    "title": s.title,
                    "target_role": s.target_role,
                    "description": s.description,
                    "key_terms": list(s.key_terms),
                }
                for s in self.sections
            ],
        }


class ResponsePlanner:
    """Plans the logical outline of an answer before generating grounded sections."""

    def __init__(self) -> None:
        self._comparative_re = re.compile(
            r"\b(?:diferen[cç]a\s+entre|compare|versus|vs|distin[cç][aã]o|em\s+compara[cç][aã]o|o\s+que\s+[eé]\s+[\w\s]+\s+e\s+[\w\s]+)\b",
            re.IGNORECASE,
        )
        self._definition_re = re.compile(
            r"\b(?:o\s+que\s+[eé]|qual\s+a\s+defini[cç][aã]o|o\s+que\s+significa|conceito\s+de)\b",
            re.IGNORECASE,
        )
        self._procedural_re = re.compile(
            r"\b(?:como\s+calcular|como\s+determinar|qual\s+o\s+m[eé]todo|passo\s+a\s+passo|como\s+funciona|procedimento)\b",
            re.IGNORECASE,
        )
        self._math_re = re.compile(
            r"\b(?:equa[cç][aã]o|matriz|formula[cç][aã]o|derivada|integral|modelo\s+matem[aá]tico)\b",
            re.IGNORECASE,
        )

    def plan(self, query: str) -> ResponseOutline:
        q_clean = query.strip()

        # 1. Check Procedural Intent
        if self._procedural_re.search(q_clean):
            return ResponseOutline(
                intent=QueryIntent.PROCEDURAL_METHOD,
                topic=q_clean,
                subtopics=(),
                sections=(
                    OutlineSection(
                        title="Visão Geral do Método",
                        target_role="explanation",
                        description="Princípio fundamental do algoritmo/procedimento",
                        key_terms=(),
                    ),
                    OutlineSection(
                        title="Etapas e Formulação",
                        target_role="procedure",
                        description="Passos de cálculo e transformações",
                        key_terms=(),
                    ),
                ),
            )

        # 2. Check Math Formulation
        if self._math_re.search(q_clean):
            return ResponseOutline(
                intent=QueryIntent.MATHEMATICAL_FORMULATION,
                topic=q_clean,
                subtopics=(),
                sections=(
                    OutlineSection(
                        title="Formulação Matemática",
                        target_role="mathematical_formulation",
                        description="Definição das matrizes, equações e variáveis",
                        key_terms=(),
                    ),
                    OutlineSection(
                        title="Interpretação Física",
                        target_role="explanation",
                        description="Significado físico das componentes matemáticas",
                        key_terms=(),
                    ),
                ),
            )

        # 3. Check Comparative Intent
        if self._comparative_re.search(q_clean) and any(sep in q_clean.lower() for sep in [" e ", " versus ", " vs ", " entre "]):
            parts = re.split(r"\b(?:e|versus|vs|entre)\b", q_clean, flags=re.IGNORECASE)
            subtopics = tuple(
                re.sub(r"^(?:o\s+que\s+[eé]|qual\s+a\s+diferen[cç]a\s+entre|compare|conceito\s+de)\s+", "", p.strip(), flags=re.IGNORECASE).strip(" ?.,")
                for p in parts if len(p.strip()) > 2
            )
            sections = []
            for sub in subtopics:
                sections.append(
                    OutlineSection(
                        title=sub.capitalize(),
                        target_role="definition",
                        description=f"Definição formal e características fundamentais de {sub}",
                        key_terms=(sub,),
                    )
                )
            if len(subtopics) >= 2:
                sections.append(
                    OutlineSection(
                        title="Distinção Operacional",
                        target_role="comparison",
                        description=f"Comparação direta e diferenças entre {' e '.join(subtopics)}",
                        key_terms=subtopics,
                    )
                )
            return ResponseOutline(
                intent=QueryIntent.COMPARATIVE_ANALYSIS,
                topic=q_clean,
                subtopics=subtopics,
                sections=tuple(sections),
            )

        # Default Definition
        return ResponseOutline(
            intent=QueryIntent.CONCEPTUAL_DEFINITION,
            topic=q_clean,
            subtopics=(),
            sections=(
                OutlineSection(
                    title="Definição Formal",
                    target_role="definition",
                    description="Conceito e escopo teórico",
                    key_terms=(),
                ),
                OutlineSection(
                    title="Detalhamento Técnico",
                    target_role="explanation",
                    description="Fundamentação operacional e propriedades",
                    key_terms=(),
                ),
            ),
        )
