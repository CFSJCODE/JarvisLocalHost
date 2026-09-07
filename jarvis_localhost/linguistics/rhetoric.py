"""Rhetorical role classification for academic and technical text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

class RhetoricalRole(str, Enum):
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    MATHEMATICAL_FORMULATION = "mathematical_formulation"
    EXAMPLE = "example"
    CONCLUSION = "conclusion"
    EVIDENCE = "evidence"


_RHETORICAL_INDICATORS: dict[RhetoricalRole, tuple[str, ...]] = {
    RhetoricalRole.DEFINITION: (
        "é a ciência que", "consiste em", "define-se", "é definido como",
        "denomina-se", "entende-se por", "é o processo de", "trata-se de",
        "refere-se a", "é o estudo", "chamamos de", "conhecido como"
    ),
    RhetoricalRole.COMPARISON: (
        "em contraste", "ao contrário de", "por outro lado", "versus",
        "em comparação", "a diferença entre", "enquanto o", "diverge de",
        "vantagem em relação", "desvantagem"
    ),
    RhetoricalRole.PROCEDURE: (
        "para calcular", "o procedimento consiste", "o algoritmo", "o método",
        "passo a passo", "em primeiro lugar", "em seguida", "determina-se",
        "executa-se", "obter a solução", "implementação"
    ),
    RhetoricalRole.MATHEMATICAL_FORMULATION: (
        "dada a equação", "matriz de transformação", "o vetor", "a derivada",
        "calculando a integral", "a formulação", "o sistema linear",
        "substituindo na equação", "coordenadas homogêneas"
    ),
    RhetoricalRole.EXAMPLE: (
        "por exemplo", "como exemplo", "considere o", "a título de exemplo",
        "no caso do manipulador", "como ilustrado", "caso prático"
    ),
    RhetoricalRole.CONCLUSION: (
        "em conclusão", "conclui-se", "em resumo", "portanto",
        "finalmente", "em suma", "resumindo os resultados"
    ),
    RhetoricalRole.EVIDENCE: (
        "como observado na tabela", "os dados mostram", "verificou-se que",
        "experimentalmente", "as medições indicam", "o gráfico"
    ),
    RhetoricalRole.EXPLANATION: (
        "isso ocorre porque", "a razão fundamental", "o motivo",
        "funciona através", "permite que", "possibilita", "opera segundo"
    ),
}

_COMPILED_RHETORICAL: dict[RhetoricalRole, list[re.Pattern]] = {
    role: [
        re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for phrase in phrases
    ]
    for role, phrases in _RHETORICAL_INDICATORS.items()
}


@dataclass(frozen=True)
class RhetoricalProfile:
    primary_role: RhetoricalRole
    confidence: float
    role_scores: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "primary_role": self.primary_role.value,
            "confidence": round(self.confidence, 4),
            "role_scores": {k: round(v, 4) for k, v in self.role_scores.items()},
        }


def classify_rhetoric(text: str) -> RhetoricalProfile:
    """Classify the rhetorical function of a paragraph or chunk deterministically."""
    scores: dict[RhetoricalRole, float] = {role: 0.0 for role in RhetoricalRole}

    for role, patterns in _COMPILED_RHETORICAL.items():
        match_count = sum(len(pat.findall(text)) for pat in patterns)
        scores[role] = float(match_count)

    # Heuristic priors based on structure:
    # Math density booster
    math_symbols = len(re.findall(r"[\=\+\-\*\/\^\_\{\}\\\<\>\≤\≥\±\∑\∏\∫\√\α\β\γ\θ\λ\μ\π\σ\ω\τ]", text))
    if math_symbols > 3:
        scores[RhetoricalRole.MATHEMATICAL_FORMULATION] += 1.5 + (math_symbols * 0.1)

    total_matches = sum(scores.values())
    if total_matches == 0:
        # Default informative explanation
        return RhetoricalProfile(
            primary_role=RhetoricalRole.EXPLANATION,
            confidence=0.5,
            role_scores={r.value: 0.125 for r in RhetoricalRole},
        )

    # Normalize distribution
    norm_scores = {r.value: s / total_matches for r, s in scores.items()}
    best_role = max(scores.items(), key=lambda item: item[1])[0]
    best_confidence = norm_scores[best_role.value]

    return RhetoricalProfile(
        primary_role=best_role,
        confidence=best_confidence,
        role_scores=norm_scores,
    )
