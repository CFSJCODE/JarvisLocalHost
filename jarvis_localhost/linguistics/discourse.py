"""Discourse marker extraction and logical transition analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

_CONNECTIVES: dict[str, tuple[str, ...]] = {
    "additive": (
        "além disso", "ademais", "outrossim", "também", "bem como",
        "por sua vez", "paralelamente", "da mesma forma", "igualmente"
    ),
    "adversative": (
        "entretanto", "no entanto", "por outro lado", "em contraste",
        "todavia", "contudo", "não obstante", "ao contrário", "embora"
    ),
    "consecutive": (
        "portanto", "consequentemente", "logo", "dessa forma", "assim",
        "por conseguinte", "como resultado", "em decorrência", "por isso"
    ),
    "sequential": (
        "primeiramente", "em primeiro lugar", "em seguida", "posteriormente",
        "finalmente", "por fim", "inicialmente", "logo após"
    ),
    "exemplificative": (
        "por exemplo", "isto é", "a saber", "como no caso",
        "a título de exemplo", "exemplificando", "em particular"
    ),
}

# Compile case-insensitive word boundary patterns
_COMPILED_CONNECTIVES: dict[str, list[re.Pattern]] = {
    category: [
        re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for phrase in phrases
    ]
    for category, phrases in _CONNECTIVES.items()
}


@dataclass(frozen=True)
class DiscourseProfile:
    total_connective_count: int
    connective_density: float
    additive_count: int
    adversative_count: int
    consecutive_count: int
    sequential_count: int
    exemplificative_count: int
    primary_discourse_function: str
    transition_flow: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "total_connective_count": self.total_connective_count,
            "connective_density": round(self.connective_density, 4),
            "additive_count": self.additive_count,
            "adversative_count": self.adversative_count,
            "consecutive_count": self.consecutive_count,
            "sequential_count": self.sequential_count,
            "exemplificative_count": self.exemplificative_count,
            "primary_discourse_function": self.primary_discourse_function,
            "transition_flow": list(self.transition_flow),
        }


def analyze_discourse(text: str) -> DiscourseProfile:
    """Analyze discourse transitions, connectives and structural logical flow."""
    words = text.split()
    total_words = max(1, len(words))

    matches_by_category: dict[str, list[tuple[int, str]]] = {
        cat: [] for cat in _CONNECTIVES
    }

    all_matches: list[tuple[int, str, str]] = []  # (char_index, category, matched_text)

    for category, patterns in _COMPILED_CONNECTIVES.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                pos = match.start()
                matched_str = match.group(0)
                matches_by_category[category].append((pos, matched_str))
                all_matches.append((pos, category, matched_str))

    all_matches.sort(key=lambda item: item[0])
    total_connectives = len(all_matches)

    # Determine primary function
    category_counts = {cat: len(m) for cat, m in matches_by_category.items()}
    if total_connectives == 0:
        primary = "neutral_informative"
    else:
        primary = max(category_counts.items(), key=lambda item: item[1])[0]

    transition_flow = tuple(item[1] for item in all_matches[:8])

    return DiscourseProfile(
        total_connective_count=total_connectives,
        connective_density=total_connectives / total_words,
        additive_count=category_counts["additive"],
        adversative_count=category_counts["adversative"],
        consecutive_count=category_counts["consecutive"],
        sequential_count=category_counts["sequential"],
        exemplificative_count=category_counts["exemplificative"],
        primary_discourse_function=primary,
        transition_flow=transition_flow,
    )
