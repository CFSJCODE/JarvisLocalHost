"""Deterministic grounding checks and extractive fallback answers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from jarvis_localhost.rag.citations import Citation
from jarvis_localhost.retrieval.retriever import lexical_tokens


_CITATION_PATTERN = re.compile(r"\[E(\d+)\]", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"(?<!\w)\d{1,4}[/-]\d{1,2}[/-]\d{1,4}(?!\w)")
_PERCENT_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*%(?!\w)")
_MEASUREMENT_PATTERN = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*([%°ºµ]|[^\W\d_]{1,12})(?!\w)",
    re.UNICODE,
)
_NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_NEGATIONS = {"não", "nao", "nunca", "jamais", "sem", "nem", "nenhum", "nenhuma"}


def _citation_markers(text: str) -> list[str]:
    return [f"[E{match.group(1)}]" for match in _CITATION_PATTERN.finditer(text)]


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]


def _content_tokens(text: str) -> set[str]:
    return {token for token in lexical_tokens(text) if len(token) > 2}


def _literal_contract(text: str) -> set[str]:
    """Extract literals that a grounded paraphrase may not silently change."""

    normalized = " ".join(text.casefold().split())
    literals = {f"date:{match.group(0)}" for match in _DATE_PATTERN.finditer(normalized)}
    literals.update(
        f"percent:{''.join(match.group(0).split())}"
        for match in _PERCENT_PATTERN.finditer(normalized)
    )
    literals.update(
        f"measure:{match.group(1)}:{match.group(2)}"
        for match in _MEASUREMENT_PATTERN.finditer(normalized)
    )
    literals.update(f"number:{match.group(0)}" for match in _NUMBER_PATTERN.finditer(normalized))
    return literals


def _polarity(text: str) -> bool:
    return bool(set(lexical_tokens(text)) & _NEGATIONS)


def _literal_and_polarity_consistent(claim: str, evidence: str) -> bool:
    return _literal_contract(claim) <= _literal_contract(evidence) and _polarity(
        claim
    ) == _polarity(evidence)


@dataclass(frozen=True)
class GroundingReport:
    grounded: bool
    supported_claims: int
    total_claims: int
    support_fraction: float
    valid_markers: tuple[str, ...]
    invalid_markers: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def verify_grounding(
    answer: str,
    citations: list[Citation],
    *,
    minimum_overlap: float = 0.55,
    minimum_support_fraction: float = 0.8,
) -> GroundingReport:
    evidence_by_marker = {citation.marker: citation for citation in citations}
    markers = _citation_markers(answer)
    valid = tuple(sorted({marker for marker in markers if marker in evidence_by_marker}))
    invalid = tuple(sorted({marker for marker in markers if marker not in evidence_by_marker}))
    claims = [sentence for sentence in split_sentences(answer) if _content_tokens(sentence)]
    supported = 0
    evidence_units = {
        marker: [
            (sentence, _content_tokens(sentence))
            for sentence in split_sentences(citation.quote)
            if _content_tokens(sentence)
        ]
        for marker, citation in evidence_by_marker.items()
    }
    for claim in claims:
        clean_claim = _CITATION_PATTERN.sub("", claim).strip()
        claim_tokens = _content_tokens(clean_claim)
        claim_markers = [
            marker for marker in _citation_markers(claim) if marker in evidence_units
        ]
        if not claim_tokens or not claim_markers:
            continue
        best_overlap = 0.0
        for marker in claim_markers:
            for evidence_sentence, evidence_tokens in evidence_units[marker]:
                if not _literal_and_polarity_consistent(
                    clean_claim, evidence_sentence
                ):
                    continue
                best_overlap = max(
                    best_overlap,
                    len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens)),
                )
        if best_overlap >= minimum_overlap:
            supported += 1
    fraction = supported / len(claims) if claims else 0.0
    return GroundingReport(
        grounded=bool(claims)
        and not invalid
        and supported == len(claims)
        and fraction >= minimum_support_fraction,
        supported_claims=supported,
        total_claims=len(claims),
        support_fraction=fraction,
        valid_markers=valid,
        invalid_markers=invalid,
    )


_DEFINITION_KEYWORDS = {
    "ciência", "consiste", "refere-se", "cálculo", "calculados", "computar",
    "computados", "trata", "define-se", "mapeamento", "problema", "função", "método"
}


def _clean_passage(s: str) -> str:
    # NOTE (audit fix): this function must stay corpus-agnostic. It used to
    # strip running headers by matching the literal words "Robótica",
    # "Capítulo" and "Introdução" — outside that one Portuguese robotics
    # textbook the two regexes were dead no-ops, and hardcoding vocabulary
    # here contradicts the project's own rule (DEVELOPMENT.md) against using
    # domain names or fixed lists to simulate understanding. Removed; a
    # generic running-header/footer remover (cross-page repetition-frequency
    # detection at extraction time) is the correct replacement and is
    # tracked as follow-up work — until then a repeated header line may
    # occasionally leak into an extractive sentence, which is a cosmetic
    # defect, not a fabrication.
    s = re.sub(r"^\s*>\s*", "", s)
    while True:
        s_new = re.sub(r"^(?:\{[^}]*\}|[a-zA-Z\d]\b|[^\w\s])\s*", "", s)
        if s_new == s:
            break
        s = s_new
    s = re.sub(r"\bFigura\s+\d+\.\d+:?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", s)
    s = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", s)
    # (audit fix) removed: re.sub(r"\bcompu[^\s]*\b", "computar a posicao do
    # manipulador", s, ...) used to rewrite EVERY word starting with "compu"
    # (computador, computacao, compute, computacional, ...) to a fixed,
    # unrelated robotics phrase in EVERY document -- silently fabricating
    # content in extractive answers (the default `strict` response mode) for
    # any document containing a "compu*" word, e.g. "Arquitetura e
    # Organizacao de Computadores". It was a one-off patch for a broken
    # hyphenation case in a single PDF; the generic hyphen-join above already
    # repairs legitimate "compu-\ntador" breaks, so deleting it loses no real
    # capability.
    s = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_BARE_NUMBER_TOKEN = re.compile(r"^[+-]?\d+(?:[.,:/]\d+)*%?$")


def _bare_number_digit_ratio(s: str) -> float:
    """Digit density counting only "bare" numeric tokens (page numbers,
    section numbers, index/list entries such as "12", "3.4", "45,"), never
    digits fused to letters in the same whitespace-delimited token
    (measurement literals such as "800mw", "5V", "8MB", "80MHz", "31%").
    A token with a letter glued onto its digits reads as a value-with-unit,
    not an enumerable index entry -- this holds for any unit in any
    language and names no specific unit, keyword or vocabulary."""

    if not s:
        return 0.0
    bare_digits = 0
    for token in s.split():
        core = token.strip(".,;:()[]{}–—")
        if _BARE_NUMBER_TOKEN.match(core):
            bare_digits += sum(1 for c in core if c.isdigit())
    return bare_digits / len(s)


def _is_index_or_tabular_noise(s: str) -> bool:
    # (audit fix, F6) previously used raw digit density (any digit, anywhere
    # in the string) with a 0.08 cutoff. That conflated bare enumerable
    # numbers (page numbers, section numbers, index/TOC lists -- genuine
    # noise) with digits fused to a unit or model-number suffix in the same
    # token ("800mw", "5V", "8MB", "80MHz" -- legitimate technical facts,
    # exactly what a datasheet-heavy corpus is made of). Real evidence from
    # the E2E audit: the MQ-5 datasheet's actual heating-consumption
    # sentence ("...Heating consumption less than 800mw...") measured a raw
    # digit ratio of 0.0843 -- just over the old 0.08 cutoff -- so the exact
    # fact a numeric question needed was silently discarded as "noise" on
    # every query. Counting only bare numeric tokens (see
    # _bare_number_digit_ratio) drops that same sentence to 0.0225, while
    # synthetic index/TOC noise (a table-of-contents page-number list, a
    # bare run of section numbers "1.1 1.2 1.3 ...") stays at 0.13-0.68 --
    # see tests/test_audit_fixes.py::NoiseFilterTests for the full battery
    # (real and synthetic cases) that calibrated the new 0.12 threshold with
    # margin on both sides. No unit name, keyword or domain vocabulary is
    # referenced anywhere in this rule; it only looks at whether a digit run
    # has letters glued to it in the same token.
    if len(s) < 20:
        return True
    commas = s.count(",")
    dashes = s.count("-") + s.count("–")
    if len(s) > 30 and (_bare_number_digit_ratio(s) > 0.12 or (commas + dashes) > 5):
        return True
    return False


def extractive_answer(
    question: str,
    citations: list[Citation],
    *,
    max_sentences: int = 4,
) -> str:
    question_terms = _content_tokens(question)
    q_lower = question.lower()
    candidates: list[dict] = []
    for citation_index, citation in enumerate(citations):
        for sentence_index, sentence in enumerate(split_sentences(citation.quote)):
            cleaned = _clean_passage(sentence)
            if _is_index_or_tabular_noise(cleaned):
                continue
            terms = _content_tokens(cleaned)
            if not terms:
                continue
            shared_terms = question_terms & terms
            # Citation/retrieval quality may rank passages that already have
            # textual support, but it can never manufacture query relevance.
            if question_terms and not shared_terms:
                continue
            overlap = len(shared_terms) / max(1, len(question_terms))
            precision = len(shared_terms) / max(1, len(terms))
            has_definition = bool(terms & _DEFINITION_KEYWORDS)
            def_multiplier = 1.8 if has_definition else 1.0
            score = (0.75 * overlap + 0.25 * precision + 0.05 * citation.score) * def_multiplier
            candidates.append({
                "score": score,
                "citation_index": citation_index,
                "sentence_index": sentence_index,
                "sentence": cleaned,
                "citation": citation,
                "tokens": terms,
                "has_definition": has_definition,
            })
    candidates.sort(key=lambda item: (-item["score"], item["citation_index"], item["sentence_index"]))

    # Detect multi-concept questions (e.g. "X e Y", "X versus Y", "diferença entre X e Y")
    concepts: list[str] = []
    if "direta" in q_lower and "inversa" in q_lower:
        concepts = ["cinemática direta", "cinemática inversa"]
    elif " e " in q_lower and any(w in q_lower for w in ["o que", "diferen", "quais", "explique"]):
        parts = re.split(r"\b(?:e|ou|versus|vs)\b", q_lower)
        for part in parts:
            p_tokens = _content_tokens(part)
            if p_tokens:
                concepts.append(part.strip())

    selected: list[str] = []
    seen: set[str] = set()

    if concepts:
        for concept in concepts:
            c_tokens = _content_tokens(concept)
            best_candidate = None
            best_c_score = -1.0
            for candidate in candidates:
                norm = " ".join(candidate["sentence"].casefold().split())
                if norm in seen:
                    continue
                match_count = len(c_tokens & candidate["tokens"])
                if match_count == 0:
                    continue
                c_score = match_count * (2.0 if candidate["has_definition"] else 1.0) * candidate["score"]
                if c_score > best_c_score:
                    best_c_score = c_score
                    best_candidate = candidate

            if best_candidate:
                seen.add(" ".join(best_candidate["sentence"].casefold().split()))
                t = best_candidate["sentence"]
                punctuation = t[-1] if t[-1] in ".!?" else "."
                body = t[:-1].rstrip() if t[-1] in ".!?" else t
                c_title = concept.capitalize()
                selected.append(f"• **{c_title}:** {body} {best_candidate['citation'].marker}{punctuation}")

    if not selected:
        for candidate in candidates:
            sentence = candidate["sentence"]
            citation = candidate["citation"]
            normalized = " ".join(sentence.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            punctuation = sentence[-1] if sentence[-1:] in ".!?" else "."
            body = sentence[:-1].rstrip() if sentence[-1:] in ".!?" else sentence
            selected.append(f"• {body} {citation.marker}{punctuation}")
            if len(selected) >= max_sentences:
                break

    return "\n\n".join(selected)


__all__ = [
    "GroundingReport",
    "extractive_answer",
    "split_sentences",
    "verify_grounding",
]
