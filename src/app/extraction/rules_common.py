"""Shared extraction rule helpers for OCR text matching and scoring."""

from __future__ import annotations

import re
from collections.abc import Sequence
from re import Pattern

from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine

_WHITESPACE_PATTERN = re.compile(r"\s+")
_TOKEN_PATTERN = re.compile(r"\S+")


def cleanup_text(text: str) -> str:
    """Normalize OCR text by collapsing whitespace and trimming edges."""
    normalized = text.replace("\u00a0", " ")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def tokenize_text(text: str) -> list[str]:
    """Split cleaned OCR text into non-empty tokens."""
    cleaned = cleanup_text(text)
    if not cleaned:
        return []

    return _TOKEN_PATTERN.findall(cleaned)


def build_synonym_pattern(synonyms: Sequence[str]) -> Pattern[str]:
    """Build a case-insensitive regex pattern for label synonyms."""
    prepared_synonyms = _prepare_synonyms(synonyms)
    parts: list[str] = []

    for synonym in sorted(prepared_synonyms, key=len, reverse=True):
        escaped = re.escape(synonym)
        if _contains_word_characters(synonym):
            parts.append(rf"(?<!\w){escaped}(?!\w)")
        else:
            parts.append(escaped)

    pattern = "|".join(parts)
    return re.compile(pattern, flags=re.IGNORECASE)


def has_synonym(text: str, synonyms: Sequence[str]) -> bool:
    """Return whether text contains at least one configured synonym."""
    cleaned = cleanup_text(text)
    if not cleaned:
        return False

    try:
        pattern = build_synonym_pattern(synonyms)
    except ValueError:
        return False

    return pattern.search(cleaned) is not None


def extract_value_after_label(
    text: str,
    synonyms: Sequence[str],
) -> str | None:
    """Extract the value tail that appears after a matched label."""
    cleaned = cleanup_text(text)
    if not cleaned:
        return None

    try:
        pattern = build_synonym_pattern(synonyms)
    except ValueError:
        return None

    match = pattern.search(cleaned)
    if match is None:
        return None

    tail = cleaned[match.end() :].strip()
    while tail.startswith(".") or tail.startswith(")") or tail.startswith("-"):
        tail = tail[1:].strip()

    if not tail:
        return None

    return tail


def clamp_confidence(score: float) -> float:
    """Clamp confidence scores into the inclusive [0.0, 1.0] range."""
    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 1.0
    return float(score)


def combine_confidence_scores(scores: Sequence[float]) -> float:
    """Combine confidence values using bounded arithmetic mean."""
    if not scores:
        return 0.0

    clamped = [clamp_confidence(score) for score in scores]
    return sum(clamped) / len(clamped)


def estimate_line_confidence(
    line: RecognizedLine,
    *,
    synonym_matched: bool = False,
) -> float:
    """Estimate line quality from line and token confidence signals."""
    scores = [line.confidence]
    scores.extend(token.confidence for token in line.tokens)

    estimated = combine_confidence_scores(scores)
    if synonym_matched:
        estimated += 0.05

    return clamp_confidence(estimated)


def select_best_matching_line(
    lines: LineRecognitionResult,
    synonyms: Sequence[str],
) -> RecognizedLine | None:
    """Select best synonym-matching line using confidence-first ordering."""
    best_line: RecognizedLine | None = None
    best_score = -1.0
    best_position = (0.0, 0.0)

    for line in lines:
        if not has_synonym(line.text, synonyms):
            continue

        score = estimate_line_confidence(line, synonym_matched=True)
        position = (line.bounding_box[1], line.bounding_box[0])

        should_replace = False
        if best_line is None:
            should_replace = True
        elif score > best_score:
            should_replace = True
        elif score == best_score and position < best_position:
            should_replace = True

        if should_replace:
            best_line = line
            best_score = score
            best_position = position

    return best_line


def _prepare_synonyms(synonyms: Sequence[str]) -> list[str]:
    """Validate and normalize label synonyms before regex generation."""
    prepared: list[str] = []
    seen: set[str] = set()

    for synonym in synonyms:
        cleaned = cleanup_text(synonym)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        prepared.append(cleaned)

    if not prepared:
        raise ValueError("Synonym set must contain at least one non-empty label.")

    return prepared


def _contains_word_characters(value: str) -> bool:
    """Return whether value includes at least one alphanumeric symbol."""
    return any(character.isalnum() for character in value)


__all__ = [
    "build_synonym_pattern",
    "clamp_confidence",
    "cleanup_text",
    "combine_confidence_scores",
    "estimate_line_confidence",
    "extract_value_after_label",
    "has_synonym",
    "select_best_matching_line",
    "tokenize_text",
]
