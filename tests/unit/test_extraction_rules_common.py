"""Unit tests for shared extraction helpers in rules_common."""

from __future__ import annotations

import pytest

from app.extraction.rules_common import build_synonym_pattern
from app.extraction.rules_common import clamp_confidence
from app.extraction.rules_common import cleanup_text
from app.extraction.rules_common import combine_confidence_scores
from app.extraction.rules_common import estimate_line_confidence
from app.extraction.rules_common import extract_value_after_label
from app.extraction.rules_common import has_synonym
from app.extraction.rules_common import select_best_matching_line
from app.extraction.rules_common import tokenize_text
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken


def _build_line(
    *,
    text: str,
    y: float,
    confidence: float,
    token_confidence: float | None = None,
) -> RecognizedLine:
    """Build one OCR line fixture with a single token."""
    resolved_token_conf = confidence
    if token_confidence is not None:
        resolved_token_conf = token_confidence

    token = RecognizedToken(
        text=text,
        polygon=[
            (10.0, y),
            (110.0, y),
            (110.0, y + 20.0),
            (10.0, y + 20.0),
        ],
        bounding_box=(10.0, y, 100.0, 20.0),
        confidence=resolved_token_conf,
    )

    return RecognizedLine(
        text=text,
        tokens=[token],
        bounding_box=(10.0, y, 100.0, 20.0),
        confidence=confidence,
    )


def test_cleanup_text_normalizes_whitespace_and_nbsp() -> None:
    """Normalize mixed OCR whitespace into single spaces."""
    text = "  РОЖДЕНИЯ\u00a0 \t  ГОР. МОСКВА \n"

    assert cleanup_text(text) == "РОЖДЕНИЯ ГОР. МОСКВА"


def test_tokenize_text_splits_cleaned_text_without_empty_tokens() -> None:
    """Tokenize cleaned OCR text while dropping empty elements."""
    text = "   4c)\t ГИБДД  7701  "

    assert tokenize_text(text) == ["4c)", "ГИБДД", "7701"]


def test_build_synonym_pattern_rejects_empty_synonym_set() -> None:
    """Reject synonym patterns when no valid labels are provided."""
    with pytest.raises(ValueError, match="at least one"):
        build_synonym_pattern(["", "   "])


def test_has_synonym_matches_case_insensitive_labels() -> None:
    """Match labels in line text regardless of input casing."""
    text = "Дата выдачи"
    assert has_synonym(text, ["дата выдачи"])

    text = "ГОР. МОСКВА"
    assert has_synonym(text, ["гор. москва"])


def test_extract_value_after_label_returns_cleaned_tail_value() -> None:
    """Extract value content that follows a known label."""
    text = "Дата рождения   19.10.1969  "

    assert extract_value_after_label(text, ["дата рождения"]) == "19.10.1969"


def test_extract_value_after_label_returns_none_when_no_value_exists() -> None:
    """Return no value when line contains label without trailing value."""
    text = "Дата рождения   "

    assert extract_value_after_label(text, ["дата рождения"]) is None


def test_select_best_matching_line_prefers_higher_confidence() -> None:
    """Select best line candidate among synonym-matching lines."""
    low_conf = _build_line(
        text="Дата рождения   19.10.1969  ",
        y=20.0,
        confidence=0.62,
    )
    high_conf = _build_line(
        text="Дата рождения 01.12.1990",
        y=40.0,
        confidence=0.91,
    )

    selected = select_best_matching_line(
        [low_conf, high_conf],
        ["дата рождения"],
    )

    assert selected is not None
    assert selected.text == "Дата рождения 01.12.1990"


def test_select_best_matching_line_returns_none_when_not_found() -> None:
    """Return no matching line when labels do not appear."""
    lines = [_build_line(text="МЕСТО РОЖДЕНИЯ: ГОР. МОСКВА", y=15.0, confidence=0.9)]

    assert select_best_matching_line(lines, ["дата рождения"]) is None


def test_combine_confidence_scores_handles_empty_and_clamping() -> None:
    """Average confidence scores with bounds-safety behavior."""
    assert combine_confidence_scores([]) == pytest.approx(0.0)
    assert combine_confidence_scores([-0.2, 0.4, 1.4]) == pytest.approx(
        (0.0 + 0.4 + 1.0) / 3.0,
    )


def test_estimate_line_confidence_applies_bonus_and_clamp() -> None:
    """Apply synonym-match bonus without exceeding confidence bound."""
    line = _build_line(
        text="5.  77 07 123456",
        y=12.0,
        confidence=0.98,
        token_confidence=0.99,
    )

    boosted = estimate_line_confidence(line, synonym_matched=True)
    plain = estimate_line_confidence(line, synonym_matched=False)

    assert plain < boosted
    assert clamp_confidence(boosted) == pytest.approx(1.0)
