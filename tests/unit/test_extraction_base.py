"""Unit tests for extraction base contract and extracted fields model."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from app.extraction.base_extractor import BaseExtractor
from app.extraction.base_extractor import ExtractedFieldsModel
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken


def _build_recognized_line() -> RecognizedLine:
    """Build one synthetic OCR line payload for extractor tests."""
    token = RecognizedToken(
        text="JOHN",
        polygon=[
            (10.0, 8.0),
            (44.0, 8.0),
            (44.0, 28.0),
            (10.0, 28.0),
        ],
        bounding_box=(10.0, 8.0, 34.0, 20.0),
        confidence=0.91,
    )

    return RecognizedLine(
        text="JOHN",
        tokens=[token],
        bounding_box=(10.0, 8.0, 34.0, 20.0),
        confidence=0.91,
    )


class _CountingExtractor(BaseExtractor):
    """Test double that records delegated extraction calls."""

    def __init__(self) -> None:
        self.calls = 0

    def extract_from_lines(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        self.calls += 1
        assert len(ocr_lines) == 1
        return ExtractedFieldsModel(full_name="JOHN")


class _InvalidExtractor(BaseExtractor):
    """Test double returning a wrong extraction payload type."""

    def extract_from_lines(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        del ocr_lines
        return "invalid"  # type: ignore[return-value]


def test_extracted_fields_model_defaults_to_explicit_nulls() -> None:
    """Keep missing extracted fields explicit as null values."""
    fields = ExtractedFieldsModel()

    assert fields.card_number is None
    assert fields.full_name is None
    assert fields.license_class is None


def test_base_extractor_contract_declares_expected_return_type() -> None:
    """Declare extraction contract return type for implementations."""
    extract_hints = get_type_hints(BaseExtractor.extract)
    impl_hints = get_type_hints(BaseExtractor.extract_from_lines)

    assert extract_hints["return"] == ExtractedFieldsModel
    assert impl_hints["return"] == ExtractedFieldsModel


def test_base_extractor_returns_empty_fields_for_empty_ocr_lines() -> None:
    """Return explicit-null fields without calling extractor logic."""
    extractor = _CountingExtractor()

    result = extractor.extract([])

    assert extractor.calls == 0
    assert result.model_dump() == ExtractedFieldsModel().model_dump()


def test_base_extractor_delegates_for_non_empty_ocr_lines() -> None:
    """Delegate extraction when OCR lines are available."""
    extractor = _CountingExtractor()

    result = extractor.extract([_build_recognized_line()])

    assert extractor.calls == 1
    assert result.full_name == "JOHN"


def test_base_extractor_rejects_invalid_model_return() -> None:
    """Reject extractor implementations returning wrong payload types."""
    extractor = _InvalidExtractor()

    with pytest.raises(TypeError, match="ExtractedFieldsModel"):
        extractor.extract([_build_recognized_line()])
