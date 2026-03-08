"""Unit tests for OCR recognizer models and recognizer contract."""

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
from pydantic import ValidationError

from app.ocr.detector import DetectionResult
from app.ocr.detector import TextRegion
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken
from app.ocr.recognizer import TextRecognizer
from app.ocr.recognizer import TokenRecognitionResult


def _build_valid_region_payload() -> dict[str, object]:
    """Build a valid detector region payload for recognizer tests."""
    return {
        "polygon": [
            (12.0, 10.0),
            (82.0, 10.0),
            (82.0, 30.0),
            (12.0, 30.0),
        ],
        "bounding_box": (12.0, 10.0, 70.0, 20.0),
        "confidence": 0.87,
    }


def _build_valid_token_payload() -> dict[str, object]:
    """Build a valid recognized-token payload for model tests."""
    return {
        "text": "МОСКВА",
        "polygon": [
            (12.0, 10.0),
            (82.0, 10.0),
            (82.0, 30.0),
            (12.0, 30.0),
        ],
        "bounding_box": (12.0, 10.0, 70.0, 20.0),
        "confidence": 0.93,
    }


def _build_valid_line_payload() -> dict[str, object]:
    """Build a valid recognized-line payload with one token."""
    return {
        "text": "МОСКВА",
        "tokens": [RecognizedToken.model_validate(_build_valid_token_payload())],
        "bounding_box": (12.0, 10.0, 70.0, 20.0),
        "confidence": 0.92,
    }


def test_recognized_token_accepts_valid_payload() -> None:
    """Parse recognized-token geometry, text, and confidence values."""
    token = RecognizedToken.model_validate(_build_valid_token_payload())

    assert token.text == "МОСКВА"
    assert len(token.polygon) == 4
    assert token.bounding_box == (12.0, 10.0, 70.0, 20.0)
    assert token.confidence == pytest.approx(0.93)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_recognized_token_rejects_confidence_out_of_range(
    confidence: float,
) -> None:
    """Reject token confidence values outside [0.0, 1.0]."""
    payload = _build_valid_token_payload()
    payload["confidence"] = confidence

    with pytest.raises(ValidationError):
        RecognizedToken.model_validate(payload)


def test_recognized_token_rejects_empty_text() -> None:
    """Require non-empty text values in recognized token payloads."""
    payload = _build_valid_token_payload()
    payload["text"] = ""

    with pytest.raises(ValidationError):
        RecognizedToken.model_validate(payload)


def test_recognized_token_rejects_invalid_polygon_vertex_count() -> None:
    """Require exactly four vertices for recognized token polygons."""
    payload = _build_valid_token_payload()
    payload["polygon"] = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    with pytest.raises(ValidationError):
        RecognizedToken.model_validate(payload)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0.0, 20.0),
        (70.0, 0.0),
        (-1.0, 20.0),
    ],
)
def test_recognized_token_rejects_non_positive_bbox_dimensions(
    width: float,
    height: float,
) -> None:
    """Reject recognized token boxes with non-positive dimensions."""
    payload = _build_valid_token_payload()
    payload["bounding_box"] = (12.0, 10.0, width, height)

    with pytest.raises(ValidationError):
        RecognizedToken.model_validate(payload)


def test_recognized_line_accepts_valid_payload() -> None:
    """Parse recognized-line payload with nested token values."""
    line = RecognizedLine.model_validate(_build_valid_line_payload())

    assert line.text == "МОСКВА"
    assert len(line.tokens) == 1
    assert isinstance(line.tokens[0], RecognizedToken)
    assert line.confidence == pytest.approx(0.92)


def test_recognized_line_rejects_empty_tokens() -> None:
    """Require at least one token in each recognized line payload."""
    payload = _build_valid_line_payload()
    payload["tokens"] = []

    with pytest.raises(ValidationError):
        RecognizedLine.model_validate(payload)


def test_text_recognizer_contract_declares_expected_return_types() -> None:
    """Declare recognizer return typings for tokens and lines."""
    recognize_hints = get_type_hints(TextRecognizer.recognize)
    line_hints = get_type_hints(TextRecognizer.group_tokens_to_lines)

    assert recognize_hints["return"] == TokenRecognitionResult
    assert line_hints["return"] == LineRecognitionResult


def test_text_recognizer_protocol_accepts_valid_implementation() -> None:
    """Allow protocol implementations for recognize and grouping methods."""

    class _StubRecognizer:
        def recognize(
            self,
            aligned_image: np.ndarray,
            regions: DetectionResult,
        ) -> TokenRecognitionResult:
            assert aligned_image.size > 0
            assert len(regions) == 1
            return [RecognizedToken.model_validate(_build_valid_token_payload())]

        def group_tokens_to_lines(
            self,
            tokens: TokenRecognitionResult,
        ) -> LineRecognitionResult:
            assert len(tokens) == 1
            return [RecognizedLine.model_validate(_build_valid_line_payload())]

    recognizer: TextRecognizer = _StubRecognizer()
    aligned_image = np.zeros((32, 64, 3), dtype=np.uint8)
    regions = [TextRegion.model_validate(_build_valid_region_payload())]

    tokens = recognizer.recognize(aligned_image, regions)
    lines = recognizer.group_tokens_to_lines(tokens)

    assert isinstance(recognizer, TextRecognizer)
    assert len(tokens) == 1
    assert isinstance(tokens[0], RecognizedToken)
    assert len(lines) == 1
    assert isinstance(lines[0], RecognizedLine)
