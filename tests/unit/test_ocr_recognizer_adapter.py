"""Unit smoke test for local OCR recognizer adapter integration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ocr.detector import DetectionResult
from app.ocr.detector import TextRegion
from app.ocr.recognizer import EasyOCRTextRecognizer
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _StubEasyOCRReader:
    """Test double that mimics EasyOCR Reader.readtext output."""

    def __init__(self) -> None:
        self.calls = 0

    def readtext(
        self,
        image: np.ndarray,
        *,
        detail: int = 1,
        paragraph: bool = False,
    ) -> list[tuple[list[list[float]], str, float]]:
        self.calls += 1

        assert image.size > 0
        assert detail == 1
        assert paragraph is False

        return [
            (
                [
                    [12.0, 10.0],
                    [82.0, 10.0],
                    [82.0, 30.0],
                    [12.0, 30.0],
                ],
                "МОСКВА",
                0.93,
            )
        ]


def _build_token(
    *,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: float,
) -> RecognizedToken:
    """Build synthetic token payloads for deterministic grouping tests."""
    return RecognizedToken(
        text=text,
        polygon=[
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ],
        bounding_box=(x, y, width, height),
        confidence=confidence,
    )


def test_easyocr_recognizer_returns_tokens_with_text_on_fixture() -> None:
    """Return recognized tokens with text from fixture image and regions."""
    fixture_path = PROJECT_ROOT / "images/id_cards/passport_min.png"
    aligned_image = cv2.imread(str(fixture_path))

    assert aligned_image is not None

    regions: DetectionResult = [
        TextRegion(
            polygon=[
                (12.0, 10.0),
                (82.0, 10.0),
                (82.0, 30.0),
                (12.0, 30.0),
            ],
            bounding_box=(12.0, 10.0, 70.0, 20.0),
            confidence=0.87,
        )
    ]

    reader = _StubEasyOCRReader()
    recognizer = EasyOCRTextRecognizer(reader=reader)

    tokens = recognizer.recognize(aligned_image, regions)

    assert reader.calls == 1
    assert len(tokens) == 1
    assert isinstance(tokens[0], RecognizedToken)
    assert tokens[0].text == "МОСКВА"
    assert tokens[0].confidence == pytest.approx(0.93)
    assert tokens[0].bounding_box == pytest.approx((12.0, 10.0, 70.0, 20.0))


def test_group_tokens_to_lines_is_deterministic_with_unsorted_tokens() -> None:
    """Assemble lines in stable top-to-bottom and left-to-right order."""
    recognizer = EasyOCRTextRecognizer(reader=_StubEasyOCRReader())
    tokens = [
        _build_token(
            text="WORLD",
            x=60.0,
            y=10.0,
            width=40.0,
            height=20.0,
            confidence=0.70,
        ),
        _build_token(
            text="AGENT",
            x=50.0,
            y=42.0,
            width=45.0,
            height=20.0,
            confidence=0.60,
        ),
        _build_token(
            text="HELLO",
            x=10.0,
            y=10.0,
            width=45.0,
            height=20.0,
            confidence=0.90,
        ),
        _build_token(
            text="COPILOT",
            x=10.0,
            y=40.0,
            width=35.0,
            height=20.0,
            confidence=0.80,
        ),
    ]

    lines = recognizer.group_tokens_to_lines(tokens)

    assert len(lines) == 2
    assert isinstance(lines[0], RecognizedLine)
    assert lines[0].text == "HELLO WORLD"
    assert [token.text for token in lines[0].tokens] == ["HELLO", "WORLD"]
    assert lines[0].bounding_box == pytest.approx((10.0, 10.0, 90.0, 20.0))
    assert lines[0].confidence == pytest.approx(0.80)

    assert lines[1].text == "COPILOT AGENT"
    assert [token.text for token in lines[1].tokens] == ["COPILOT", "AGENT"]
    assert lines[1].bounding_box == pytest.approx((10.0, 40.0, 85.0, 22.0))
    assert lines[1].confidence == pytest.approx(0.70)


def test_group_tokens_to_lines_groups_nearby_y_offsets() -> None:
    """Group close vertical offsets into one line for stable extraction."""
    recognizer = EasyOCRTextRecognizer(reader=_StubEasyOCRReader())
    tokens = [
        _build_token(
            text="ID",
            x=10.0,
            y=11.0,
            width=18.0,
            height=18.0,
            confidence=0.90,
        ),
        _build_token(
            text="CARD",
            x=34.0,
            y=14.0,
            width=38.0,
            height=18.0,
            confidence=0.82,
        ),
    ]

    lines = recognizer.group_tokens_to_lines(tokens)

    assert len(lines) == 1
    assert lines[0].text == "ID CARD"
    assert lines[0].bounding_box == pytest.approx((10.0, 11.0, 62.0, 21.0))
    assert lines[0].confidence == pytest.approx(0.86)


def test_group_tokens_to_lines_returns_empty_for_empty_input() -> None:
    """Return no lines when recognizer receives no OCR tokens."""
    recognizer = EasyOCRTextRecognizer(reader=_StubEasyOCRReader())

    assert recognizer.group_tokens_to_lines([]) == []
