"""Unit smoke test for local OCR recognizer adapter integration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ocr.detector import DetectionResult
from app.ocr.detector import TextRegion
from app.ocr.recognizer import EasyOCRTextRecognizer
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
