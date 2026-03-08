"""Unit smoke test for local OCR detector adapter integration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ocr.detector import EasyOCRTextDetector
from app.ocr.detector import TextRegion

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
                "SERIES",
                0.93,
            )
        ]


def test_easyocr_detector_returns_regions_with_confidence_on_fixture() -> None:
    """Return text regions with confidence from a fixture image input."""
    fixture_path = PROJECT_ROOT / "images/id_cards/passport_min.png"
    aligned_image = cv2.imread(str(fixture_path))

    assert aligned_image is not None

    reader = _StubEasyOCRReader()
    detector = EasyOCRTextDetector(reader=reader)

    regions = detector.detect(aligned_image)

    assert reader.calls == 1
    assert len(regions) == 1
    assert isinstance(regions[0], TextRegion)
    assert regions[0].confidence == pytest.approx(0.93)
    assert regions[0].bounding_box == pytest.approx((12.0, 10.0, 70.0, 20.0))
