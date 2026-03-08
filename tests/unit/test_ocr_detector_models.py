"""Unit tests for OCR text-region model and detector contract."""

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
from pydantic import ValidationError

from app.ocr.detector import DetectionResult
from app.ocr.detector import TextDetector
from app.ocr.detector import TextRegion


def _build_valid_region_payload() -> dict[str, object]:
    """Build a valid detector region payload for model tests."""
    return {
        "polygon": [
            (10.0, 20.0),
            (210.0, 20.0),
            (210.0, 68.0),
            (10.0, 68.0),
        ],
        "bounding_box": (10.0, 20.0, 200.0, 48.0),
        "confidence": 0.87,
    }


def test_text_region_accepts_valid_payload() -> None:
    """Parse geometry and confidence values for detector output regions."""
    region = TextRegion.model_validate(_build_valid_region_payload())

    assert len(region.polygon) == 4
    assert region.bounding_box == (10.0, 20.0, 200.0, 48.0)
    assert region.confidence == pytest.approx(0.87)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_text_region_rejects_confidence_out_of_range(confidence: float) -> None:
    """Reject region payloads with confidence outside [0.0, 1.0]."""
    payload = _build_valid_region_payload()
    payload["confidence"] = confidence

    with pytest.raises(ValidationError):
        TextRegion.model_validate(payload)


def test_text_region_rejects_invalid_polygon_vertex_count() -> None:
    """Require exactly four vertices for detector quadrilateral geometry."""
    payload = _build_valid_region_payload()
    payload["polygon"] = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    with pytest.raises(ValidationError):
        TextRegion.model_validate(payload)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0.0, 24.0),
        (12.0, 0.0),
        (-1.0, 24.0),
    ],
)
def test_text_region_rejects_non_positive_bbox_dimensions(
    width: float,
    height: float,
) -> None:
    """Reject detector bounding boxes with non-positive width or height."""
    payload = _build_valid_region_payload()
    payload["bounding_box"] = (10.0, 20.0, width, height)

    with pytest.raises(ValidationError):
        TextRegion.model_validate(payload)


def test_text_detector_contract_declares_detection_result_return_type() -> None:
    """Declare detector return typing as list[TextRegion]."""
    type_hints = get_type_hints(TextDetector.detect)

    assert type_hints["return"] == DetectionResult


def test_text_detector_protocol_accepts_valid_detector_implementation() -> None:
    """Allow protocol implementations that return TextRegion results."""

    class _StubDetector:
        def detect(self, aligned_image: np.ndarray) -> DetectionResult:
            return [TextRegion.model_validate(_build_valid_region_payload())]

    detector: TextDetector = _StubDetector()
    aligned_image = np.zeros((32, 64, 3), dtype=np.uint8)

    regions = detector.detect(aligned_image)

    assert isinstance(detector, TextDetector)
    assert len(regions) == 1
    assert isinstance(regions[0], TextRegion)
