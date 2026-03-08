"""Fixture tests for document boundary detection."""

from pathlib import Path

import cv2
import pytest

from app.preprocessing.document_preprocessor import DocumentPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_path", "min_area_ratio"),
    [
        ("images/bank_cards/pr-card.png", 0.10),
        ("images/id_cards/passport_min.png", 0.05),
        ("images/drivers_licenses/voditelskoe-udostoverenie.jpg", 0.10),
    ],
)
def test_detect_document_boundary_on_representative_fixtures(
    relative_path: str,
    min_area_ratio: float,
) -> None:
    """Detect four boundary corners and valid bounds on fixture images."""
    preprocessor = DocumentPreprocessor()
    image_path = PROJECT_ROOT / relative_path

    image = cv2.imread(str(image_path))
    assert image is not None

    result = preprocessor.detect_document_boundary(image)

    assert len(result.corners) == 4
    assert result.area_ratio >= min_area_ratio

    height, width = image.shape[:2]
    for x_coord, y_coord in result.corners:
        assert 0 <= x_coord < width
        assert 0 <= y_coord < height

    x_min, y_min, box_width, box_height = result.bounds
    assert 0 <= x_min < width
    assert 0 <= y_min < height
    assert box_width > 0
    assert box_height > 0
    assert x_min + box_width <= width
    assert y_min + box_height <= height
