"""Fixture tests for perspective correction output geometry."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.preprocessing.document_preprocessor import DocumentPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _expected_output_size(corners: list[tuple[int, int]]) -> tuple[int, int]:
    """Compute canonical output size from ordered boundary corners."""
    points = np.asarray(corners, dtype=np.float32)
    top_left, top_right, bottom_right, bottom_left = points

    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)

    target_width = max(1, int(round(max(top_width, bottom_width))))
    target_height = max(1, int(round(max(left_height, right_height))))

    return target_width, target_height


@pytest.mark.parametrize(
    "relative_path",
    [
        "images/bank_cards/bank-cards.jpg",
        "images/id_cards/yQAAAgNof2A-960.jpg",
        "images/drivers_licenses/1385315836_1578903606.jpg",
    ],
)
def test_apply_perspective_correction_returns_expected_geometry(
    relative_path: str,
) -> None:
    """Warp fixture documents into canonical rectangular output geometry."""
    preprocessor = DocumentPreprocessor()
    image_path = PROJECT_ROOT / relative_path

    image = cv2.imread(str(image_path))
    assert image is not None

    boundary = preprocessor.detect_document_boundary(image)
    aligned = preprocessor.apply_perspective_correction(image, boundary)

    expected_width, expected_height = _expected_output_size(boundary.corners)

    assert aligned.ndim == image.ndim
    assert aligned.dtype == image.dtype
    assert aligned.shape[1] == expected_width
    assert aligned.shape[0] == expected_height
    assert (aligned.shape[1] >= aligned.shape[0]) == (
        expected_width >= expected_height
    )
