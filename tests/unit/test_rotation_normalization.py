"""Unit tests for post-alignment rotation normalization."""

import cv2
import numpy as np
import pytest

from app.preprocessing.document_preprocessor import DocumentPreprocessor


def _build_upright_fixture() -> np.ndarray:
    """Build a deterministic upright document-like fixture image."""
    image = np.full((200, 340, 3), 245, dtype=np.uint8)

    cv2.rectangle(image, (10, 10), (329, 189), (30, 30, 30), 2)
    cv2.rectangle(image, (26, 24), (314, 52), (35, 35, 35), -1)
    cv2.putText(
        image,
        "HEADER",
        (34, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    line_y = 80
    for text in ("Name: Jane Doe", "ID: A1234567", "DOB: 1990-01-01"):
        cv2.putText(
            image,
            text,
            (30, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        line_y += 34

    return image


def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by arbitrary angle while preserving full content."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    abs_cos = abs(transform[0, 0])
    abs_sin = abs(transform[0, 1])
    bound_width = int(round((height * abs_sin) + (width * abs_cos)))
    bound_height = int(round((height * abs_cos) + (width * abs_sin)))

    transform[0, 2] += (bound_width / 2.0) - center[0]
    transform[1, 2] += (bound_height / 2.0) - center[1]

    border_value: int | tuple[int, int, int]
    if image.ndim == 2:
        border_value = 255
    else:
        border_value = (255, 255, 255)

    return cv2.warpAffine(
        image,
        transform,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _orientation_signature(image: np.ndarray) -> tuple[float, float, float]:
    """Return top/bottom intensity and row/column projection ratio."""
    grayscale = image
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grayscale_float = grayscale.astype(np.float32)
    band_height = max(1, int(round(grayscale.shape[0] * 0.30)))
    top_mean = float(np.mean(grayscale_float[:band_height, :]))
    bottom_mean = float(np.mean(grayscale_float[-band_height:, :]))

    _, threshold = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    row_projection = threshold.sum(axis=1).astype(np.float32)
    col_projection = threshold.sum(axis=0).astype(np.float32)
    projection_ratio = float(
        np.var(row_projection) / (np.var(col_projection) + 1e-6)
    )

    return top_mean, bottom_mean, projection_ratio


@pytest.mark.parametrize(
    "angle_deg",
    [0.0, 90.0, 180.0, 270.0, 17.0, -28.0, 123.0],
)
def test_normalize_rotation_handles_arbitrary_angles(
    angle_deg: float,
) -> None:
    """Normalize arbitrary-angle rotated fixtures to an upright orientation."""
    preprocessor = DocumentPreprocessor()
    upright = _build_upright_fixture()
    rotated = _rotate_bound(upright, angle_deg)

    normalized = preprocessor.normalize_rotation(rotated)

    top_mean, bottom_mean, projection_ratio = _orientation_signature(
        normalized
    )

    assert normalized.dtype == rotated.dtype
    assert normalized.shape[1] >= normalized.shape[0]
    assert top_mean < bottom_mean
    assert projection_ratio > 1.05


def test_normalize_rotation_supports_grayscale_inputs() -> None:
    """Preserve grayscale outputs while restoring upright orientation."""
    preprocessor = DocumentPreprocessor()

    upright_bgr = _build_upright_fixture()
    upright_gray = cv2.cvtColor(upright_bgr, cv2.COLOR_BGR2GRAY)
    rotated_gray = _rotate_bound(upright_gray, -22.0)

    normalized_gray = preprocessor.normalize_rotation(rotated_gray)

    top_mean, bottom_mean, projection_ratio = _orientation_signature(
        normalized_gray
    )

    assert normalized_gray.ndim == 2
    assert normalized_gray.dtype == upright_gray.dtype
    assert normalized_gray.shape[1] >= normalized_gray.shape[0]
    assert top_mean < bottom_mean
    assert projection_ratio > 1.05
