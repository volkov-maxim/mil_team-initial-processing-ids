"""Unit tests for denoise and contrast normalization preprocessing."""

import cv2
import numpy as np

from app.preprocessing.document_preprocessor import DocumentPreprocessor


def _build_low_contrast_noisy_fixture() -> np.ndarray:
    """Build a deterministic low-contrast fixture with synthetic noise."""
    image = np.full((200, 340, 3), 132, dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (327, 187), (122, 122, 122), 2)
    cv2.putText(
        image,
        "NAME: JANE DOE",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (126, 126, 126),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "ID: A1234567",
        (24, 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (126, 126, 126),
        2,
        cv2.LINE_AA,
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(loc=0.0, scale=8.0, size=image.shape)
    noisy = image.astype(np.int16) + noise.astype(np.int16)

    return np.clip(noisy, 0, 255).astype(np.uint8)


def _grayscale_stddev(image: np.ndarray) -> float:
    """Compute grayscale intensity standard deviation for contrast checks."""
    grayscale = image
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(np.std(grayscale))


def test_denoise_contrast_normalization_transforms_fixture() -> None:
    """Apply denoise+contrast to produce a transformed OCR-friendly image."""
    preprocessor = DocumentPreprocessor()
    fixture = _build_low_contrast_noisy_fixture()

    transformed = preprocessor.denoise_contrast(fixture)

    assert transformed.dtype == fixture.dtype
    assert transformed.shape == fixture.shape
    assert not np.array_equal(transformed, fixture)


def test_contrast_normalization_improves_low_contrast_fixture() -> None:
    """Increase grayscale contrast when contrast normalization is enabled."""
    preprocessor = DocumentPreprocessor()
    fixture = _build_low_contrast_noisy_fixture()

    transformed = preprocessor.denoise_contrast(
        fixture,
        apply_denoise=False,
        apply_contrast_normalization=True,
    )

    assert transformed.dtype == fixture.dtype
    assert transformed.shape == fixture.shape
    assert _grayscale_stddev(transformed) > _grayscale_stddev(fixture)


def test_denoise_contrast_normalization_can_be_disabled() -> None:
    """Return an unchanged image when both optional transforms are disabled."""
    preprocessor = DocumentPreprocessor()
    fixture = _build_low_contrast_noisy_fixture()

    transformed = preprocessor.denoise_contrast(
        fixture,
        apply_denoise=False,
        apply_contrast_normalization=False,
    )

    assert transformed.dtype == fixture.dtype
    assert transformed.shape == fixture.shape
    assert np.array_equal(transformed, fixture)
    assert transformed is not fixture


def test_denoise_contrast_normalization_handles_small_grayscale_inputs() -> None:
    """Avoid crashing when denoise/contrast is applied to small grayscale data."""
    preprocessor = DocumentPreprocessor()
    image = np.full((12, 18), 126, dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (15, 9), 134, 1)

    transformed = preprocessor.denoise_contrast(image)

    assert transformed.ndim == 2
    assert transformed.dtype == image.dtype
    assert transformed.shape == image.shape
