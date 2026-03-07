"""Unit tests for preprocessing type/size validation."""

import cv2
import numpy as np

from app.preprocessing.document_preprocessor import DocumentPreprocessor


def _encode_png(image: np.ndarray) -> bytes:
    """Encode a uint8 image into PNG bytes for test payloads."""
    success, encoded = cv2.imencode(".png", image)
    assert success is True
    return encoded.tobytes()


def _build_document_like_png_bytes() -> bytes:
    """Build a deterministic document-like image payload."""
    image = np.full((160, 256, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (246, 150), (15, 15, 15), 2)
    cv2.putText(
        image,
        "NAME: JOHN DOE",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "ID: A1234567",
        (18, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return _encode_png(image)


def _build_blank_png_bytes() -> bytes:
    """Build a nearly uniform image payload for blank-image checks."""
    image = np.full((160, 256, 3), 240, dtype=np.uint8)
    return _encode_png(image)


def _build_tiny_png_bytes() -> bytes:
    """Build a tiny image payload that should fail readability checks."""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (21, 21), (255, 255, 255), 1)
    return _encode_png(image)


def test_validate_type_size_readability_accepts_allowed_type_and_size() -> None:
    """Accept payloads with supported media type and valid byte size."""
    payload = _build_document_like_png_bytes()
    preprocessor = DocumentPreprocessor(max_file_size_bytes=len(payload) + 16)

    outcome = preprocessor.validate_type_size_readability(
        payload,
        content_type="image/png",
    )

    assert outcome.is_valid is True
    assert outcome.failure_reason is None
    assert outcome.message is None
    assert outcome.details["content_type"] == "image/png"
    assert outcome.details["file_size_bytes"] == len(payload)


def test_validate_type_size_readability_rejects_unsupported_media_type() -> None:
    """Block unsupported payload content types with typed failure details."""
    payload = _build_document_like_png_bytes()
    preprocessor = DocumentPreprocessor(max_file_size_bytes=len(payload) + 16)

    outcome = preprocessor.validate_type_size_readability(
        payload,
        content_type="application/pdf",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "unsupported_media_type"
    assert outcome.message == "Unsupported media type."
    assert outcome.details["content_type"] == "application/pdf"


def test_validate_type_size_readability_rejects_payload_above_size_limit() -> None:
    """Block payloads that exceed configured maximum file size."""
    payload = _build_document_like_png_bytes()
    preprocessor = DocumentPreprocessor(max_file_size_bytes=len(payload) - 1)

    outcome = preprocessor.validate_type_size_readability(
        payload,
        content_type="image/png",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "file_too_large"
    assert outcome.message == "File size exceeds the configured limit."
    assert outcome.details["file_size_bytes"] == len(payload)
    assert outcome.details["max_file_size_bytes"] == len(payload) - 1


def test_validate_type_size_readability_normalizes_content_type_header() -> None:
    """Accept content types that include optional header parameters."""
    payload = _build_document_like_png_bytes()
    preprocessor = DocumentPreprocessor(max_file_size_bytes=len(payload) + 16)

    outcome = preprocessor.validate_type_size_readability(
        payload,
        content_type="image/jpeg; charset=binary",
    )

    assert outcome.is_valid is True
    assert outcome.details["content_type"] == "image/jpeg"


def test_validate_type_size_readability_rejects_corrupt_image_payload() -> None:
    """Classify undecodable bytes as unreadable payloads."""
    preprocessor = DocumentPreprocessor(max_file_size_bytes=1024)

    outcome = preprocessor.validate_type_size_readability(
        b"not-a-valid-image",
        content_type="image/png",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "unreadable_payload"


def test_validate_type_size_readability_rejects_blank_images() -> None:
    """Classify near-uniform images as unreadable due to low contrast."""
    payload = _build_blank_png_bytes()
    preprocessor = DocumentPreprocessor(max_file_size_bytes=len(payload) + 16)

    outcome = preprocessor.validate_type_size_readability(
        payload,
        content_type="image/png",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "blank_or_low_contrast"


def test_validate_type_size_readability_rejects_non_document_like_images(
) -> None:
    """Classify tiny images as non-document-like inputs."""
    payload = _build_tiny_png_bytes()
    preprocessor = DocumentPreprocessor(max_file_size_bytes=len(payload) + 16)

    outcome = preprocessor.validate_type_size_readability(
        payload,
        content_type="image/png",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "non_document_like"
