"""Unit tests for safe image byte decoding helpers."""

import cv2
import numpy as np
import pytest

from app.core.exceptions import UnprocessableDocumentError
from app.preprocessing.image_io import decode_image_bytes


def _build_valid_png_bytes() -> bytes:
    """Build a tiny in-memory PNG payload for decode tests."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (5, 55, 205)

    success, encoded_image = cv2.imencode(".png", image)
    assert success is True

    return encoded_image.tobytes()


def test_decode_image_bytes_returns_array_for_valid_png() -> None:
    """Decode valid image bytes into a non-empty ndarray."""
    decoded = decode_image_bytes(_build_valid_png_bytes())

    assert decoded.shape == (8, 8, 3)
    assert decoded.dtype == np.uint8
    assert decoded.size > 0


def test_decode_image_bytes_raises_for_invalid_payload() -> None:
    """Raise typed unprocessable error when bytes cannot be decoded."""
    with pytest.raises(UnprocessableDocumentError) as exc_info:
        decode_image_bytes(b"not-a-valid-image")

    assert exc_info.value.error_code == "unprocessable_document"


def test_decode_image_bytes_raises_for_empty_payload() -> None:
    """Raise typed unprocessable error when payload bytes are empty."""
    with pytest.raises(UnprocessableDocumentError):
        decode_image_bytes(b"")
