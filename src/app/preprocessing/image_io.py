"""Image decode helpers used by preprocessing stages."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import UnprocessableDocumentError


def decode_image_bytes(
    image_bytes: bytes,
    *,
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """Decode image bytes safely and fail with a typed processing error."""
    if len(image_bytes) == 0:
        raise UnprocessableDocumentError(
            details={"reason": "empty_payload"},
        )

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)

    try:
        decoded_image = cv2.imdecode(buffer, flags)
    except cv2.error as error:
        raise UnprocessableDocumentError(
            details={
                "reason": "decoder_runtime_error",
                "opencv_error": str(error),
            },
        ) from error

    if decoded_image is None or decoded_image.size == 0:
        raise UnprocessableDocumentError(
            details={"reason": "invalid_or_corrupt_payload"},
        )

    return decoded_image


__all__ = ["decode_image_bytes"]
