"""Document preprocessing primitives for input validation checks."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.core.exceptions import UnprocessableDocumentError
from app.preprocessing.image_io import decode_image_bytes

DEFAULT_ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
)
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_MIN_DOCUMENT_DIMENSION_PX = 96
DEFAULT_MIN_CONTRAST_STDDEV = 12.0
DEFAULT_MIN_EDGE_DENSITY = 0.01
DEFAULT_MIN_LAPLACIAN_VARIANCE = 20.0


class ValidationOutcome(BaseModel):
    """Structured outcome for input validation stages."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    failure_reason: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _normalize_content_type(content_type: str | None) -> str | None:
    """Normalize content type value by dropping optional parameters."""
    if content_type is None:
        return None

    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()

    if normalized == "":
        return None

    return normalized


class DocumentPreprocessor:
    """Validate input payload constraints before alignment stages."""

    def __init__(
        self,
        *,
        allowed_content_types: Collection[str] | None = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        min_document_dimension_px: int = DEFAULT_MIN_DOCUMENT_DIMENSION_PX,
        min_contrast_stddev: float = DEFAULT_MIN_CONTRAST_STDDEV,
        min_edge_density: float = DEFAULT_MIN_EDGE_DENSITY,
        min_laplacian_variance: float = DEFAULT_MIN_LAPLACIAN_VARIANCE,
    ) -> None:
        """Initialize validation limits for content type and payload size."""
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be greater than zero")
        if min_document_dimension_px <= 0:
            raise ValueError(
                "min_document_dimension_px must be greater than zero"
            )
        if min_contrast_stddev < 0.0:
            raise ValueError("min_contrast_stddev must be non-negative")
        if min_edge_density < 0.0 or min_edge_density > 1.0:
            raise ValueError("min_edge_density must be between 0 and 1")
        if min_laplacian_variance < 0.0:
            raise ValueError("min_laplacian_variance must be non-negative")

        source_types = (
            DEFAULT_ALLOWED_CONTENT_TYPES
            if allowed_content_types is None
            else allowed_content_types
        )

        normalized_types = tuple(
            normalized_type
            for normalized_type in (
                _normalize_content_type(source_type)
                for source_type in source_types
            )
            if normalized_type is not None
        )
        if len(normalized_types) == 0:
            raise ValueError("allowed_content_types cannot be empty")

        self.allowed_content_types = frozenset(normalized_types)
        self.max_file_size_bytes = max_file_size_bytes
        self.min_document_dimension_px = min_document_dimension_px
        self.min_contrast_stddev = min_contrast_stddev
        self.min_edge_density = min_edge_density
        self.min_laplacian_variance = min_laplacian_variance

    def _assess_readability(
        self,
        image_bytes: bytes,
    ) -> ValidationOutcome:
        """Assess whether image payload appears readable for OCR processing."""
        try:
            decoded_image = decode_image_bytes(image_bytes)
        except UnprocessableDocumentError as error:
            details = {}
            if error.details is not None:
                details = dict(error.details)

            return ValidationOutcome(
                is_valid=False,
                failure_reason="unreadable_payload",
                message="Image payload cannot be decoded.",
                details=details,
            )

        height, width = decoded_image.shape[:2]
        if (
            min(height, width)
            < self.min_document_dimension_px
        ):
            return ValidationOutcome(
                is_valid=False,
                failure_reason="non_document_like",
                message="Image is too small to contain a readable document.",
                details={
                    "height": int(height),
                    "width": int(width),
                    "min_document_dimension_px": (
                        self.min_document_dimension_px
                    ),
                },
            )

        grayscale = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
        contrast_stddev = float(np.std(grayscale))
        if contrast_stddev < self.min_contrast_stddev:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="blank_or_low_contrast",
                message="Image appears blank or too low-contrast.",
                details={
                    "contrast_stddev": contrast_stddev,
                    "min_contrast_stddev": self.min_contrast_stddev,
                },
            )

        laplacian_variance = float(
            cv2.Laplacian(grayscale, cv2.CV_64F).var()
        )
        edges = cv2.Canny(grayscale, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        if (
            laplacian_variance < self.min_laplacian_variance
            and edge_density < self.min_edge_density
        ):
            return ValidationOutcome(
                is_valid=False,
                failure_reason="non_document_like",
                message=(
                    "Image does not contain enough visual structure for OCR."
                ),
                details={
                    "laplacian_variance": laplacian_variance,
                    "min_laplacian_variance": self.min_laplacian_variance,
                    "edge_density": edge_density,
                    "min_edge_density": self.min_edge_density,
                },
            )

        return ValidationOutcome(
            is_valid=True,
            details={
                "height": int(height),
                "width": int(width),
                "contrast_stddev": contrast_stddev,
                "laplacian_variance": laplacian_variance,
                "edge_density": edge_density,
            },
        )

    def validate_type_size_readability(
        self,
        image_bytes: bytes,
        *,
        content_type: str | None,
    ) -> ValidationOutcome:
        """Validate media type and byte-size constraints for an upload."""
        normalized_content_type = _normalize_content_type(content_type)
        if normalized_content_type is None:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="missing_content_type",
                message="Content type is required.",
                details={"content_type": content_type},
            )

        if normalized_content_type not in self.allowed_content_types:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="unsupported_media_type",
                message="Unsupported media type.",
                details={
                    "content_type": normalized_content_type,
                    "allowed_content_types": sorted(
                        self.allowed_content_types
                    ),
                },
            )

        file_size_bytes = len(image_bytes)
        if file_size_bytes > self.max_file_size_bytes:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="file_too_large",
                message="File size exceeds the configured limit.",
                details={
                    "content_type": normalized_content_type,
                    "file_size_bytes": file_size_bytes,
                    "max_file_size_bytes": self.max_file_size_bytes,
                },
            )

        readability_outcome = self._assess_readability(image_bytes)
        if not readability_outcome.is_valid:
            return ValidationOutcome(
                is_valid=False,
                failure_reason=readability_outcome.failure_reason,
                message=readability_outcome.message,
                details={
                    "content_type": normalized_content_type,
                    "file_size_bytes": file_size_bytes,
                    **readability_outcome.details,
                },
            )

        return ValidationOutcome(
            is_valid=True,
            details={
                "content_type": normalized_content_type,
                "file_size_bytes": file_size_bytes,
                **readability_outcome.details,
            },
        )


__all__ = [
    "DEFAULT_ALLOWED_CONTENT_TYPES",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "DEFAULT_MIN_CONTRAST_STDDEV",
    "DEFAULT_MIN_DOCUMENT_DIMENSION_PX",
    "DEFAULT_MIN_EDGE_DENSITY",
    "DEFAULT_MIN_LAPLACIAN_VARIANCE",
    "DocumentPreprocessor",
    "ValidationOutcome",
]
