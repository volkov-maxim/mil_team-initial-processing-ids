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
DEFAULT_BOUNDARY_MIN_AREA_RATIO = 0.05
DEFAULT_BOUNDARY_APPROXIMATION_RATIO = 0.02
DEFAULT_BOUNDARY_CANNY_THRESHOLD_LOW = 60
DEFAULT_BOUNDARY_CANNY_THRESHOLD_HIGH = 180


class ValidationOutcome(BaseModel):
    """Structured outcome for input validation stages."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    failure_reason: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class BoundaryDetectionResult(BaseModel):
    """Structured result for detected document corners and bounds."""

    model_config = ConfigDict(extra="forbid")

    corners: list[tuple[int, int]] = Field(min_length=4, max_length=4)
    bounds: tuple[int, int, int, int]
    contour_area: float = Field(ge=0.0)
    area_ratio: float = Field(ge=0.0, le=1.0)


def _normalize_content_type(content_type: str | None) -> str | None:
    """Normalize content type value by dropping optional parameters."""
    if content_type is None:
        return None

    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()

    if normalized == "":
        return None

    return normalized


def _order_boundary_points(points: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[2] = points[np.argmax(sums)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


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
        boundary_min_area_ratio: float = DEFAULT_BOUNDARY_MIN_AREA_RATIO,
        boundary_approximation_ratio: float = (
            DEFAULT_BOUNDARY_APPROXIMATION_RATIO
        ),
        boundary_canny_threshold_low: int = (
            DEFAULT_BOUNDARY_CANNY_THRESHOLD_LOW
        ),
        boundary_canny_threshold_high: int = (
            DEFAULT_BOUNDARY_CANNY_THRESHOLD_HIGH
        ),
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
        if boundary_min_area_ratio < 0.0 or boundary_min_area_ratio > 1.0:
            raise ValueError("boundary_min_area_ratio must be between 0 and 1")
        if boundary_approximation_ratio <= 0.0:
            raise ValueError(
                "boundary_approximation_ratio must be greater than zero"
            )
        if boundary_canny_threshold_low < 0:
            raise ValueError(
                "boundary_canny_threshold_low must be non-negative"
            )
        if boundary_canny_threshold_high <= boundary_canny_threshold_low:
            raise ValueError(
                "boundary_canny_threshold_high must exceed low threshold"
            )

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
        self.boundary_min_area_ratio = boundary_min_area_ratio
        self.boundary_approximation_ratio = boundary_approximation_ratio
        self.boundary_canny_threshold_low = boundary_canny_threshold_low
        self.boundary_canny_threshold_high = boundary_canny_threshold_high

    def detect_document_boundary(
        self,
        image: np.ndarray,
    ) -> BoundaryDetectionResult:
        """Detect four document corners and a bounding box in an image."""
        if image.size == 0:
            raise UnprocessableDocumentError(
                message="Cannot detect boundary on an empty image.",
                details={"reason": "empty_image"},
            )

        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise UnprocessableDocumentError(
                message="Unsupported image shape for boundary detection.",
                details={"reason": "invalid_image_shape"},
            )

        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            self.boundary_canny_threshold_low,
            self.boundary_canny_threshold_high,
        )
        dilation_kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.dilate(edges, dilation_kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) == 0:
            raise UnprocessableDocumentError(
                message="Unable to detect document boundary.",
                details={"reason": "no_contours_found"},
            )

        image_height, image_width = grayscale.shape[:2]
        image_area = float(image_height * image_width)
        min_boundary_area = image_area * self.boundary_min_area_ratio

        selected_points: np.ndarray | None = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            contour_area = float(cv2.contourArea(contour))
            if contour_area < min_boundary_area:
                continue

            contour_perimeter = cv2.arcLength(contour, True)
            if contour_perimeter <= 0.0:
                continue

            approximation = cv2.approxPolyDP(
                contour,
                self.boundary_approximation_ratio * contour_perimeter,
                True,
            )
            if len(approximation) == 4:
                selected_points = approximation.reshape(4, 2).astype(
                    np.float32
                )
                break

        if selected_points is None:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = float(cv2.contourArea(largest_contour))
            if largest_area < min_boundary_area:
                raise UnprocessableDocumentError(
                    message="Unable to detect document boundary.",
                    details={
                        "reason": "boundary_area_below_threshold",
                        "boundary_min_area_ratio": (
                            self.boundary_min_area_ratio
                        ),
                    },
                )

            rotated_rect = cv2.minAreaRect(largest_contour)
            selected_points = cv2.boxPoints(rotated_rect).astype(np.float32)

        ordered_points = _order_boundary_points(selected_points)
        ordered_points[:, 0] = np.clip(
            ordered_points[:, 0],
            0,
            image_width - 1,
        )
        ordered_points[:, 1] = np.clip(
            ordered_points[:, 1],
            0,
            image_height - 1,
        )

        contour_area = float(cv2.contourArea(ordered_points))
        area_ratio = contour_area / image_area
        area_ratio = float(np.clip(area_ratio, 0.0, 1.0))

        corners_array = np.rint(ordered_points).astype(int)
        x_coords = corners_array[:, 0]
        y_coords = corners_array[:, 1]
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))

        return BoundaryDetectionResult(
            corners=[
                (int(x_coord), int(y_coord))
                for x_coord, y_coord in corners_array
            ],
            bounds=(
                x_min,
                y_min,
                int(x_max - x_min + 1),
                int(y_max - y_min + 1),
            ),
            contour_area=contour_area,
            area_ratio=area_ratio,
        )

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
    "BoundaryDetectionResult",
    "DEFAULT_BOUNDARY_APPROXIMATION_RATIO",
    "DEFAULT_BOUNDARY_CANNY_THRESHOLD_HIGH",
    "DEFAULT_BOUNDARY_CANNY_THRESHOLD_LOW",
    "DEFAULT_BOUNDARY_MIN_AREA_RATIO",
    "DEFAULT_ALLOWED_CONTENT_TYPES",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "DEFAULT_MIN_CONTRAST_STDDEV",
    "DEFAULT_MIN_DOCUMENT_DIMENSION_PX",
    "DEFAULT_MIN_EDGE_DENSITY",
    "DEFAULT_MIN_LAPLACIAN_VARIANCE",
    "DocumentPreprocessor",
    "ValidationOutcome",
]
