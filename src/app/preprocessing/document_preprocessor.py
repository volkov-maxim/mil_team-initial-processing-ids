"""Document preprocessing primitives for input validation checks."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any
from typing import Literal

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
DEFAULT_PERSPECTIVE_INTERPOLATION = cv2.INTER_LINEAR
DEFAULT_ROTATION_TOP_BAND_FRACTION = 0.30
DEFAULT_ROTATION_LANDSCAPE_BONUS = 0.25


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


AlignmentStage = Literal[
    "boundary_detection",
    "perspective_correction",
    "rotation_normalization",
    "denoise_contrast",
]


class AlignmentFailureDiagnostic(BaseModel):
    """Typed alignment failure payload for stage-level diagnostics."""

    model_config = ConfigDict(extra="forbid")

    stage: AlignmentStage
    reason: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


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


def _compute_perspective_target_size(
    ordered_points: np.ndarray,
) -> tuple[int, int]:
    """Compute canonical output size from ordered boundary points."""
    top_left, top_right, bottom_right, bottom_left = ordered_points

    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)

    target_width = max(1, int(round(max(top_width, bottom_width))))
    target_height = max(1, int(round(max(left_height, right_height))))

    return target_width, target_height


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale while preserving grayscale inputs."""
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by arbitrary angle while preserving complete content."""
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


def _estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate arbitrary skew angle in degrees for deskew rotation."""
    grayscale = _as_grayscale(image)
    _, threshold = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    coordinates = cv2.findNonZero(threshold)
    if coordinates is None or len(coordinates) < 20:
        return 0.0

    angle = float(cv2.minAreaRect(coordinates)[-1])
    if angle < -45.0:
        return 90.0 + angle

    return angle


def _crop_foreground_region(image: np.ndarray) -> np.ndarray:
    """Crop image to foreground bounds while preserving background margins."""
    grayscale = _as_grayscale(image)
    _, threshold = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    coordinates = cv2.findNonZero(threshold)
    if coordinates is None:
        return image

    x_coord, y_coord, width, height = cv2.boundingRect(coordinates)
    return image[y_coord : y_coord + height, x_coord : x_coord + width]


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
        perspective_interpolation: int = DEFAULT_PERSPECTIVE_INTERPOLATION,
        rotation_top_band_fraction: float = (
            DEFAULT_ROTATION_TOP_BAND_FRACTION
        ),
        rotation_landscape_bonus: float = DEFAULT_ROTATION_LANDSCAPE_BONUS,
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
        if perspective_interpolation < 0:
            raise ValueError("perspective_interpolation must be non-negative")
        if rotation_top_band_fraction <= 0.0:
            raise ValueError(
                "rotation_top_band_fraction must be greater than zero"
            )
        if rotation_top_band_fraction >= 1.0:
            raise ValueError(
                "rotation_top_band_fraction must be less than one"
            )
        if rotation_landscape_bonus < 0.0:
            raise ValueError("rotation_landscape_bonus must be non-negative")

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
        self.perspective_interpolation = perspective_interpolation
        self.rotation_top_band_fraction = rotation_top_band_fraction
        self.rotation_landscape_bonus = rotation_landscape_bonus

    def _build_alignment_failure_error(
        self,
        *,
        stage: AlignmentStage,
        error: UnprocessableDocumentError,
    ) -> UnprocessableDocumentError:
        """Wrap stage failures with a structured typed diagnostic payload."""
        source_details: dict[str, Any] = {}
        if error.details is not None:
            source_details = dict(error.details)

        reason_value = source_details.get("reason")
        reason = "unknown_alignment_reason"
        if isinstance(reason_value, str) and reason_value != "":
            reason = reason_value

        diagnostic = AlignmentFailureDiagnostic(
            stage=stage,
            reason=reason,
            message=error.message,
            details=source_details,
        )

        return UnprocessableDocumentError(
            message="Document alignment failed.",
            details={
                "reason": "alignment_failure",
                "failure_stage": stage,
                "alignment_diagnostic": diagnostic.model_dump(),
            },
        )

    def align_image(
        self,
        image: np.ndarray,
        *,
        apply_denoise: bool = True,
        apply_contrast_normalization: bool = True,
    ) -> np.ndarray:
        """Align image through boundary, perspective, rotation, and cleanup."""
        try:
            boundary = self.detect_document_boundary(image)
        except UnprocessableDocumentError as error:
            raise self._build_alignment_failure_error(
                stage="boundary_detection",
                error=error,
            ) from error

        try:
            perspective_corrected = self.apply_perspective_correction(
                image,
                boundary,
            )
        except UnprocessableDocumentError as error:
            raise self._build_alignment_failure_error(
                stage="perspective_correction",
                error=error,
            ) from error

        try:
            rotation_normalized = self.normalize_rotation(
                perspective_corrected
            )
        except UnprocessableDocumentError as error:
            raise self._build_alignment_failure_error(
                stage="rotation_normalization",
                error=error,
            ) from error

        try:
            return self.denoise_contrast(
                rotation_normalized,
                apply_denoise=apply_denoise,
                apply_contrast_normalization=(
                    apply_contrast_normalization
                ),
            )
        except UnprocessableDocumentError as error:
            raise self._build_alignment_failure_error(
                stage="denoise_contrast",
                error=error,
            ) from error

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

    def apply_perspective_correction(
        self,
        image: np.ndarray,
        boundary: BoundaryDetectionResult,
    ) -> np.ndarray:
        """Warp document boundary to a canonical rectangular image plane."""
        if image.size == 0:
            raise UnprocessableDocumentError(
                message="Cannot correct perspective on an empty image.",
                details={"reason": "empty_image"},
            )
        if image.ndim not in {2, 3}:
            raise UnprocessableDocumentError(
                message="Unsupported image shape for perspective correction.",
                details={"reason": "invalid_image_shape"},
            )

        source_points = _order_boundary_points(
            np.asarray(boundary.corners, dtype=np.float32)
        )
        target_width, target_height = _compute_perspective_target_size(
            source_points
        )

        destination_points = np.array(
            [
                [0.0, 0.0],
                [float(target_width - 1), 0.0],
                [float(target_width - 1), float(target_height - 1)],
                [0.0, float(target_height - 1)],
            ],
            dtype=np.float32,
        )

        transform_matrix = cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )
        aligned_image = cv2.warpPerspective(
            image,
            transform_matrix,
            (target_width, target_height),
            flags=self.perspective_interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned_image

    def _rotation_score(self, image: np.ndarray) -> float:
        """Compute orientation score used for rotation normalization."""
        grayscale_uint8 = _as_grayscale(image)
        grayscale = grayscale_uint8.astype(np.float32)

        _, threshold = cv2.threshold(
            grayscale_uint8,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        row_projection = threshold.sum(axis=1).astype(np.float32)
        col_projection = threshold.sum(axis=0).astype(np.float32)
        horizontal_bias = float(
            np.var(row_projection) / (np.var(col_projection) + 1e-6)
        )

        band_height = max(
            1,
            int(round(grayscale.shape[0] * self.rotation_top_band_fraction)),
        )
        top_mean = float(np.mean(grayscale[:band_height, :]))
        bottom_mean = float(np.mean(grayscale[-band_height:, :]))
        top_darkness_bias = (bottom_mean - top_mean) / 255.0

        landscape_bias = 0.0
        if image.shape[1] >= image.shape[0]:
            landscape_bias = self.rotation_landscape_bonus

        return horizontal_bias + top_darkness_bias + landscape_bias

    def normalize_rotation(self, image: np.ndarray) -> np.ndarray:
        """Normalize orientation for perspective-corrected document images."""
        if image.size == 0:
            raise UnprocessableDocumentError(
                message="Cannot normalize rotation on an empty image.",
                details={"reason": "empty_image"},
            )
        if image.ndim not in {2, 3}:
            raise UnprocessableDocumentError(
                message="Unsupported image shape for rotation normalization.",
                details={"reason": "invalid_image_shape"},
            )

        skew_angle = _estimate_skew_angle(image)
        deskewed = _rotate_bound(image, skew_angle)

        candidates: tuple[np.ndarray, ...] = (
            deskewed,
            cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(deskewed, cv2.ROTATE_180),
            cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
        cropped_candidates = [
            _crop_foreground_region(candidate) for candidate in candidates
        ]
        scores = [
            self._rotation_score(candidate)
            for candidate in cropped_candidates
        ]
        best_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))

        return cropped_candidates[best_index]

    def denoise_contrast(
        self,
        image: np.ndarray,
        *,
        apply_denoise: bool = True,
        apply_contrast_normalization: bool = True,
    ) -> np.ndarray:
        """Apply optional denoise and contrast normalization for OCR."""
        if image.size == 0:
            raise UnprocessableDocumentError(
                message="Cannot denoise or normalize an empty image.",
                details={"reason": "empty_image"},
            )
        if image.ndim not in {2, 3}:
            raise UnprocessableDocumentError(
                message=(
                    "Unsupported image shape for denoise/contrast stage."
                ),
                details={"reason": "invalid_image_shape"},
            )

        processed = image.copy()

        if apply_denoise:
            try:
                if processed.ndim == 2:
                    processed = cv2.fastNlMeansDenoising(
                        processed,
                        None,
                        7,
                        7,
                        21,
                    )
                else:
                    processed = cv2.fastNlMeansDenoisingColored(
                        processed,
                        None,
                        7,
                        7,
                        7,
                        21,
                    )
            except cv2.error:
                pass

        if apply_contrast_normalization:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            try:
                if processed.ndim == 2:
                    processed = clahe.apply(processed)
                else:
                    lab_image = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                    channel_l, channel_a, channel_b = cv2.split(lab_image)
                    normalized_l = clahe.apply(channel_l)
                    merged = cv2.merge(
                        (normalized_l, channel_a, channel_b)
                    )
                    processed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            except cv2.error:
                pass

        return processed

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
    "AlignmentFailureDiagnostic",
    "AlignmentStage",
    "BoundaryDetectionResult",
    "DEFAULT_BOUNDARY_APPROXIMATION_RATIO",
    "DEFAULT_BOUNDARY_CANNY_THRESHOLD_HIGH",
    "DEFAULT_BOUNDARY_CANNY_THRESHOLD_LOW",
    "DEFAULT_BOUNDARY_MIN_AREA_RATIO",
    "DEFAULT_PERSPECTIVE_INTERPOLATION",
    "DEFAULT_ROTATION_LANDSCAPE_BONUS",
    "DEFAULT_ROTATION_TOP_BAND_FRACTION",
    "DEFAULT_ALLOWED_CONTENT_TYPES",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "DEFAULT_MIN_CONTRAST_STDDEV",
    "DEFAULT_MIN_DOCUMENT_DIMENSION_PX",
    "DEFAULT_MIN_EDGE_DENSITY",
    "DEFAULT_MIN_LAPLACIAN_VARIANCE",
    "DocumentPreprocessor",
    "ValidationOutcome",
]
