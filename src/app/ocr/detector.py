"""OCR text detection models and detector interface contract."""

from __future__ import annotations

import importlib
from typing import Annotated
from typing import Any
from typing import Protocol
from typing import Sequence
from typing import cast
from typing import runtime_checkable

import numpy as np
from pydantic import ValidationError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.core.exceptions import InternalProcessingError

PolygonVertex = tuple[float, float]
PositiveDimension = Annotated[float, Field(gt=0.0)]
BoundingBox = tuple[float, float, PositiveDimension, PositiveDimension]


class TextRegion(BaseModel):
    """Single detected text region with geometry and confidence."""

    model_config = ConfigDict(extra="forbid")

    polygon: list[PolygonVertex] = Field(min_length=4, max_length=4)
    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)


DetectionResult = list[TextRegion]

EasyOCRRawPolygon = Sequence[Sequence[float | int]]
EasyOCRRawDetection = tuple[EasyOCRRawPolygon, str, float | int]


class EasyOCRReader(Protocol):
    """Protocol for EasyOCR reader objects used by the detector adapter."""

    def readtext(
        self,
        image: np.ndarray,
        *,
        detail: int = 1,
        paragraph: bool = False,
    ) -> Sequence[EasyOCRRawDetection]:
        """Run OCR and return boxes, text, and confidence tuples."""
        ...


def _build_easyocr_reader(
    *,
    languages: tuple[str, ...],
    gpu: bool,
) -> EasyOCRReader:
    """Build an EasyOCR reader instance for local detection."""
    try:
        easyocr_module = importlib.import_module("easyocr")
    except ModuleNotFoundError as error:
        raise InternalProcessingError(
            message="EasyOCR dependency is not installed.",
            error_code="ocr_detector_dependency_missing",
            details={"backend": "easyocr"},
        ) from error

    reader_factory = getattr(easyocr_module, "Reader", None)
    if reader_factory is None:
        raise InternalProcessingError(
            message="EasyOCR Reader factory is unavailable.",
            error_code="ocr_detector_backend_invalid",
            details={"backend": "easyocr"},
        )

    return cast(EasyOCRReader, reader_factory(list(languages), gpu=gpu))


def _normalize_polygon(raw_polygon: EasyOCRRawPolygon) -> list[PolygonVertex]:
    """Normalize EasyOCR polygon points into typed float vertices."""
    polygon: list[PolygonVertex] = []

    for raw_point in raw_polygon:
        if len(raw_point) < 2:
            raise ValueError("Polygon point must contain x and y values.")

        x_coord = float(raw_point[0])
        y_coord = float(raw_point[1])
        polygon.append((x_coord, y_coord))

    return polygon


def _build_bounding_box(polygon: list[PolygonVertex]) -> BoundingBox:
    """Build axis-aligned bounding box from polygon vertex coordinates."""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]

    min_x = min(x_coords)
    min_y = min(y_coords)
    max_x = max(x_coords)
    max_y = max(y_coords)

    return (
        float(min_x),
        float(min_y),
        float(max_x - min_x),
        float(max_y - min_y),
    )


class EasyOCRTextDetector:
    """Local OCR detector adapter backed by EasyOCR."""

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("ru",),
        gpu: bool = False,
        reader: EasyOCRReader | None = None,
    ) -> None:
        """Create detector with explicit or lazily constructed backend."""
        if reader is None:
            self._reader: EasyOCRReader = _build_easyocr_reader(
                languages=languages,
                gpu=gpu,
            )
        else:
            self._reader = reader

    def detect(self, aligned_image: np.ndarray) -> DetectionResult:
        """Detect text regions and normalize them into domain models."""
        try:
            raw_detections = self._reader.readtext(
                aligned_image,
                detail=1,
                paragraph=False,
            )
        except Exception as error:
            raise InternalProcessingError(
                message="OCR detector runtime failed.",
                error_code="ocr_detector_runtime_failure",
                details={
                    "backend": "easyocr",
                    "error": str(error),
                },
            ) from error

        return [
            self._to_text_region(raw_detection)
            for raw_detection in raw_detections
        ]

    def _to_text_region(
        self,
        raw_detection: EasyOCRRawDetection,
    ) -> TextRegion:
        """Convert one EasyOCR detection tuple into a validated region."""
        try:
            raw_polygon = raw_detection[0]
            confidence = float(raw_detection[2])
            polygon = _normalize_polygon(raw_polygon)
            bounding_box = _build_bounding_box(polygon)
        except (IndexError, TypeError, ValueError) as error:
            raise InternalProcessingError(
                message="OCR detector returned an invalid detection format.",
                error_code="ocr_detector_invalid_output",
                details={
                    "backend": "easyocr",
                    "raw_detection": _safe_repr(raw_detection),
                },
            ) from error

        payload = {
            "polygon": polygon,
            "bounding_box": bounding_box,
            "confidence": confidence,
        }

        try:
            return TextRegion.model_validate(payload)
        except ValidationError as error:
            raise InternalProcessingError(
                message="OCR detector output failed region validation.",
                error_code="ocr_detector_invalid_output",
                details={
                    "backend": "easyocr",
                    "validation_error": str(error),
                },
            ) from error


def _safe_repr(value: Any, *, max_len: int = 240) -> str:
    """Return a bounded string representation for diagnostic payloads."""
    rendered = repr(value)
    if len(rendered) <= max_len:
        return rendered

    return rendered[: max_len - 3] + "..."


@runtime_checkable
class TextDetector(Protocol):
    """Contract for OCR text detection adapters."""

    def detect(self, aligned_image: np.ndarray) -> DetectionResult:
        """Return detected text regions for the provided aligned image."""
        ...


__all__ = [
    "BoundingBox",
    "DetectionResult",
    "EasyOCRRawDetection",
    "EasyOCRRawPolygon",
    "EasyOCRReader",
    "EasyOCRTextDetector",
    "PositiveDimension",
    "PolygonVertex",
    "TextDetector",
    "TextRegion",
]
