"""OCR text detection models and detector interface contract."""

from __future__ import annotations

from typing import Annotated
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError

from app.core.exceptions import InternalProcessingError
from app.ocr.easyocr_common import EasyOCRRawDetection
from app.ocr.easyocr_common import EasyOCRRawPolygon
from app.ocr.easyocr_common import EasyOCRReader
from app.ocr.easyocr_common import build_bounding_box
from app.ocr.easyocr_common import build_easyocr_reader
from app.ocr.easyocr_common import normalize_polygon
from app.ocr.easyocr_common import safe_repr

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
            self._reader: EasyOCRReader = build_easyocr_reader(
                languages=languages,
                gpu=gpu,
                dependency_error_code="ocr_detector_dependency_missing",
                backend_invalid_error_code="ocr_detector_backend_invalid",
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
            polygon = normalize_polygon(raw_polygon)
            bounding_box = build_bounding_box(polygon)
        except (IndexError, TypeError, ValueError) as error:
            raise InternalProcessingError(
                message="OCR detector returned an invalid detection format.",
                error_code="ocr_detector_invalid_output",
                details={
                    "backend": "easyocr",
                    "raw_detection": safe_repr(raw_detection),
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
