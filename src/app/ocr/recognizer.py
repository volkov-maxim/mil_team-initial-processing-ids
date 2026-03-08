"""OCR recognition domain models and recognizer interface contract."""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError

from app.core.exceptions import InternalProcessingError
from app.ocr.detector import BoundingBox
from app.ocr.detector import DetectionResult
from app.ocr.detector import PolygonVertex
from app.ocr.easyocr_common import EasyOCRRawDetection
from app.ocr.easyocr_common import EasyOCRReader
from app.ocr.easyocr_common import build_bounding_box
from app.ocr.easyocr_common import build_easyocr_reader
from app.ocr.easyocr_common import normalize_polygon
from app.ocr.easyocr_common import safe_repr


class RecognizedToken(BaseModel):
    """Single recognized token with geometry and confidence metadata."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    polygon: list[PolygonVertex] = Field(min_length=4, max_length=4)
    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)


class RecognizedLine(BaseModel):
    """One line assembled from recognized tokens."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    tokens: list[RecognizedToken] = Field(min_length=1)
    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)


TokenRecognitionResult = list[RecognizedToken]
LineRecognitionResult = list[RecognizedLine]


class EasyOCRTextRecognizer:
    """Local OCR recognizer adapter backed by EasyOCR."""

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("ru",),
        gpu: bool = False,
        reader: EasyOCRReader | None = None,
    ) -> None:
        """Create recognizer with explicit or lazily constructed backend."""
        if reader is None:
            self._reader: EasyOCRReader = build_easyocr_reader(
                languages=languages,
                gpu=gpu,
                dependency_error_code="ocr_recognizer_dependency_missing",
                backend_invalid_error_code="ocr_recognizer_backend_invalid",
            )
        else:
            self._reader = reader

    def recognize(
        self,
        aligned_image: np.ndarray,
        regions: DetectionResult,
    ) -> TokenRecognitionResult:
        """Recognize text and return token outputs with geometry."""
        try:
            raw_detections = self._reader.readtext(
                aligned_image,
                detail=1,
                paragraph=False,
            )
        except Exception as error:
            raise InternalProcessingError(
                message="OCR recognizer runtime failed.",
                error_code="ocr_recognizer_runtime_failure",
                details={
                    "backend": "easyocr",
                    "error": str(error),
                },
            ) from error

        return [
            self._to_recognized_token(raw_detection)
            for raw_detection in raw_detections
        ]

    def group_tokens_to_lines(
        self,
        tokens: TokenRecognitionResult,
    ) -> LineRecognitionResult:
        """Group token-level outputs into stable line-level structures."""
        raise NotImplementedError(
            "Token-to-line grouping will be implemented in T029."
        )

    def _to_recognized_token(
        self,
        raw_detection: EasyOCRRawDetection,
    ) -> RecognizedToken:
        """Convert one EasyOCR detection tuple into a validated token."""
        try:
            raw_polygon = raw_detection[0]
            text = str(raw_detection[1])
            confidence = float(raw_detection[2])
            polygon = normalize_polygon(raw_polygon)
            bounding_box = build_bounding_box(polygon)
        except (IndexError, TypeError, ValueError) as error:
            raise InternalProcessingError(
                message="OCR recognizer returned an invalid detection format.",
                error_code="ocr_recognizer_invalid_output",
                details={
                    "backend": "easyocr",
                    "raw_detection": safe_repr(raw_detection),
                },
            ) from error

        payload = {
            "text": text,
            "polygon": polygon,
            "bounding_box": bounding_box,
            "confidence": confidence,
        }

        try:
            return RecognizedToken.model_validate(payload)
        except ValidationError as error:
            raise InternalProcessingError(
                message="OCR recognizer output failed token validation.",
                error_code="ocr_recognizer_invalid_output",
                details={
                    "backend": "easyocr",
                    "validation_error": str(error),
                },
            ) from error


@runtime_checkable
class TextRecognizer(Protocol):
    """Contract for OCR recognizer adapters and line grouping support."""

    def recognize(
        self,
        aligned_image: np.ndarray,
        regions: DetectionResult,
    ) -> TokenRecognitionResult:
        """Return recognized token outputs for detected text regions."""
        ...

    def group_tokens_to_lines(
        self,
        tokens: TokenRecognitionResult,
    ) -> LineRecognitionResult:
        """Group token-level outputs into stable line-level structures."""
        ...


__all__ = [
    "EasyOCRReader",
    "EasyOCRTextRecognizer",
    "LineRecognitionResult",
    "RecognizedLine",
    "RecognizedToken",
    "TextRecognizer",
    "TokenRecognitionResult",
]
