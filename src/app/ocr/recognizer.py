"""OCR recognition domain models and recognizer interface contract."""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.ocr.detector import BoundingBox
from app.ocr.detector import DetectionResult
from app.ocr.detector import PolygonVertex


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
    "LineRecognitionResult",
    "RecognizedLine",
    "RecognizedToken",
    "TextRecognizer",
    "TokenRecognitionResult",
]
