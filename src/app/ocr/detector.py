"""OCR text detection models and detector interface contract."""

from __future__ import annotations

from typing import Annotated
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

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


@runtime_checkable
class TextDetector(Protocol):
    """Contract for OCR text detection adapters."""

    def detect(self, aligned_image: np.ndarray) -> DetectionResult:
        """Return detected text regions for the provided aligned image."""
        ...


__all__ = [
    "BoundingBox",
    "DetectionResult",
    "PositiveDimension",
    "PolygonVertex",
    "TextDetector",
    "TextRegion",
]
