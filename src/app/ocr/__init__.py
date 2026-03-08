"""Text detection and OCR package."""

from app.ocr.detector import BoundingBox
from app.ocr.detector import DetectionResult
from app.ocr.detector import PolygonVertex
from app.ocr.detector import PositiveDimension
from app.ocr.detector import TextDetector
from app.ocr.detector import TextRegion

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "PolygonVertex",
    "PositiveDimension",
    "TextDetector",
    "TextRegion",
]
