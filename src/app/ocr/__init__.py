"""Text detection and OCR package."""

from app.ocr.detector import BoundingBox
from app.ocr.detector import DetectionResult
from app.ocr.detector import EasyOCRTextDetector
from app.ocr.detector import PolygonVertex
from app.ocr.detector import PositiveDimension
from app.ocr.detector import TextDetector
from app.ocr.detector import TextRegion
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken
from app.ocr.recognizer import TextRecognizer
from app.ocr.recognizer import TokenRecognitionResult

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "EasyOCRTextDetector",
    "LineRecognitionResult",
    "PolygonVertex",
    "PositiveDimension",
    "RecognizedLine",
    "RecognizedToken",
    "TextDetector",
    "TextRecognizer",
    "TextRegion",
    "TokenRecognitionResult",
]
