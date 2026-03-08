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
        if not tokens:
            return []

        sorted_tokens = sorted(
            tokens,
            key=lambda token: (
                _token_center_y(token),
                token.bounding_box[1],
                token.bounding_box[0],
                token.text,
            ),
        )

        line_buckets: list[list[RecognizedToken]] = []
        for token in sorted_tokens:
            target_bucket = _find_matching_line_bucket(token, line_buckets)
            if target_bucket is None:
                line_buckets.append([token])
                continue

            target_bucket.append(token)

        ordered_buckets = sorted(
            line_buckets,
            key=lambda bucket: (_line_center_y(bucket), _line_min_x(bucket)),
        )

        lines: LineRecognitionResult = []
        for bucket in ordered_buckets:
            ordered_tokens = sorted(
                bucket,
                key=lambda token: (
                    token.bounding_box[0],
                    _token_center_y(token),
                    token.text,
                ),
            )
            line_text = " ".join(token.text for token in ordered_tokens)
            line_confidence = (
                sum(token.confidence for token in ordered_tokens)
                / len(ordered_tokens)
            )
            line_payload = {
                "text": line_text,
                "tokens": ordered_tokens,
                "bounding_box": _merge_line_bounding_box(ordered_tokens),
                "confidence": line_confidence,
            }
            lines.append(RecognizedLine.model_validate(line_payload))

        return lines

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


def _token_center_y(token: RecognizedToken) -> float:
    """Return vertical midpoint used for stable token ordering/grouping."""
    return token.bounding_box[1] + (token.bounding_box[3] / 2.0)


def _line_center_y(tokens: list[RecognizedToken]) -> float:
    """Return vertical midpoint of line candidates for ordering and merge."""
    min_y = min(token.bounding_box[1] for token in tokens)
    max_y = max(
        token.bounding_box[1] + token.bounding_box[3]
        for token in tokens
    )
    return min_y + ((max_y - min_y) / 2.0)


def _line_min_x(tokens: list[RecognizedToken]) -> float:
    """Return left-most x-coordinate among line candidate tokens."""
    return min(token.bounding_box[0] for token in tokens)


def _line_height(tokens: list[RecognizedToken]) -> float:
    """Return merged height of current line candidate tokens."""
    min_y = min(token.bounding_box[1] for token in tokens)
    max_y = max(
        token.bounding_box[1] + token.bounding_box[3]
        for token in tokens
    )
    return max_y - min_y


def _find_matching_line_bucket(
    token: RecognizedToken,
    line_buckets: list[list[RecognizedToken]],
) -> list[RecognizedToken] | None:
    """Find best matching line for token using vertical center tolerance."""
    token_center = _token_center_y(token)
    token_height = token.bounding_box[3]

    best_bucket: list[RecognizedToken] | None = None
    best_distance: float | None = None
    for bucket in line_buckets:
        bucket_center = _line_center_y(bucket)
        bucket_height = _line_height(bucket)
        distance = abs(token_center - bucket_center)

        # Allow grouping when centers are close relative to line/token height.
        tolerance = max(bucket_height, token_height) * 0.6
        if distance > tolerance:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_bucket = bucket

    return best_bucket


def _merge_line_bounding_box(tokens: list[RecognizedToken]) -> BoundingBox:
    """Merge token boxes into one axis-aligned line bounding box."""
    min_x = min(token.bounding_box[0] for token in tokens)
    min_y = min(token.bounding_box[1] for token in tokens)
    max_x = max(
        token.bounding_box[0] + token.bounding_box[2]
        for token in tokens
    )
    max_y = max(
        token.bounding_box[1] + token.bounding_box[3]
        for token in tokens
    )

    return (
        float(min_x),
        float(min_y),
        float(max_x - min_x),
        float(max_y - min_y),
    )


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
