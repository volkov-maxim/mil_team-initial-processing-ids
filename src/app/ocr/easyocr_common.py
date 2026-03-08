"""Shared EasyOCR helper types and utility functions."""

from __future__ import annotations

import importlib
from typing import Any
from typing import Protocol
from typing import Sequence
from typing import cast

import numpy as np

from app.core.exceptions import InternalProcessingError

EasyOCRRawPolygon = Sequence[Sequence[float | int]]
EasyOCRRawDetection = tuple[EasyOCRRawPolygon, str, float | int]
NormalizedPolygon = list[tuple[float, float]]
AxisAlignedBoundingBox = tuple[float, float, float, float]


class EasyOCRReader(Protocol):
    """Protocol for EasyOCR reader objects used by OCR adapters."""

    def readtext(
        self,
        image: np.ndarray,
        *,
        detail: int = 1,
        paragraph: bool = False,
    ) -> Sequence[EasyOCRRawDetection]:
        """Run OCR and return boxes, text, and confidence tuples."""
        ...


def build_easyocr_reader(
    *,
    languages: tuple[str, ...],
    gpu: bool,
    dependency_error_code: str,
    backend_invalid_error_code: str,
) -> EasyOCRReader:
    """Build an EasyOCR reader instance for local OCR processing."""
    try:
        easyocr_module = importlib.import_module("easyocr")
    except ModuleNotFoundError as error:
        raise InternalProcessingError(
            message="EasyOCR dependency is not installed.",
            error_code=dependency_error_code,
            details={"backend": "easyocr"},
        ) from error

    reader_factory = getattr(easyocr_module, "Reader", None)
    if reader_factory is None:
        raise InternalProcessingError(
            message="EasyOCR Reader factory is unavailable.",
            error_code=backend_invalid_error_code,
            details={"backend": "easyocr"},
        )

    return cast(EasyOCRReader, reader_factory(list(languages), gpu=gpu))


def normalize_polygon(raw_polygon: EasyOCRRawPolygon) -> NormalizedPolygon:
    """Normalize EasyOCR polygon points into typed float vertices."""
    polygon: NormalizedPolygon = []

    for raw_point in raw_polygon:
        if len(raw_point) < 2:
            raise ValueError("Polygon point must contain x and y values.")

        x_coord = float(raw_point[0])
        y_coord = float(raw_point[1])
        polygon.append((x_coord, y_coord))

    return polygon


def build_bounding_box(
    polygon: Sequence[tuple[float, float]],
) -> AxisAlignedBoundingBox:
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


def safe_repr(value: Any, *, max_len: int = 240) -> str:
    """Return a bounded string representation for diagnostic payloads."""
    rendered = repr(value)
    if len(rendered) <= max_len:
        return rendered

    return rendered[: max_len - 3] + "..."


__all__ = [
    "AxisAlignedBoundingBox",
    "EasyOCRRawDetection",
    "EasyOCRRawPolygon",
    "EasyOCRReader",
    "NormalizedPolygon",
    "build_bounding_box",
    "build_easyocr_reader",
    "normalize_polygon",
    "safe_repr",
]
