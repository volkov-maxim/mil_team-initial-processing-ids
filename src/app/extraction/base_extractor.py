"""Base extractor interface and extraction-layer field model."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from pydantic import ConfigDict

from app.api.schemas import ExtractedFields
from app.ocr.recognizer import LineRecognitionResult


class ExtractedFieldsModel(ExtractedFields):
    """Extraction-layer field model with explicit-null defaults."""

    model_config = ConfigDict(extra="forbid")


class BaseExtractor(ABC):
    """Base contract for document-specific OCR extraction adapters."""

    def extract(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        """Extract normalized fields from grouped OCR lines."""
        if not ocr_lines:
            return self.empty_fields()

        extracted_fields = self.extract_from_lines(ocr_lines)
        if not isinstance(extracted_fields, ExtractedFieldsModel):
            raise TypeError(
                "Extractor implementations must return ExtractedFieldsModel."
            )

        return extracted_fields

    def empty_fields(self) -> ExtractedFieldsModel:
        """Return explicit-null fields for empty or partial extraction."""
        return ExtractedFieldsModel()

    @abstractmethod
    def extract_from_lines(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        """Extract document-specific fields from OCR line outputs."""


__all__ = ["BaseExtractor", "ExtractedFieldsModel"]
