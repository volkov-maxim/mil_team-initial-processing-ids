"""Typed core exception primitives for API and pipeline error handling."""

from __future__ import annotations

from enum import StrEnum
from http import HTTPStatus
from typing import Any, Mapping


class ErrorCategory(StrEnum):
    """Stable category values used for error mapping and observability."""

    INPUT_VALIDATION = "input_validation"
    DOCUMENT_PROCESSING = "document_processing"
    INTERNAL = "internal"


class AppCoreError(Exception):
    """Base typed exception with machine-readable payload metadata."""

    default_error_code: str = "internal_processing_error"
    default_message: str = "Internal processing failure."
    default_status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    default_category: ErrorCategory = ErrorCategory.INTERNAL

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize an error instance with optional overrides."""
        resolved_message = self.default_message if message is None else message
        super().__init__(resolved_message)

        self.message = resolved_message
        self.error_code = (
            self.default_error_code if error_code is None else error_code
        )
        self.status_code = int(self.default_status_code)
        self.category = self.default_category
        self.details = None if details is None else dict(details)

    def to_payload(self) -> dict[str, Any]:
        """Return the exception payload for API error envelope composition."""
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
        }

        if self.details:
            payload["details"] = self.details

        return payload


class InputValidationError(AppCoreError):
    """Represent request-shape and input-validation failures (HTTP 400)."""

    default_error_code = "invalid_input"
    default_message = "Request input is invalid."
    default_status_code = HTTPStatus.BAD_REQUEST
    default_category = ErrorCategory.INPUT_VALIDATION


class UnprocessableDocumentError(AppCoreError):
    """Represent unreadable or non-processable document failures (HTTP 422)."""

    default_error_code = "unprocessable_document"
    default_message = "Document image is unreadable or cannot be aligned."
    default_status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    default_category = ErrorCategory.DOCUMENT_PROCESSING


class InternalProcessingError(AppCoreError):
    """Represent internal service/runtime failures (HTTP 500)."""

    default_error_code = "internal_processing_error"
    default_message = "Internal processing failure."
    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_category = ErrorCategory.INTERNAL


__all__ = [
    "AppCoreError",
    "ErrorCategory",
    "InputValidationError",
    "InternalProcessingError",
    "UnprocessableDocumentError",
]