"""Typed API error response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict


class ApiErrorResponse(BaseModel):
    """Base API error envelope shared by all HTTP error responses."""

    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    details: dict[str, Any] | None = None


class BadRequestErrorResponse(ApiErrorResponse):
    """Error envelope for ``HTTP 400`` invalid input responses."""

    error_code: str = "invalid_input"
    message: str = "Request input is invalid."


class UnprocessableEntityErrorResponse(ApiErrorResponse):
    """Error envelope for ``HTTP 422`` unprocessable document responses."""

    error_code: str = "unprocessable_document"
    message: str = "Document image is unreadable or cannot be aligned."


class InternalServerErrorResponse(ApiErrorResponse):
    """Error envelope for ``HTTP 500`` internal processing failures."""

    error_code: str = "internal_processing_error"
    message: str = "Internal processing failure."


__all__ = [
    "ApiErrorResponse",
    "BadRequestErrorResponse",
    "InternalServerErrorResponse",
    "UnprocessableEntityErrorResponse",
]
