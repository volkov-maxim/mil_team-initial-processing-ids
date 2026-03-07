"""Document preprocessing primitives for input validation checks."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

DEFAULT_ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
)
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class ValidationOutcome(BaseModel):
    """Structured outcome for input validation stages."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    failure_reason: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _normalize_content_type(content_type: str | None) -> str | None:
    """Normalize content type value by dropping optional parameters."""
    if content_type is None:
        return None

    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()

    if normalized == "":
        return None

    return normalized


class DocumentPreprocessor:
    """Validate input payload constraints before alignment stages."""

    def __init__(
        self,
        *,
        allowed_content_types: Collection[str] | None = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        """Initialize validation limits for content type and payload size."""
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be greater than zero")

        source_types = (
            DEFAULT_ALLOWED_CONTENT_TYPES
            if allowed_content_types is None
            else allowed_content_types
        )

        normalized_types = tuple(
            normalized_type
            for normalized_type in (
                _normalize_content_type(source_type)
                for source_type in source_types
            )
            if normalized_type is not None
        )
        if len(normalized_types) == 0:
            raise ValueError("allowed_content_types cannot be empty")

        self.allowed_content_types = frozenset(normalized_types)
        self.max_file_size_bytes = max_file_size_bytes

    def validate_type_size_readability(
        self,
        image_bytes: bytes,
        *,
        content_type: str | None,
    ) -> ValidationOutcome:
        """Validate media type and byte-size constraints for an upload."""
        normalized_content_type = _normalize_content_type(content_type)
        if normalized_content_type is None:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="missing_content_type",
                message="Content type is required.",
                details={"content_type": content_type},
            )

        if normalized_content_type not in self.allowed_content_types:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="unsupported_media_type",
                message="Unsupported media type.",
                details={
                    "content_type": normalized_content_type,
                    "allowed_content_types": sorted(
                        self.allowed_content_types
                    ),
                },
            )

        file_size_bytes = len(image_bytes)
        if file_size_bytes > self.max_file_size_bytes:
            return ValidationOutcome(
                is_valid=False,
                failure_reason="file_too_large",
                message="File size exceeds the configured limit.",
                details={
                    "content_type": normalized_content_type,
                    "file_size_bytes": file_size_bytes,
                    "max_file_size_bytes": self.max_file_size_bytes,
                },
            )

        return ValidationOutcome(
            is_valid=True,
            details={
                "content_type": normalized_content_type,
                "file_size_bytes": file_size_bytes,
            },
        )


__all__ = [
    "DEFAULT_ALLOWED_CONTENT_TYPES",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "DocumentPreprocessor",
    "ValidationOutcome",
]
