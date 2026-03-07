"""API request and response schema contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Annotated

from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class DocumentTypeHint(str, Enum):
    """Supported user-provided document type hints."""

    BANK_CARD = "bank_card"
    ID_CARD = "id_card"
    DRIVERS_LICENSE = "drivers_license"
    AUTO = "auto"


class ProcessDocumentRequest(BaseModel):
    """Typed multipart contract for process-document input parameters."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    image: UploadFile
    document_type_hint: DocumentTypeHint = DocumentTypeHint.AUTO
    use_external_fallback: bool = False


class DocumentTypeDetected(str, Enum):
    """Document type inferred from the uploaded image."""

    BANK_CARD = "bank_card"
    ID_CARD = "id_card"
    DRIVERS_LICENSE = "drivers_license"
    UNKNOWN = "unknown"


class ExtractedFields(BaseModel):
    """Normalized extraction fields with explicit nullable values."""

    model_config = ConfigDict(extra="forbid")

    card_number: str | None = None
    cardholder_name: str | None = None
    expiry_date: str | None = None
    issuer_network: str | None = None
    bank_name: str | None = None
    full_name: str | None = None
    date_of_birth: str | None = None
    sex: str | None = None
    place_of_birth: str | None = None
    document_number: str | None = None
    license_number: str | None = None
    issuing_authority: str | None = None
    issue_date: str | None = None
    place_of_residence: str | None = None
    license_class: str | None = None


class ProcessDocumentResponse(BaseModel):
    """Typed success contract returned by ``POST /v1/process-document``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    document_type_detected: DocumentTypeDetected
    aligned_image: str
    detections: list[dict[str, Any]] = Field(default_factory=list)
    fields: ExtractedFields = Field(default_factory=ExtractedFields)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    validation_flags: list[str] = Field(default_factory=list)
    processing_metadata: dict[str, Any] = Field(default_factory=dict)


async def parse_process_document_request(
    image: Annotated[UploadFile, File(...)],
    document_type_hint: Annotated[
        DocumentTypeHint,
        Form(),
    ] = DocumentTypeHint.AUTO,
    use_external_fallback: Annotated[bool, Form()] = False,
) -> ProcessDocumentRequest:
    """Build a validated request contract from multipart payload parts."""
    return ProcessDocumentRequest(
        image=image,
        document_type_hint=document_type_hint,
        use_external_fallback=use_external_fallback,
    )
