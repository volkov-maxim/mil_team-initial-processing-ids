"""API request and response schema contracts."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import ConfigDict


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
