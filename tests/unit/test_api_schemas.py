"""Unit tests for API request schema validation contracts."""

from io import BytesIO

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from app.api.schemas import DocumentTypeHint
from app.api.schemas import ProcessDocumentRequest


def _build_upload_file() -> UploadFile:
    """Create an in-memory upload file for schema validation tests."""
    return UploadFile(file=BytesIO(b"fake-image-bytes"), filename="sample.png")


def test_process_document_request_uses_expected_defaults() -> None:
    """Use default hint and fallback values when omitted by the client."""
    request_payload = ProcessDocumentRequest.model_validate(
        {"image": _build_upload_file()}
    )

    assert request_payload.document_type_hint is DocumentTypeHint.AUTO
    assert request_payload.use_external_fallback is False


def test_process_document_request_accepts_valid_enum_and_boolean() -> None:
    """Parse valid form-compatible values for enum and boolean fields."""
    request_payload = ProcessDocumentRequest.model_validate(
        {
            "image": _build_upload_file(),
            "document_type_hint": "id_card",
            "use_external_fallback": "true",
        }
    )

    assert request_payload.document_type_hint is DocumentTypeHint.ID_CARD
    assert request_payload.use_external_fallback is True


def test_process_document_request_rejects_invalid_document_type_hint() -> None:
    """Reject unknown document type hints that are outside the contract."""
    with pytest.raises(ValidationError):
        ProcessDocumentRequest.model_validate(
            {
                "image": _build_upload_file(),
                "document_type_hint": "passport",
            }
        )


def test_process_document_request_rejects_invalid_boolean_value() -> None:
    """Reject non-boolean fallback values that cannot be parsed safely."""
    with pytest.raises(ValidationError):
        ProcessDocumentRequest.model_validate(
            {
                "image": _build_upload_file(),
                "use_external_fallback": "sometimes",
            }
        )


def test_process_document_request_rejects_missing_image() -> None:
    """Require an uploaded file to satisfy the multipart contract."""
    with pytest.raises(ValidationError):
        ProcessDocumentRequest.model_validate(
            {"document_type_hint": "bank_card"}
        )
