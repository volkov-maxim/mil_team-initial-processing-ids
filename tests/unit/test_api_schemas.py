"""Unit tests for API request schema validation contracts."""

from io import BytesIO

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from app.api.schemas import DocumentTypeHint
from app.api.schemas import ExtractedFields
from app.api.schemas import ProcessDocumentRequest
from app.api.schemas import ProcessDocumentResponse


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


def test_process_document_response_includes_required_contract_keys() -> None:
    """Serialize a payload with all required top-level response keys."""
    response_payload = ProcessDocumentResponse(
        request_id="req-001",
        document_type_detected="id_card",
        aligned_image="artifacts/req-001/aligned.png",
        detections=[],
        field_confidence={"full_name": 0.95},
        validation_flags=["name_verified"],
        processing_metadata={"device": "cpu", "fallback_used": False},
    )

    serialized = response_payload.model_dump()

    assert "request_id" in serialized
    assert "document_type_detected" in serialized
    assert "aligned_image" in serialized
    assert "detections" in serialized
    assert "fields" in serialized
    assert "field_confidence" in serialized
    assert "validation_flags" in serialized
    assert "processing_metadata" in serialized


def test_process_document_response_keeps_nullable_fields_explicit() -> None:
    """Keep missing extraction values as explicit null entries."""
    response_payload = ProcessDocumentResponse(
        request_id="req-002",
        document_type_detected="drivers_license",
        aligned_image="artifacts/req-002/aligned.png",
        detections=[],
        fields=ExtractedFields(
            full_name="Alex Doe",
            date_of_birth=None,
            license_number=None,
        ),
        field_confidence={},
        validation_flags=[],
        processing_metadata={"device": "cpu", "fallback_used": False},
    )

    serialized = response_payload.model_dump()

    assert serialized["fields"]["full_name"] == "Alex Doe"
    assert serialized["fields"]["date_of_birth"] is None
    assert serialized["fields"]["license_number"] is None


def test_extracted_fields_match_document_schema_contract() -> None:
    """Keep response field keys aligned with Chapter 8 schema contract."""
    expected_field_names = {
        "card_number",
        "cardholder_name",
        "expiry_date",
        "issuer_network",
        "bank_name",
        "full_name",
        "date_of_birth",
        "sex",
        "place_of_birth",
        "document_number",
        "issuing_authority",
        "issue_date",
        "license_number",
        "place_of_residence",
        "license_class",
    }

    actual_field_names = set(ExtractedFields.model_fields.keys())

    assert actual_field_names == expected_field_names
