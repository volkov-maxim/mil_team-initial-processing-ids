"""Unit tests for typed core exception primitives."""

from app.core.exceptions import ErrorCategory
from app.core.exceptions import InputValidationError
from app.core.exceptions import InternalProcessingError
from app.core.exceptions import UnprocessableDocumentError


def test_input_validation_error_exposes_expected_category_payload() -> None:
    """Serialize input validation failures with category and details."""
    error = InputValidationError(
        message="Unsupported media type.",
        details={"content_type": "application/pdf"},
    )

    payload = error.to_payload()

    assert error.status_code == 400
    assert error.category == ErrorCategory.INPUT_VALIDATION
    assert payload["error_code"] == "invalid_input"
    assert payload["message"] == "Unsupported media type."
    assert payload["category"] == "input_validation"
    assert payload["details"] == {"content_type": "application/pdf"}


def test_unprocessable_document_error_has_expected_defaults() -> None:
    """Use default error code/message for unreadable document failures."""
    error = UnprocessableDocumentError()

    payload = error.to_payload()

    assert error.status_code == 422
    assert error.category == ErrorCategory.DOCUMENT_PROCESSING
    assert payload["error_code"] == "unprocessable_document"
    assert payload["message"] == (
        "Document image is unreadable or cannot be aligned."
    )
    assert payload["category"] == "document_processing"
    assert "details" not in payload


def test_internal_processing_error_supports_custom_error_code() -> None:
    """Allow custom machine-readable codes while preserving category."""
    error = InternalProcessingError(
        message="OCR runtime failed.",
        error_code="ocr_runtime_failure",
        details={"stage": "ocr"},
    )

    payload = error.to_payload()

    assert error.status_code == 500
    assert error.category == ErrorCategory.INTERNAL
    assert payload["error_code"] == "ocr_runtime_failure"
    assert payload["message"] == "OCR runtime failed."
    assert payload["category"] == "internal"
    assert payload["details"] == {"stage": "ocr"}