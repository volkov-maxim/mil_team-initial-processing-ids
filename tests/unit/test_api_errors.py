"""Unit tests for typed API error response models."""

from pydantic import ValidationError

from app.api.errors import BadRequestErrorResponse
from app.api.errors import InternalServerErrorResponse
from app.api.errors import UnprocessableEntityErrorResponse


def test_bad_request_error_response_includes_required_envelope_fields() -> None:
    """Expose machine-readable code and message for 400 responses."""
    payload = BadRequestErrorResponse().model_dump()

    assert payload["error_code"] == "invalid_input"
    assert payload["message"] == "Request input is invalid."


def test_unprocessable_error_response_includes_required_envelope_fields() -> None:
    """Expose machine-readable code and message for 422 responses."""
    payload = UnprocessableEntityErrorResponse().model_dump()

    assert payload["error_code"] == "unprocessable_document"
    assert payload["message"] == (
        "Document image is unreadable or cannot be aligned."
    )


def test_internal_error_response_includes_required_envelope_fields() -> None:
    """Expose machine-readable code and message for 500 responses."""
    payload = InternalServerErrorResponse().model_dump()

    assert payload["error_code"] == "internal_processing_error"
    assert payload["message"] == "Internal processing failure."


def test_error_response_models_reject_unknown_fields() -> None:
    """Reject undeclared fields to keep API error contracts strict."""
    try:
        BadRequestErrorResponse.model_validate(
            {
                "error_code": "invalid_input",
                "message": "Request input is invalid.",
                "unexpected": "value",
            }
        )
    except ValidationError:
        return

    raise AssertionError("ValidationError was expected for unknown fields.")
