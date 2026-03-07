"""Integration tests for global exception handler mappings."""

from fastapi.testclient import TestClient

from app.core.exceptions import InputValidationError
from app.core.exceptions import InternalProcessingError
from app.core.exceptions import UnprocessableDocumentError
from app.main import app


def _ensure_exception_test_routes() -> None:
    """Register test-only routes for exercising exception handlers."""
    existing_paths = {route.path for route in app.routes}
    if "/_test/errors/input" in existing_paths:
        return

    @app.get("/_test/errors/input")
    def _raise_input_validation_error() -> None:
        raise InputValidationError(message="Unsupported media type.")

    @app.get("/_test/errors/unprocessable")
    def _raise_unprocessable_error() -> None:
        raise UnprocessableDocumentError()

    @app.get("/_test/errors/internal")
    def _raise_internal_processing_error() -> None:
        raise InternalProcessingError(
            message="OCR runtime failed.",
            error_code="ocr_runtime_failure",
        )


_ensure_exception_test_routes()


def test_input_validation_error_maps_to_400_response() -> None:
    """Return HTTP 400 envelope for input-validation exception."""
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/_test/errors/input")

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_input"
    assert response.json()["message"] == "Unsupported media type."


def test_unprocessable_document_error_maps_to_422_response() -> None:
    """Return HTTP 422 envelope for unprocessable document exception."""
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/_test/errors/unprocessable")

    assert response.status_code == 422
    assert response.json()["error_code"] == "unprocessable_document"
    assert response.json()["message"] == (
        "Document image is unreadable or cannot be aligned."
    )


def test_internal_processing_error_maps_to_500_response() -> None:
    """Return HTTP 500 envelope for internal processing exception."""
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/_test/errors/internal")

    assert response.status_code == 500
    assert response.json()["error_code"] == "ocr_runtime_failure"
    assert response.json()["message"] == "OCR runtime failed."
