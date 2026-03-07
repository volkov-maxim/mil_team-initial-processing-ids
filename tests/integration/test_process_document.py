"""Integration tests for process-document API route behavior."""

import app.api.routes as api_routes
from fastapi.testclient import TestClient

from app.core.exceptions import InputValidationError
from app.main import app
from app.pipeline.processing import STAGE_SEQUENCE


def _build_multipart_payload() -> tuple[dict[str, tuple], dict[str, str]]:
    """Build a minimal valid multipart payload for route tests."""
    files = {
        "image": (
            "document.png",
            b"fake-image-bytes",
            "image/png",
        )
    }
    data = {
        "document_type_hint": "auto",
        "use_external_fallback": "false",
    }
    return files, data


def test_process_document_returns_contract_valid_placeholder_payload() -> None:
    """Return a contract-valid success payload for a valid request."""
    client = TestClient(app)
    files, data = _build_multipart_payload()

    response = client.post(
        "/v1/process-document",
        files=files,
        data=data,
    )

    assert response.status_code == 200
    payload = response.json()
    expected_keys = {
        "request_id",
        "document_type_detected",
        "aligned_image",
        "detections",
        "fields",
        "field_confidence",
        "validation_flags",
        "processing_metadata",
    }

    assert expected_keys.issubset(payload.keys())
    assert payload["document_type_detected"] == "unknown"
    assert isinstance(payload["aligned_image"], str)
    assert payload["aligned_image"] != ""
    assert payload["processing_metadata"]["executed_stages"] == list(
        STAGE_SEQUENCE
    )


def test_process_document_maps_pipeline_errors_to_typed_envelope(
    monkeypatch,
) -> None:
    """Map pipeline core exceptions to typed API error envelopes."""

    def _raise_input_validation_error(_context) -> None:
        raise InputValidationError(message="Unsupported media type.")

    monkeypatch.setattr(
        api_routes,
        "process_document_pipeline",
        _raise_input_validation_error,
    )

    client = TestClient(app, raise_server_exceptions=False)
    files, data = _build_multipart_payload()

    response = client.post(
        "/v1/process-document",
        files=files,
        data=data,
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "invalid_input"
    assert payload["message"] == "Unsupported media type."
