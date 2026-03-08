"""Integration tests for process-document API route behavior."""

from pathlib import Path

import app.api.routes as api_routes
from fastapi.testclient import TestClient

from app.core.exceptions import InputValidationError
from app.main import app
from app.pipeline.context import PipelineContext
from app.pipeline.processing import STAGE_SEQUENCE
from app.pipeline.processing import process_document_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCUMENT_PATH = (
    PROJECT_ROOT / "images" / "bank_cards" / "bank-cards.jpg"
)


def _build_multipart_payload() -> tuple[dict[str, tuple], dict[str, str]]:
    """Build a minimal valid multipart payload for route tests."""
    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()

    files = {
        "image": (
            "bank-cards.jpg",
            image_bytes,
            "image/jpeg",
        )
    }
    data = {
        "document_type_hint": "auto",
        "use_external_fallback": "false",
    }
    return files, data


def test_process_document_returns_contract_valid_success_payload() -> None:
    """Return contract-valid payload with aligned artifact data."""
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
    assert payload["aligned_image"] == (
        f"artifacts/{payload['request_id']}/aligned-image.png"
    )
    aligned_artifact = payload["processing_metadata"]["aligned_artifact"]
    assert aligned_artifact["path"] == payload["aligned_image"]
    assert aligned_artifact["height"] > 0
    assert aligned_artifact["width"] > 0
    assert aligned_artifact["channels"] in {1, 3}
    assert payload["processing_metadata"]["executed_stages"] == list(
        STAGE_SEQUENCE
    )


def test_pipeline_result_contains_aligned_artifact_data() -> None:
    """Include aligned artifact metadata in direct pipeline output."""
    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()
    context = PipelineContext(
        request_id="req-aligned-artifact",
        image_bytes=image_bytes,
        metadata={"content_type": "image/jpeg"},
    )

    result = process_document_pipeline(context)

    assert result.aligned_image == (
        "artifacts/req-aligned-artifact/aligned-image.png"
    )
    aligned_artifact = result.processing_metadata["aligned_artifact"]
    assert aligned_artifact["path"] == result.aligned_image
    assert aligned_artifact["height"] > 0
    assert aligned_artifact["width"] > 0
    assert aligned_artifact["channels"] in {1, 3}


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
