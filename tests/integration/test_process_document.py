"""Integration tests for process-document API route behavior."""

from pathlib import Path

import app.api.routes as api_routes
import app.pipeline.processing as pipeline_processing
import numpy as np
from fastapi.testclient import TestClient

from app.core.exceptions import InputValidationError
from app.main import app
from app.ocr.detector import DetectionResult
from app.ocr.detector import TextRegion
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken
from app.ocr.recognizer import TokenRecognitionResult
from app.pipeline.context import PipelineContext
from app.pipeline.processing import STAGE_SEQUENCE
from app.pipeline.processing import process_document_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCUMENT_PATH = (
    PROJECT_ROOT / "images" / "bank_cards" / "bank-cards.jpg"
)


def _build_stub_tokens() -> TokenRecognitionResult:
    """Build deterministic tokens for integration OCR stage tests."""
    return [
        RecognizedToken(
            text="JOHN",
            polygon=[
                (10.0, 8.0),
                (44.0, 8.0),
                (44.0, 28.0),
                (10.0, 28.0),
            ],
            bounding_box=(10.0, 8.0, 34.0, 20.0),
            confidence=0.91,
        ),
        RecognizedToken(
            text="DOE",
            polygon=[
                (48.0, 8.0),
                (90.0, 8.0),
                (90.0, 28.0),
                (48.0, 28.0),
            ],
            bounding_box=(48.0, 8.0, 42.0, 20.0),
            confidence=0.87,
        ),
    ]


class _StubTextDetector:
    def detect(self, aligned_image: np.ndarray) -> DetectionResult:
        assert aligned_image.size > 0
        return [
            TextRegion(
                polygon=[
                    (10.0, 8.0),
                    (90.0, 8.0),
                    (90.0, 28.0),
                    (10.0, 28.0),
                ],
                bounding_box=(10.0, 8.0, 80.0, 20.0),
                confidence=0.92,
            )
        ]


class _StubTextRecognizer:
    def recognize(
        self,
        aligned_image: np.ndarray,
        regions: DetectionResult,
    ) -> TokenRecognitionResult:
        assert aligned_image.size > 0
        assert len(regions) == 1
        return _build_stub_tokens()

    def group_tokens_to_lines(
        self,
        tokens: TokenRecognitionResult,
    ) -> LineRecognitionResult:
        assert len(tokens) == 2
        return [
            RecognizedLine(
                text="JOHN DOE",
                tokens=tokens,
                bounding_box=(10.0, 8.0, 80.0, 20.0),
                confidence=0.89,
            )
        ]


def _build_stub_ocr_stage_outputs() -> dict[str, object]:
    """Build context stage outputs that inject deterministic OCR stubs."""
    return {
        "text_detector": _StubTextDetector(),
        "text_recognizer": _StubTextRecognizer(),
    }


def _patch_stub_ocr_resolvers(monkeypatch) -> None:
    """Patch OCR resolver helpers to avoid optional easyocr dependency."""

    def _resolve_detector(_context: PipelineContext) -> _StubTextDetector:
        return _StubTextDetector()

    def _resolve_recognizer(_context: PipelineContext) -> _StubTextRecognizer:
        return _StubTextRecognizer()

    monkeypatch.setattr(
        pipeline_processing,
        "_resolve_text_detector",
        _resolve_detector,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "_resolve_text_recognizer",
        _resolve_recognizer,
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


def test_process_document_returns_contract_valid_success_payload(
    monkeypatch,
) -> None:
    """Return contract-valid payload with aligned artifact data."""
    _patch_stub_ocr_resolvers(monkeypatch)

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
        stage_outputs=_build_stub_ocr_stage_outputs(),
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


def test_pipeline_result_includes_detections_and_ocr_lines() -> None:
    """Include OCR detections and grouped line outputs in pipeline result."""

    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()
    context = PipelineContext(
        request_id="req-ocr-output",
        image_bytes=image_bytes,
        metadata={"content_type": "image/jpeg"},
        stage_outputs=_build_stub_ocr_stage_outputs(),
    )

    result = process_document_pipeline(context)

    assert len(result.detections) == 2
    assert result.detections[0]["text"] == "JOHN"
    assert result.detections[1]["text"] == "DOE"

    ocr_lines = result.processing_metadata["ocr_lines"]
    assert len(ocr_lines) == 1
    assert ocr_lines[0]["text"] == "JOHN DOE"
    assert ocr_lines[0]["bounding_box"] == [10.0, 8.0, 80.0, 20.0]


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
