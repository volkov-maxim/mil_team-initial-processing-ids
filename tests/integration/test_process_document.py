"""Integration tests for process-document API route behavior."""

from pathlib import Path

import app.api.routes as api_routes
import app.pipeline.processing as pipeline_processing
import cv2
import numpy as np
from fastapi.testclient import TestClient
import pytest

from app.api.schemas import DocumentTypeDetected
from app.api.schemas import DocumentTypeHint
from app.api.schemas import ExtractedFields
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
from app.pipeline.result import PipelineResult
from app.storage.artifacts import ArtifactStorageManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCUMENT_PATH = (
    PROJECT_ROOT / "images" / "bank_cards" / "bank-cards.jpg"
)
SAMPLE_DRIVER_LICENSE_PATH = (
    PROJECT_ROOT / "images" / "drivers_licenses" / "orig.png"
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


def _build_line(
    *,
    text: str,
    y: float,
    confidence: float = 0.9,
) -> RecognizedLine:
    """Build one OCR line fixture with deterministic geometry."""
    token = RecognizedToken(
        text=text,
        polygon=[
            (10.0, y),
            (330.0, y),
            (330.0, y + 22.0),
            (10.0, y + 22.0),
        ],
        bounding_box=(10.0, y, 320.0, 22.0),
        confidence=confidence,
    )

    return RecognizedLine(
        text=text,
        tokens=[token],
        bounding_box=(10.0, y, 320.0, 22.0),
        confidence=confidence,
    )


def _build_run_ocr_stage_override(
    ocr_texts: list[str],
):
    """Build one stage override that injects deterministic OCR outputs."""
    lines = [
        _build_line(text=text, y=10.0 + (index * 26.0))
        for index, text in enumerate(ocr_texts)
    ]

    detections = [
        {
            "text": line.text,
            "bounding_box": list(line.bounding_box),
            "confidence": line.confidence,
        }
        for line in lines
    ]
    ocr_line_payloads = [line.model_dump(mode="json") for line in lines]

    def _run_ocr_override(context: PipelineContext) -> bool:
        context.stage_outputs["ocr_lines"] = lines
        context.stage_outputs["detections"] = detections
        context.metadata["ocr"] = {
            "regions_count": len(lines),
            "tokens_count": len(lines),
            "lines_count": len(lines),
        }
        context.metadata["ocr_lines"] = ocr_line_payloads
        return True

    return _run_ocr_override


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
    assert payload["document_type_detected"] == "id_card"
    assert payload["aligned_image"] == (
        f"artifacts/{payload['request_id']}/aligned-image.png"
    )
    assert len(payload["detections"]) > 0

    processing_metadata = payload["processing_metadata"]
    aligned_artifact = processing_metadata["aligned_artifact"]
    assert aligned_artifact["path"] == payload["aligned_image"]
    assert aligned_artifact["height"] > 0
    assert aligned_artifact["width"] > 0
    assert aligned_artifact["channels"] in {1, 3}
    assert processing_metadata["executed_stages"] == list(STAGE_SEQUENCE)

    timings = processing_metadata["timings"]
    assert timings["total_ms"] >= 0.0
    for stage_name in STAGE_SEQUENCE:
        assert stage_name in timings["stage_ms"]

    diagnostics = processing_metadata["diagnostics"]
    assert isinstance(diagnostics, list)


def test_process_document_processing_metadata_contains_trace_context(
    monkeypatch,
) -> None:
    """Expose trace metadata fields for device and model provenance."""
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

    trace = payload["processing_metadata"]["trace"]
    assert trace["device"] in {"cpu", "cuda", "auto"}

    model_versions = trace["model_versions"]
    assert "ocr_detector" in model_versions
    assert "ocr_recognizer" in model_versions

    fallback = trace["fallback"]
    assert fallback["requested"] is False
    assert fallback["used"] is False
    assert fallback["status"] == "stub"


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


def test_pipeline_persists_aligned_image_request_artifact(
    tmp_path: Path,
) -> None:
    """Persist aligned image and expose a readable request artifact path."""
    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()
    storage_manager = ArtifactStorageManager(artifacts_root=tmp_path)

    context = PipelineContext(
        request_id="req-aligned-persisted",
        image_bytes=image_bytes,
        metadata={"content_type": "image/jpeg"},
        stage_outputs={
            **_build_stub_ocr_stage_outputs(),
            "artifact_storage_manager": storage_manager,
        },
    )

    result = process_document_pipeline(context)

    assert result.aligned_image is not None
    artifact_path = Path(result.aligned_image)
    assert artifact_path.exists()
    assert artifact_path.is_file()

    persisted_image = cv2.imread(
        str(artifact_path),
        cv2.IMREAD_UNCHANGED,
    )
    assert persisted_image is not None
    assert persisted_image.size > 0

    aligned_artifact = result.processing_metadata["aligned_artifact"]
    assert aligned_artifact["path"] == result.aligned_image


def test_pipeline_persists_detection_overlay_request_artifact(
    tmp_path: Path,
) -> None:
    """Persist detection overlay image and expose a readable artifact path."""
    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()
    storage_manager = ArtifactStorageManager(artifacts_root=tmp_path)

    context = PipelineContext(
        request_id="req-overlay-persisted",
        image_bytes=image_bytes,
        metadata={"content_type": "image/jpeg"},
        stage_outputs={
            **_build_stub_ocr_stage_outputs(),
            "artifact_storage_manager": storage_manager,
        },
    )

    result = process_document_pipeline(context)

    overlay_artifact = result.processing_metadata["overlay_artifact"]
    overlay_path = Path(overlay_artifact["path"])

    assert overlay_path.exists()
    assert overlay_path.is_file()

    overlay_image = cv2.imread(
        str(overlay_path),
        cv2.IMREAD_UNCHANGED,
    )
    assert overlay_image is not None
    assert overlay_image.size > 0

    assert overlay_artifact["height"] > 0
    assert overlay_artifact["width"] > 0
    assert overlay_artifact["channels"] in {1, 3}


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


@pytest.mark.parametrize(
    (
        "ocr_texts",
        "expected_detected",
        "expected_fields",
    ),
    [
        (
            [
                "МИР",
                "2200 0312 3456 7890",
                "IVAN IVANOV",
                "VALID THRU 01/27",
            ],
            "bank_card",
            {
                "card_number": "2200 0312 3456 7890",
                "cardholder_name": "Ivan Ivanov",
                "expiry_date": "2027-01-01",
            },
        ),
        (
            [
                "45 77 695122",
                "ФАМИЛИЯ",
                "КОРОКОВ",
                "ИМЯ",
                "ЛЕОНИД",
                "ОТЧЕСТВО",
                "БОРИСОВИЧ",
                "ПОЛ МУЖ.",
                "ДАТА РОЖДЕНИЯ 19.10.1969",
                "МЕСТО РОЖДЕНИЯ ГОР. МОСКВА",
            ],
            "id_card",
            {
                "full_name": "Короков Леонид Борисович",
                "date_of_birth": "1969-10-19",
                "document_number": "45 77 695122",
            },
        ),
        (
            [
                "1. МИТИН",
                "2. АНДРЕЙ ВЛАДИМИРОВИЧ",
                "3. 04.09.1984",
                "МОСКВА",
                "4a) 21.07.2016    4b) 21.07.2026",
                "4c) ГИБДД 7723",
                "5. 77 28 089628",
                "8. МОСКВА",
                "9. B",
            ],
            "drivers_license",
            {
                "full_name": "Митин Андрей Владимирович",
                "issue_date": "2016-07-21",
                "license_number": "77 28 089628",
            },
        ),
    ],
)
def test_pipeline_extraction_stage_maps_all_document_types(
    ocr_texts: list[str],
    expected_detected: str,
    expected_fields: dict[str, str],
) -> None:
    """Map OCR lines into structured fields for all supported types."""
    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()
    context = PipelineContext(
        request_id=f"req-extraction-{expected_detected}",
        image_bytes=image_bytes,
        document_type_hint=DocumentTypeHint.AUTO,
        metadata={"content_type": "image/jpeg"},
    )

    result = process_document_pipeline(
        context,
        stage_overrides={
            "run_ocr": _build_run_ocr_stage_override(ocr_texts),
        },
    )

    assert result.document_type_detected.value == expected_detected
    extraction_metadata = result.processing_metadata["extraction"]
    assert extraction_metadata["document_type_detected"] == expected_detected
    assert extraction_metadata["non_null_field_count"] >= len(expected_fields)

    fields = result.fields.model_dump()
    for field_name, expected_value in expected_fields.items():
        assert fields[field_name] == expected_value


def test_pipeline_validation_stage_populates_flags_and_confidence() -> None:
    """Populate validation flags and field confidence in pipeline result."""
    image_bytes = SAMPLE_DRIVER_LICENSE_PATH.read_bytes()
    context = PipelineContext(
        request_id="req-validation-populated",
        image_bytes=image_bytes,
        document_type_hint=DocumentTypeHint.AUTO,
        metadata={"content_type": "image/png"},
    )

    ocr_texts = [
        "1. МИТИН",
        "2. АНДРЕЙ ВЛАДИМИРОВИЧ",
        "3. 04.09.1984",
        "МОСКВА",
        "4a) 21.07.2027    4b) 21.07.2026",
        "4c) ГИБДД 7723",
        "5. 77 28 089628",
        "8. МОСКВА",
        "9. B",
    ]

    result = process_document_pipeline(
        context,
        stage_overrides={
            "run_ocr": _build_run_ocr_stage_override(ocr_texts),
        },
    )

    assert "issue_date_after_expiry_date" in result.validation_flags
    assert "issue_date:date_out_of_range" in result.validation_flags

    assert result.field_confidence["full_name"] == pytest.approx(0.85)
    assert result.field_confidence["issue_date"] == pytest.approx(0.65)
    assert result.field_confidence["expiry_date"] == pytest.approx(0.65)

    validation_metadata = result.processing_metadata["validation"]
    assert validation_metadata["flags_count"] >= 2
    assert "issue_date_after_expiry_date" in validation_metadata[
        "validation_flags"
    ]
    assert validation_metadata["aggregate_confidence"] > 0.0


def test_pipeline_partial_extraction_keeps_explicit_null_fields() -> None:
    """Keep missing fields explicit as nulls in pipeline results."""
    image_bytes = SAMPLE_DOCUMENT_PATH.read_bytes()
    context = PipelineContext(
        request_id="req-partial-pipeline",
        image_bytes=image_bytes,
        document_type_hint=DocumentTypeHint.AUTO,
        metadata={"content_type": "image/jpeg"},
    )

    def _extract_partial_fields(stage_context: PipelineContext) -> bool:
        stage_context.stage_outputs["document_type_detected"] = (
            DocumentTypeDetected.BANK_CARD
        )
        stage_context.stage_outputs["extracted_fields"] = {
            "cardholder_name": "Mr. Alioth",
        }
        stage_context.metadata["extraction"] = {
            "document_type_detected": "bank_card",
            "extractor": "PartialExtractorStub",
            "non_null_field_count": 1,
        }
        return True

    result = process_document_pipeline(
        context,
        stage_overrides={
            "run_ocr": _build_run_ocr_stage_override(["MR. ALIOTH"]),
            "extract_fields": _extract_partial_fields,
        },
    )

    field_payload = result.fields.model_dump()
    assert set(field_payload.keys()) == set(ExtractedFields.model_fields.keys())
    assert field_payload["cardholder_name"] == "Mr. Alioth"
    assert field_payload["card_number"] is None
    assert field_payload["expiry_date"] is None


def test_process_document_partial_extraction_returns_null_fields(
    monkeypatch,
) -> None:
    """Return 200 and explicit null fields for partial extraction payloads."""

    def _return_partial_pipeline_result(
        context: PipelineContext,
    ) -> PipelineResult:
        return PipelineResult(
            request_id=context.request_id,
            document_type_detected=DocumentTypeDetected.BANK_CARD,
            aligned_image=f"artifacts/{context.request_id}/aligned-image.png",
            detections=[],
            fields=ExtractedFields(cardholder_name="Mr. Alioth"),
            field_confidence={"cardholder_name": 0.92},
            validation_flags=[],
            processing_metadata={"partial_extraction": True},
        )

    monkeypatch.setattr(
        api_routes,
        "process_document_pipeline",
        _return_partial_pipeline_result,
    )

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
    assert payload["aligned_image"] == (
        f"artifacts/{payload['request_id']}/aligned-image.png"
    )
    assert payload["detections"] == []
    assert payload["field_confidence"] == {"cardholder_name": 0.92}
    assert payload["validation_flags"] == []

    processing_metadata = payload["processing_metadata"]
    assert processing_metadata["partial_extraction"] is True
    assert processing_metadata["timings"] == {
        "total_ms": 0.0,
        "stage_ms": {},
    }
    assert processing_metadata["diagnostics"] == []

    field_payload = payload["fields"]

    assert set(field_payload.keys()) == set(ExtractedFields.model_fields.keys())
    assert field_payload["cardholder_name"] == "Mr. Alioth"
    assert field_payload["card_number"] is None
    assert field_payload["expiry_date"] is None


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
