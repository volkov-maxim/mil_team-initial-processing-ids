"""Unit tests for pipeline context and result models."""

import pytest
from pydantic import ValidationError

from app.api.schemas import DocumentTypeDetected
from app.api.schemas import DocumentTypeHint
from app.pipeline.context import PipelineContext
from app.pipeline.context import PipelineDiagnostic
from app.pipeline.context import PipelineTimings
from app.pipeline.result import PipelineResult


def test_pipeline_context_requires_request_id() -> None:
    """Reject context payloads that omit mandatory request identifiers."""
    with pytest.raises(ValidationError):
        PipelineContext.model_validate({"image_bytes": b"raw-image"})


def test_pipeline_context_has_default_diagnostics_and_timing_containers() -> None:
    """Initialize diagnostics and timing containers for new requests."""
    context = PipelineContext(
        request_id="req-013",
        image_bytes=b"raw-image",
    )

    assert context.document_type_hint is DocumentTypeHint.AUTO
    assert context.use_external_fallback is False
    assert context.diagnostics == []
    assert context.timings.total_ms == 0.0
    assert context.timings.stage_ms == {}


def test_pipeline_result_requires_request_id() -> None:
    """Reject result payloads without a request identifier."""
    with pytest.raises(ValidationError):
        PipelineResult.model_validate(
            {"document_type_detected": DocumentTypeDetected.UNKNOWN}
        )


def test_pipeline_result_exposes_diagnostics_and_timing_containers() -> None:
    """Expose diagnostics and timing containers in result payloads."""
    result = PipelineResult(
        request_id="req-014",
        document_type_detected=DocumentTypeDetected.ID_CARD,
        diagnostics=[
            PipelineDiagnostic(
                stage="ocr",
                code="low_confidence",
                message="OCR confidence is below threshold.",
                severity="warning",
            )
        ],
        timings=PipelineTimings(
            total_ms=120.5,
            stage_ms={"ocr": 33.4},
        ),
    )

    assert result.diagnostics[0].stage == "ocr"
    assert result.diagnostics[0].severity == "warning"
    assert result.timings.total_ms == 120.5
    assert result.timings.stage_ms["ocr"] == 33.4


def test_pipeline_timing_container_rejects_negative_durations() -> None:
    """Reject negative durations in total and per-stage timing values."""
    with pytest.raises(ValidationError):
        PipelineTimings.model_validate({"total_ms": -1.0})

    with pytest.raises(ValidationError):
        PipelineTimings.model_validate({"stage_ms": {"ocr": -0.01}})
