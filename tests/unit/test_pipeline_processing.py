"""Unit tests for pipeline orchestration stage ordering semantics."""

from app.pipeline.context import PipelineContext
from app.pipeline.processing import process_document_pipeline


def test_process_document_pipeline_runs_stages_in_strict_order() -> None:
    """Execute all stages in the documented processing order."""
    call_order: list[str] = []

    def record_stage(stage_name: str):
        def _stage(context: PipelineContext) -> bool:
            call_order.append(stage_name)
            return True

        return _stage

    context = PipelineContext(
        request_id="req-stage-order",
        image_bytes=b"image-bytes",
    )

    result = process_document_pipeline(
        context,
        stage_overrides={
            "validate_input": record_stage("validate_input"),
            "align_image": record_stage("align_image"),
            "run_ocr": record_stage("run_ocr"),
            "extract_fields": record_stage("extract_fields"),
            "validate_fields": record_stage("validate_fields"),
            "optional_fallback": record_stage("optional_fallback"),
            "compose_response": record_stage("compose_response"),
        },
    )

    assert call_order == [
        "validate_input",
        "align_image",
        "run_ocr",
        "extract_fields",
        "validate_fields",
        "optional_fallback",
        "compose_response",
    ]
    assert result.processing_metadata["short_circuited"] is False


def test_process_document_pipeline_short_circuits_after_failed_stage() -> None:
    """Stop executing remaining stages when a stage returns False."""
    call_order: list[str] = []

    def _validate_input(context: PipelineContext) -> bool:
        call_order.append("validate_input")
        return True

    def _align_image(context: PipelineContext) -> bool:
        call_order.append("align_image")
        return True

    def _run_ocr(context: PipelineContext) -> bool:
        call_order.append("run_ocr")
        return False

    context = PipelineContext(
        request_id="req-short-circuit",
        image_bytes=b"image-bytes",
    )

    result = process_document_pipeline(
        context,
        stage_overrides={
            "validate_input": _validate_input,
            "align_image": _align_image,
            "run_ocr": _run_ocr,
        },
    )

    assert call_order == ["validate_input", "align_image", "run_ocr"]
    assert result.processing_metadata["short_circuited"] is True
    assert result.processing_metadata["short_circuit_stage"] == "run_ocr"
