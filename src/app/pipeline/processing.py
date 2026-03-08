"""Pipeline orchestration skeleton with strict stage ordering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter

from app.api.schemas import DocumentTypeDetected
from app.api.schemas import ExtractedFields
from app.core.exceptions import InputValidationError
from app.core.exceptions import UnprocessableDocumentError
from app.pipeline.context import PipelineContext
from app.pipeline.result import PipelineResult
from app.preprocessing.document_preprocessor import DocumentPreprocessor
from app.preprocessing.image_io import decode_image_bytes

StageHandler = Callable[[PipelineContext], bool]

INPUT_VALIDATION_FAILURE_REASONS = frozenset(
    {
        "missing_content_type",
        "unsupported_media_type",
        "file_too_large",
    }
)

STAGE_SEQUENCE: tuple[str, ...] = (
    "validate_input",
    "align_image",
    "run_ocr",
    "extract_fields",
    "validate_fields",
    "optional_fallback",
    "compose_response",
)


def _validate_input(context: PipelineContext) -> bool:
    """Validate request input payload shape and readability constraints."""
    preprocessor = _resolve_preprocessor(context)
    metadata_content_type = context.metadata.get("content_type")
    content_type: str | None
    if isinstance(metadata_content_type, str):
        content_type = metadata_content_type
    else:
        content_type = None

    outcome = preprocessor.validate_type_size_readability(
        context.image_bytes,
        content_type=content_type,
    )
    if outcome.is_valid:
        return True

    details = dict(outcome.details)
    if outcome.failure_reason is not None:
        details["failure_reason"] = outcome.failure_reason

    if outcome.failure_reason in INPUT_VALIDATION_FAILURE_REASONS:
        raise InputValidationError(
            message=outcome.message,
            details=details,
        )

    raise UnprocessableDocumentError(
        message=outcome.message,
        details=details,
    )


def _align_image(context: PipelineContext) -> bool:
    """Align document geometry before OCR and extraction stages."""
    preprocessor = _resolve_preprocessor(context)
    source_image = decode_image_bytes(context.image_bytes)
    aligned_image_array = preprocessor.align_image(source_image)

    context.stage_outputs["aligned_image_array"] = aligned_image_array

    artifact_path = f"artifacts/{context.request_id}/aligned-image.png"
    context.artifacts["aligned_image"] = artifact_path

    height, width = aligned_image_array.shape[:2]
    channels = 1
    if aligned_image_array.ndim == 3:
        channels = int(aligned_image_array.shape[2])

    context.metadata["aligned_artifact"] = {
        "path": artifact_path,
        "height": int(height),
        "width": int(width),
        "channels": channels,
    }

    return True


def _run_ocr(context: PipelineContext) -> bool:
    """Run OCR to generate tokens and line-level text candidates."""
    return True


def _extract_fields(context: PipelineContext) -> bool:
    """Extract schema-aligned fields from OCR outputs."""
    return True


def _validate_fields(context: PipelineContext) -> bool:
    """Validate extracted fields and compute consistency flags."""
    return True


def _optional_fallback(context: PipelineContext) -> bool:
    """Optionally invoke fallback extractor when enabled by policy."""
    return True


def _compose_response(context: PipelineContext) -> bool:
    """Compose final pipeline result artifacts and metadata."""
    return True


DEFAULT_STAGE_HANDLERS: dict[str, StageHandler] = {
    "validate_input": _validate_input,
    "align_image": _align_image,
    "run_ocr": _run_ocr,
    "extract_fields": _extract_fields,
    "validate_fields": _validate_fields,
    "optional_fallback": _optional_fallback,
    "compose_response": _compose_response,
}


def _resolve_stage_handlers(
    stage_overrides: Mapping[str, StageHandler] | None,
) -> dict[str, StageHandler]:
    """Merge default stage handlers with optional test/runtime overrides."""
    handlers = dict(DEFAULT_STAGE_HANDLERS)
    if stage_overrides:
        handlers.update(stage_overrides)
    return handlers


def _resolve_preprocessor(context: PipelineContext) -> DocumentPreprocessor:
    """Resolve a request-scoped preprocessor instance."""
    cached = context.stage_outputs.get("preprocessor")
    if isinstance(cached, DocumentPreprocessor):
        return cached

    preprocessor = DocumentPreprocessor()
    context.stage_outputs["preprocessor"] = preprocessor
    return preprocessor


def _build_result(
    context: PipelineContext,
    *,
    executed_stages: list[str],
    short_circuit_stage: str | None,
) -> PipelineResult:
    """Build a contract-valid pipeline result from current context state."""
    context.metadata["executed_stages"] = executed_stages
    context.metadata["short_circuited"] = short_circuit_stage is not None
    context.metadata["short_circuit_stage"] = short_circuit_stage
    processing_metadata = dict(context.metadata)

    return PipelineResult(
        request_id=context.request_id,
        document_type_detected=DocumentTypeDetected.UNKNOWN,
        aligned_image=context.artifacts.get("aligned_image"),
        fields=ExtractedFields(),
        processing_metadata=processing_metadata,
        diagnostics=context.diagnostics,
        timings=context.timings,
    )


def process_document_pipeline(
    context: PipelineContext,
    *,
    stage_overrides: Mapping[str, StageHandler] | None = None,
) -> PipelineResult:
    """Run pipeline stages in strict contract order with short-circuiting."""
    handlers = _resolve_stage_handlers(stage_overrides)
    executed_stages: list[str] = []
    short_circuit_stage: str | None = None
    start_total = perf_counter()

    for stage_name in STAGE_SEQUENCE:
        stage_handler = handlers[stage_name]

        start_stage = perf_counter()
        should_continue = stage_handler(context)
        context.timings.stage_ms[stage_name] = (
            perf_counter() - start_stage
        ) * 1000.0

        executed_stages.append(stage_name)

        if not should_continue:
            short_circuit_stage = stage_name
            break

    context.timings.total_ms = (perf_counter() - start_total) * 1000.0

    return _build_result(
        context,
        executed_stages=executed_stages,
        short_circuit_stage=short_circuit_stage,
    )


__all__ = ["STAGE_SEQUENCE", "process_document_pipeline"]
