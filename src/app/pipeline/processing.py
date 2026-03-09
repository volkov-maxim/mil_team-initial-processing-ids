"""Pipeline orchestration skeleton with strict stage ordering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

import numpy as np
from app.api.schemas import DocumentTypeDetected
from app.api.schemas import DocumentTypeHint
from app.api.schemas import ExtractedFields
from app.core.exceptions import InputValidationError
from app.core.exceptions import InternalProcessingError
from app.core.exceptions import UnprocessableDocumentError
from app.extraction.dispatcher import DocumentTypeDispatcher
from app.ocr.detector import EasyOCRTextDetector
from app.ocr.detector import TextDetector
from app.ocr.recognizer import EasyOCRTextRecognizer
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import TextRecognizer
from app.ocr.recognizer import TokenRecognitionResult
from app.pipeline.context import PipelineContext
from app.pipeline.result import PipelineResult
from app.preprocessing.document_preprocessor import DocumentPreprocessor
from app.preprocessing.image_io import decode_image_bytes
from app.storage.artifacts import ArtifactStorageManager
from app.validation.confidence import ConfidenceScorer
from app.validation.consistency_checks import ConsistencyChecks
from app.validation.field_validators import FieldValidationResult
from app.validation.field_validators import FieldValidators
from pydantic import ValidationError

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

_DETECTED_TYPE_BY_HINT: dict[DocumentTypeHint, DocumentTypeDetected] = {
    DocumentTypeHint.BANK_CARD: DocumentTypeDetected.BANK_CARD,
    DocumentTypeHint.ID_CARD: DocumentTypeDetected.ID_CARD,
    DocumentTypeHint.DRIVERS_LICENSE: DocumentTypeDetected.DRIVERS_LICENSE,
}

_DATE_VALIDATION_FIELDS: tuple[str, ...] = (
    "date_of_birth",
    "issue_date",
    "expiry_date",
)

_NUMBER_VALIDATION_FIELDS: tuple[str, ...] = (
    "card_number",
    "document_number",
    "license_number",
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
    storage_manager = _resolve_artifact_storage_manager(context)
    source_image = decode_image_bytes(context.image_bytes)
    aligned_image_array = preprocessor.align_image(source_image)

    context.stage_outputs["aligned_image_array"] = aligned_image_array

    aligned_artifact = storage_manager.persist_aligned_image(
        request_id=context.request_id,
        aligned_image=aligned_image_array,
    )

    context.artifacts["aligned_image"] = aligned_artifact.path
    context.metadata["aligned_artifact"] = aligned_artifact.as_metadata()

    return True


def _run_ocr(context: PipelineContext) -> bool:
    """Run OCR to generate tokens and line-level text candidates."""
    aligned_image = context.stage_outputs.get("aligned_image_array")
    if not isinstance(aligned_image, np.ndarray):
        raise InternalProcessingError(
            message="Aligned image is unavailable for OCR stage.",
            error_code="ocr_aligned_image_missing",
            details={"stage": "run_ocr"},
        )

    detector = _resolve_text_detector(context)
    recognizer = _resolve_text_recognizer(context)

    regions = detector.detect(aligned_image)
    tokens = recognizer.recognize(aligned_image, regions)
    lines = recognizer.group_tokens_to_lines(tokens)

    context.stage_outputs["ocr_regions"] = regions
    context.stage_outputs["ocr_tokens"] = tokens
    context.stage_outputs["ocr_lines"] = lines
    context.stage_outputs["detections"] = _serialize_token_detections(tokens)

    context.metadata["ocr"] = {
        "regions_count": len(regions),
        "tokens_count": len(tokens),
        "lines_count": len(lines),
    }
    context.metadata["ocr_lines"] = _serialize_ocr_lines(lines)

    return True


def _extract_fields(context: PipelineContext) -> bool:
    """Extract schema-aligned fields from OCR outputs."""
    raw_ocr_lines = context.stage_outputs.get("ocr_lines")
    if not isinstance(raw_ocr_lines, list):
        raise InternalProcessingError(
            message="OCR lines are unavailable for extraction stage.",
            error_code="extraction_ocr_lines_missing",
            details={"stage": "extract_fields"},
        )

    ocr_lines: LineRecognitionResult = raw_ocr_lines
    dispatcher = _resolve_document_dispatcher(context)
    resolved_document_type = dispatcher.resolve_document_type(
        document_type_hint=context.document_type_hint,
        ocr_lines=ocr_lines,
    )
    extractor = dispatcher.resolve_extractor(
        document_type_hint=context.document_type_hint,
        ocr_lines=ocr_lines,
    )
    extracted_fields = extractor.extract(ocr_lines)

    context.stage_outputs["document_type_detected"] = resolved_document_type
    context.stage_outputs["extracted_fields"] = extracted_fields
    context.metadata["extraction"] = {
        "document_type_detected": resolved_document_type.value,
        "extractor": extractor.__class__.__name__,
        "non_null_field_count": _count_non_null_fields(extracted_fields),
    }

    return True


def _validate_fields(context: PipelineContext) -> bool:
    """Validate extracted fields and compute consistency flags."""
    extracted_fields = _resolve_validation_fields(context)
    detected_type = _resolve_detected_document_type(context)

    field_validators = _resolve_field_validators(context)
    consistency_checks = _resolve_consistency_checks(context)
    confidence_scorer = _resolve_confidence_scorer(context)

    field_validation_flags = _collect_field_validation_flags(
        validators=field_validators,
        fields=extracted_fields,
    )
    consistency_flags = consistency_checks.generate_flags(extracted_fields)
    validation_flags = _merge_validation_flags(
        field_validation_flags,
        consistency_flags,
    )

    confidence_result = confidence_scorer.score(
        fields=extracted_fields,
        document_type=detected_type,
        validation_flags=validation_flags,
    )

    context.stage_outputs["validation_flags"] = validation_flags
    context.stage_outputs["field_confidence"] = (
        confidence_result.field_confidence
    )
    context.metadata["validation"] = {
        "validation_flags": validation_flags,
        "flags_count": len(validation_flags),
        "aggregate_confidence": confidence_result.aggregate_confidence,
    }

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


def _resolve_artifact_storage_manager(
    context: PipelineContext,
) -> ArtifactStorageManager:
    """Resolve a request-scoped artifact storage manager instance."""
    cached = context.stage_outputs.get("artifact_storage_manager")
    if isinstance(cached, ArtifactStorageManager):
        return cached

    manager = ArtifactStorageManager.from_settings()
    context.stage_outputs["artifact_storage_manager"] = manager
    return manager


def _resolve_text_detector(context: PipelineContext) -> TextDetector:
    """Resolve a request-scoped text detector instance."""
    cached = context.stage_outputs.get("text_detector")
    if isinstance(cached, TextDetector):
        return cached

    detector = EasyOCRTextDetector()
    context.stage_outputs["text_detector"] = detector
    return detector


def _resolve_text_recognizer(context: PipelineContext) -> TextRecognizer:
    """Resolve a request-scoped text recognizer instance."""
    cached = context.stage_outputs.get("text_recognizer")
    if isinstance(cached, TextRecognizer):
        return cached

    recognizer = EasyOCRTextRecognizer()
    context.stage_outputs["text_recognizer"] = recognizer
    return recognizer


def _resolve_document_dispatcher(
    context: PipelineContext,
) -> DocumentTypeDispatcher:
    """Resolve a request-scoped document-type dispatcher instance."""
    cached = context.stage_outputs.get("document_dispatcher")
    if isinstance(cached, DocumentTypeDispatcher):
        return cached

    dispatcher = DocumentTypeDispatcher()
    context.stage_outputs["document_dispatcher"] = dispatcher
    return dispatcher


def _resolve_field_validators(context: PipelineContext) -> FieldValidators:
    """Resolve a request-scoped field-validators instance."""
    cached = context.stage_outputs.get("field_validators")
    if isinstance(cached, FieldValidators):
        return cached

    validators = FieldValidators()
    context.stage_outputs["field_validators"] = validators
    return validators


def _resolve_consistency_checks(context: PipelineContext) -> ConsistencyChecks:
    """Resolve a request-scoped consistency-checks instance."""
    cached = context.stage_outputs.get("consistency_checks")
    if isinstance(cached, ConsistencyChecks):
        return cached

    checks = ConsistencyChecks()
    context.stage_outputs["consistency_checks"] = checks
    return checks


def _resolve_confidence_scorer(context: PipelineContext) -> ConfidenceScorer:
    """Resolve a request-scoped confidence scorer instance."""
    cached = context.stage_outputs.get("confidence_scorer")
    if isinstance(cached, ConfidenceScorer):
        return cached

    scorer = ConfidenceScorer()
    context.stage_outputs["confidence_scorer"] = scorer
    return scorer


def _serialize_token_detections(
    tokens: TokenRecognitionResult,
) -> list[dict[str, Any]]:
    """Serialize recognized tokens for pipeline/API detection payloads."""
    return [token.model_dump(mode="json") for token in tokens]


def _serialize_ocr_lines(
    lines: LineRecognitionResult,
) -> list[dict[str, Any]]:
    """Serialize recognized OCR lines for processing metadata output."""
    return [line.model_dump(mode="json") for line in lines]


def _build_result(
    context: PipelineContext,
    *,
    executed_stages: list[str],
    short_circuit_stage: str | None,
) -> PipelineResult:
    """Build a contract-valid pipeline result from current context state."""
    detections: list[dict[str, Any]] = []
    raw_detections = context.stage_outputs.get("detections")
    if isinstance(raw_detections, list):
        detections = [
            dict(item)
            for item in raw_detections
            if isinstance(item, dict)
        ]

    detected_type = _resolve_detected_document_type(context)

    fields = _resolve_result_fields(context.stage_outputs.get("extracted_fields"))

    field_confidence: dict[str, float] = {}
    raw_field_confidence = context.stage_outputs.get("field_confidence")
    if isinstance(raw_field_confidence, Mapping):
        for field_name, score in raw_field_confidence.items():
            if not isinstance(field_name, str):
                continue
            if not isinstance(score, int | float):
                continue
            field_confidence[field_name] = float(score)

    validation_flags: list[str] = []
    raw_validation_flags = context.stage_outputs.get("validation_flags")
    if isinstance(raw_validation_flags, list):
        validation_flags = [
            flag
            for flag in raw_validation_flags
            if isinstance(flag, str)
        ]

    context.metadata["executed_stages"] = executed_stages
    context.metadata["short_circuited"] = short_circuit_stage is not None
    context.metadata["short_circuit_stage"] = short_circuit_stage
    processing_metadata = dict(context.metadata)

    return PipelineResult(
        request_id=context.request_id,
        document_type_detected=detected_type,
        aligned_image=context.artifacts.get("aligned_image"),
        detections=detections,
        fields=fields,
        field_confidence=field_confidence,
        validation_flags=validation_flags,
        processing_metadata=processing_metadata,
        diagnostics=context.diagnostics,
        timings=context.timings,
    )


def _resolve_detected_document_type(
    context: PipelineContext,
) -> DocumentTypeDetected:
    """Resolve detected document type from stage outputs with fallback."""
    raw_detected_type = context.stage_outputs.get("document_type_detected")
    if isinstance(raw_detected_type, DocumentTypeDetected):
        return raw_detected_type

    if isinstance(raw_detected_type, DocumentTypeHint):
        return _DETECTED_TYPE_BY_HINT.get(
            raw_detected_type,
            DocumentTypeDetected.UNKNOWN,
        )

    return DocumentTypeDetected.UNKNOWN


def _count_non_null_fields(extracted_fields: ExtractedFields) -> int:
    """Count extracted fields that have non-null values."""
    payload = extracted_fields.model_dump()
    return sum(value is not None for value in payload.values())


def _resolve_validation_fields(context: PipelineContext) -> ExtractedFields:
    """Resolve extracted fields for validation stage with strict checks."""
    raw_fields = context.stage_outputs.get("extracted_fields")
    if isinstance(raw_fields, ExtractedFields):
        return raw_fields

    if isinstance(raw_fields, Mapping):
        try:
            coerced_fields = ExtractedFields.model_validate(dict(raw_fields))
        except ValidationError as exc:
            raise InternalProcessingError(
                message=(
                    "Extracted fields payload is invalid for validation stage."
                ),
                error_code="validation_fields_invalid",
                details={
                    "stage": "validate_fields",
                    "errors": exc.errors(),
                },
            ) from exc

        context.stage_outputs["extracted_fields"] = coerced_fields
        return coerced_fields

    raise InternalProcessingError(
        message="Extracted fields are unavailable for validation stage.",
        error_code="validation_fields_missing",
        details={"stage": "validate_fields"},
    )


def _resolve_result_fields(raw_fields: object) -> ExtractedFields:
    """Resolve result fields and preserve explicit-null defaults."""
    if isinstance(raw_fields, ExtractedFields):
        return raw_fields

    if isinstance(raw_fields, Mapping):
        try:
            return ExtractedFields.model_validate(dict(raw_fields))
        except ValidationError:
            return ExtractedFields()

    return ExtractedFields()


def _collect_field_validation_flags(
    *,
    validators: FieldValidators,
    fields: ExtractedFields,
) -> list[str]:
    """Collect deterministic field-level validation error flags."""
    flags: list[str] = []

    for field_name in _DATE_VALIDATION_FIELDS:
        value = getattr(fields, field_name)
        result = validators.validate_date_plausibility(
            field_name=field_name,
            value=value,
        )
        _append_field_validation_flag(flags=flags, result=result)

    for field_name in _NUMBER_VALIDATION_FIELDS:
        value = getattr(fields, field_name)
        result = validators.validate_number_pattern(
            field_name=field_name,
            value=value,
        )
        _append_field_validation_flag(flags=flags, result=result)

    return flags


def _append_field_validation_flag(
    *,
    flags: list[str],
    result: FieldValidationResult,
) -> None:
    """Append one field-level validation flag when validation fails."""
    if result.is_valid or result.error_code is None:
        return

    flag = f"{result.field_name}:{result.error_code}"
    if flag not in flags:
        flags.append(flag)


def _merge_validation_flags(*groups: list[str]) -> list[str]:
    """Merge flag groups while preserving first-seen order."""
    merged: list[str] = []
    for group in groups:
        for flag in group:
            if flag in merged:
                continue
            merged.append(flag)

    return merged


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
