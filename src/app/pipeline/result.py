"""Pipeline result models for normalized output and diagnostics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.api.schemas import DocumentTypeDetected
from app.api.schemas import ExtractedFields
from app.pipeline.context import PipelineDiagnostic
from app.pipeline.context import PipelineTimings


class PipelineResult(BaseModel):
    """Final output from the processing pipeline before API serialization."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    document_type_detected: DocumentTypeDetected
    aligned_image: str | None = None
    detections: list[dict[str, Any]] = Field(default_factory=list)
    fields: ExtractedFields = Field(default_factory=ExtractedFields)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    validation_flags: list[str] = Field(default_factory=list)
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[PipelineDiagnostic] = Field(default_factory=list)
    timings: PipelineTimings = Field(default_factory=PipelineTimings)


__all__ = ["PipelineResult"]
