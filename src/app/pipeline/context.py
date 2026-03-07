"""Pipeline context models for request-scoped processing state."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.api.schemas import DocumentTypeHint

NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class PipelineDiagnostic(BaseModel):
    """Structured diagnostic emitted by a pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    code: str
    message: str
    severity: Literal[
        "info",
        "warning",
        "error",
    ] = "info"
    details: dict[str, Any] | None = None


class PipelineTimings(BaseModel):
    """Container for total and per-stage pipeline durations."""

    model_config = ConfigDict(extra="forbid")

    total_ms: NonNegativeFloat = 0.0
    stage_ms: dict[str, NonNegativeFloat] = Field(default_factory=dict)


class PipelineContext(BaseModel):
    """Request-scoped context shared between pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    image_bytes: bytes
    document_type_hint: DocumentTypeHint = DocumentTypeHint.AUTO
    use_external_fallback: bool = False
    diagnostics: list[PipelineDiagnostic] = Field(default_factory=list)
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    artifacts: dict[str, str] = Field(default_factory=dict)
    stage_outputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "PipelineContext",
    "PipelineDiagnostic",
    "PipelineTimings",
]
