"""Trace context models for processing metadata provenance."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.core.config import load_settings


class FallbackTraceStub(BaseModel):
    """Stub fallback trace payload until fallback telemetry is implemented."""

    model_config = ConfigDict(extra="forbid")

    requested: bool = False
    used: bool = False
    status: str = "stub"


class TraceContext(BaseModel):
    """Request trace metadata for runtime and model provenance."""

    model_config = ConfigDict(extra="forbid")

    device: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    fallback: FallbackTraceStub = Field(default_factory=FallbackTraceStub)

    @classmethod
    def from_pipeline_context(
        cls,
        *,
        use_external_fallback: bool,
        stage_outputs: Mapping[str, Any],
    ) -> "TraceContext":
        """Build trace metadata from request-scoped pipeline context."""
        settings = load_settings()
        return cls(
            device=settings.device_mode,
            model_versions=_resolve_model_versions(stage_outputs),
            fallback=FallbackTraceStub(requested=use_external_fallback),
        )


def _resolve_model_versions(
    stage_outputs: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve model provenance for OCR detector and recognizer stages."""
    easyocr_version = _resolve_package_version("easyocr")

    detector_component = _resolve_component_name(
        stage_outputs.get("text_detector"),
        default_name="EasyOCRTextDetector",
    )
    recognizer_component = _resolve_component_name(
        stage_outputs.get("text_recognizer"),
        default_name="EasyOCRTextRecognizer",
    )

    return {
        "ocr_detector": (
            f"{detector_component}@easyocr:{easyocr_version}"
        ),
        "ocr_recognizer": (
            f"{recognizer_component}@easyocr:{easyocr_version}"
        ),
    }


def _resolve_component_name(
    component: object,
    *,
    default_name: str,
) -> str:
    """Resolve component class name for trace provenance metadata."""
    if component is None:
        return default_name

    return component.__class__.__name__


def _resolve_package_version(package_name: str) -> str:
    """Resolve installed package version with a stable fallback marker."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not_installed"


__all__ = ["FallbackTraceStub", "TraceContext"]
