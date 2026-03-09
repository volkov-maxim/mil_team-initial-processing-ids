"""Application environment configuration models and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field


def _parse_bool(raw_value: str) -> bool:
    """Parse a boolean value from a user-provided environment string."""
    normalized_value = raw_value.strip().lower()
    truthy_values = {"1", "true", "yes", "on"}
    falsy_values = {"0", "false", "no", "off"}

    if normalized_value in truthy_values:
        return True

    if normalized_value in falsy_values:
        return False

    message = f"Invalid boolean value: {raw_value!r}."
    raise ValueError(message)


class AppSettings(BaseModel):
    """Typed runtime settings for the document processing service."""

    device_mode: str = Field(default="auto", pattern="^(cpu|cuda|auto)$")
    gpu_memory_budget_gb: float = Field(default=10.0, gt=0.0)
    use_external_fallback_default: bool = False
    fallback_base_url: str | None = None
    fallback_proxy_url: str | None = None
    artifacts_root: Path = Path("artifacts")
    aligned_subdir: str = "aligned"
    overlay_subdir: str = "overlays"
    fallback_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    required_field_confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AppSettings":
        """Create settings from provided environment variables."""
        source = os.environ if env is None else env
        values: dict[str, Any] = {}

        device_mode = source.get("APP_DEVICE_MODE")
        if device_mode is not None:
            values["device_mode"] = device_mode

        gpu_memory_budget_gb = source.get("APP_GPU_MEMORY_BUDGET_GB")
        if gpu_memory_budget_gb is not None:
            values["gpu_memory_budget_gb"] = float(gpu_memory_budget_gb)

        use_external_fallback_default = source.get(
            "APP_USE_EXTERNAL_FALLBACK_DEFAULT"
        )
        if use_external_fallback_default is not None:
            values["use_external_fallback_default"] = _parse_bool(
                use_external_fallback_default
            )

        fallback_base_url = source.get("APP_FALLBACK_BASE_URL")
        if fallback_base_url is not None:
            values["fallback_base_url"] = fallback_base_url

        fallback_proxy_url = source.get("APP_FALLBACK_PROXY_URL")
        if fallback_proxy_url is not None:
            values["fallback_proxy_url"] = fallback_proxy_url

        artifacts_root = source.get("APP_ARTIFACTS_ROOT")
        if artifacts_root is not None:
            values["artifacts_root"] = Path(artifacts_root)

        aligned_subdir = source.get("APP_ALIGNED_SUBDIR")
        if aligned_subdir is not None:
            values["aligned_subdir"] = aligned_subdir

        overlay_subdir = source.get("APP_OVERLAY_SUBDIR")
        if overlay_subdir is not None:
            values["overlay_subdir"] = overlay_subdir

        fallback_confidence_threshold = source.get(
            "APP_FALLBACK_CONFIDENCE_THRESHOLD"
        )
        if fallback_confidence_threshold is not None:
            values["fallback_confidence_threshold"] = float(
                fallback_confidence_threshold
            )

        required_field_confidence_threshold = source.get(
            "APP_REQUIRED_FIELD_CONFIDENCE_THRESHOLD"
        )
        if required_field_confidence_threshold is not None:
            values["required_field_confidence_threshold"] = float(
                required_field_confidence_threshold
            )

        return cls.model_validate(values)

    @property
    def aligned_artifacts_dir(self) -> Path:
        """Build the request artifact path for aligned image outputs."""
        return self.artifacts_root / self.aligned_subdir

    @property
    def overlay_artifacts_dir(self) -> Path:
        """Build the request artifact path for overlay image outputs."""
        return self.artifacts_root / self.overlay_subdir


def load_settings(env: Mapping[str, str] | None = None) -> AppSettings:
    """Load application settings from environment variables."""
    return AppSettings.from_env(env)