"""Unit tests for environment-backed application settings."""

from pathlib import Path

from app.core.config import AppSettings


def test_app_settings_defaults() -> None:
    """Return expected defaults when no environment variables are set."""
    settings = AppSettings.from_env({})

    assert settings.device_mode == "auto"
    assert settings.use_external_fallback_default is False
    assert settings.fallback_base_url is None
    assert settings.fallback_proxy_url is None
    assert settings.artifacts_root == Path("artifacts")
    assert settings.aligned_subdir == "aligned"
    assert settings.overlay_subdir == "overlays"
    assert settings.fallback_confidence_threshold == 0.70
    assert settings.required_field_confidence_threshold == 0.80


def test_app_settings_env_overrides() -> None:
    """Override defaults from environment variables with typed parsing."""
    env = {
        "APP_DEVICE_MODE": "cpu",
        "APP_USE_EXTERNAL_FALLBACK_DEFAULT": "true",
        "APP_FALLBACK_BASE_URL": "https://example.test/v1",
        "APP_FALLBACK_PROXY_URL": "http://proxy.test:8080",
        "APP_ARTIFACTS_ROOT": "var/artifacts",
        "APP_ALIGNED_SUBDIR": "aligned-images",
        "APP_OVERLAY_SUBDIR": "detected-overlay",
        "APP_FALLBACK_CONFIDENCE_THRESHOLD": "0.55",
        "APP_REQUIRED_FIELD_CONFIDENCE_THRESHOLD": "0.92",
    }

    settings = AppSettings.from_env(env)

    assert settings.device_mode == "cpu"
    assert settings.use_external_fallback_default is True
    assert settings.fallback_base_url == "https://example.test/v1"
    assert settings.fallback_proxy_url == "http://proxy.test:8080"
    assert settings.artifacts_root == Path("var/artifacts")
    assert settings.aligned_subdir == "aligned-images"
    assert settings.overlay_subdir == "detected-overlay"
    assert settings.fallback_confidence_threshold == 0.55
    assert settings.required_field_confidence_threshold == 0.92