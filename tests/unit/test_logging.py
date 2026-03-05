"""Unit tests for structured logging bootstrap configuration."""

import json
from io import StringIO

from app.core.logging import get_configured_app_logger


def test_configure_application_logger_emits_structured_json() -> None:
    """Emit logs with required structured fields for application events."""
    stream = StringIO()
    logger = get_configured_app_logger(
        level="INFO",
        stream=stream,
        logger_name="app.test",
    )

    logger.info("logger initialized", extra={"request_id": "req-001"})
    output_line = stream.getvalue().strip()
    payload = json.loads(output_line)

    assert payload["timestamp"]
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "logger initialized"
    assert payload["request_id"] == "req-001"