"""Structured logging bootstrap utilities for the application."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from io import TextIOBase
from typing import Any

APP_LOGGER_NAME = "app"

_RESERVED_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def _normalize_log_level(level: str | int) -> int:
    """Convert a string or integer log level into the numeric level value."""
    if isinstance(level, int):
        return level

    normalized_level = level.strip().upper()
    resolved_level = logging.getLevelName(normalized_level)
    if isinstance(resolved_level, int):
        return resolved_level

    message = f"Unsupported log level: {level!r}."
    raise ValueError(message)


def _build_timestamp() -> str:
    """Return an RFC 3339-like UTC timestamp with millisecond precision."""
    timestamp = datetime.now(tz=timezone.utc)
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class StructuredJsonFormatter(logging.Formatter):
    """Serialize log records as JSON with common and context fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record into a structured JSON string."""
        payload: dict[str, Any] = {
            "timestamp": _build_timestamp(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_FIELDS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_configured_app_logger(
    level: str | int = "INFO",
    stream: TextIOBase | None = None,
    logger_name: str = APP_LOGGER_NAME,
) -> logging.Logger:
    """Configure and return an application logger with JSON output."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(_normalize_log_level(level))
    logger.handlers.clear()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredJsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_simple_app_logger(logger_name: str = APP_LOGGER_NAME) -> logging.Logger:
    """Return the configured application logger by name."""
    return logging.getLogger(logger_name)