"""Normalization utilities for extracted OCR values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

ParseStatus = Literal["success", "invalid"]

_WHITESPACE_PATTERN = re.compile(r"\s+")
_YMD_PATTERN = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
_DMY_PATTERN = re.compile(r"^(\d{1,2})[./](\d{1,2})[./](\d{2,4})$")
_YM_PATTERN = re.compile(r"^(\d{4})[-/](\d{1,2})$")
_MY_PATTERN = re.compile(r"^(\d{1,2})[./](\d{2,4})$")
_TWO_DIGIT_YEAR_CUTOFF = 40


@dataclass(frozen=True)
class DateParseResult:
    """Normalized date value and parse status."""

    normalized: str | None
    parse_status: ParseStatus
    raw_value: str


@dataclass(frozen=True)
class NameNormalizeResult:
    """Normalized full name with preserved raw source value."""

    normalized: str
    raw_value: str


@dataclass(frozen=True)
class DocumentNumberResult:
    """Normalized document number with preserved raw source value."""

    normalized: str
    raw_value: str


def normalize_date(value: str) -> DateParseResult:
    """Normalize a date-like string into ``YYYY-MM-DD`` when possible."""
    raw_value = value
    cleaned = _cleanup(value)
    if not cleaned:
        return DateParseResult(normalized=None, parse_status="invalid", raw_value=raw_value)

    parsed = _parse_ymd(cleaned)
    if parsed is None:
        parsed = _parse_dmy(cleaned)
    if parsed is None:
        parsed = _parse_partial_ym(cleaned)
    if parsed is None:
        parsed = _parse_partial_my(cleaned)

    if parsed is None:
        return DateParseResult(normalized=None, parse_status="invalid", raw_value=raw_value)

    return DateParseResult(normalized=parsed, parse_status="success", raw_value=raw_value)


def normalize_name(value: str) -> NameNormalizeResult:
    """Normalize spacing and casing while keeping name separators."""
    raw_value = value
    cleaned = _cleanup(value)
    if not cleaned:
        return NameNormalizeResult(normalized="", raw_value=raw_value)

    tokens = cleaned.split(" ")
    normalized_tokens = [_normalize_name_token(token) for token in tokens]
    normalized = " ".join(normalized_tokens)
    return NameNormalizeResult(normalized=normalized, raw_value=raw_value)


def normalize_document_number(value: str) -> DocumentNumberResult:
    """Normalize document number text without removing semantic separators."""
    raw_value = value
    cleaned = _cleanup(value)
    normalized = cleaned.upper()
    return DocumentNumberResult(normalized=normalized, raw_value=raw_value)


def _cleanup(value: str) -> str:
    """Collapse whitespace and trim surrounding spaces."""
    normalized = value.replace("\u00a0", " ")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def _parse_ymd(value: str) -> str | None:
    """Parse ``YYYY-MM-DD`` or ``YYYY/MM/DD``."""
    match = _YMD_PATTERN.fullmatch(value)
    if match is None:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    return _to_iso_date(year=year, month=month, day=day)


def _parse_dmy(value: str) -> str | None:
    """Parse ``DD.MM.YYYY`` and ``DD/MM/YYYY`` plus two-digit years."""
    match = _DMY_PATTERN.fullmatch(value)
    if match is None:
        return None

    day = int(match.group(1))
    month = int(match.group(2))
    year = _normalize_year(int(match.group(3)))
    return _to_iso_date(year=year, month=month, day=day)


def _parse_partial_ym(value: str) -> str | None:
    """Parse ``YYYY-MM`` or ``YYYY/MM`` as first day of month."""
    match = _YM_PATTERN.fullmatch(value)
    if match is None:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    return _to_iso_date(year=year, month=month, day=1)


def _parse_partial_my(value: str) -> str | None:
    """Parse ``MM/YY`` or ``MM/YYYY`` as first day of month."""
    match = _MY_PATTERN.fullmatch(value)
    if match is None:
        return None

    month = int(match.group(1))
    year = _normalize_year(int(match.group(2)))
    return _to_iso_date(year=year, month=month, day=1)


def _normalize_year(year: int) -> int:
    """Expand two-digit years with a deterministic century cutoff."""
    if year >= 100:
        return year

    if year <= _TWO_DIGIT_YEAR_CUTOFF:
        return 2000 + year

    return 1900 + year


def _to_iso_date(*, year: int, month: int, day: int) -> str | None:
    """Return an ISO date if components form a valid calendar day."""
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None

    return parsed.isoformat()


def _normalize_name_token(token: str) -> str:
    """Title-case token segments while preserving ``-`` and ``'`` symbols."""
    segments = re.split(r"([-'])", token)
    normalized_segments = [_capitalize_segment(segment) for segment in segments]
    return "".join(normalized_segments)


def _capitalize_segment(segment: str) -> str:
    """Capitalize one segment unless it is a separator."""
    if segment in {"-", "'"}:
        return segment

    if not segment:
        return segment

    return segment[0].upper() + segment[1:].lower()


__all__ = [
    "DateParseResult",
    "DocumentNumberResult",
    "NameNormalizeResult",
    "normalize_date",
    "normalize_document_number",
    "normalize_name",
]
