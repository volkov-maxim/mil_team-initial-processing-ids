"""Field-level validators for normalized extraction outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_MIN_PLAUSIBLE_DATE = date(1900, 1, 1)
_MAX_FUTURE_YEARS = 30

_DATE_FIELDS = frozenset(
    {
        "date_of_birth",
        "issue_date",
        "expiry_date",
    }
)

_NUMBER_PATTERNS: dict[str, re.Pattern[str]] = {
    "card_number": re.compile(
        r"^(?:\d{13,19}|\d{4}(?:[ -]\d{4}){3})$"
    ),
    "document_number": re.compile(r"^(?:\d{10}|\d{2}\s?\d{2}\s?\d{6})$"),
    "license_number": re.compile(r"^(?:\d{10}|\d{2}\s?\d{2}\s?\d{6})$"),
}


@dataclass(frozen=True)
class FieldValidationResult:
    """Validation status for one normalized field value."""

    field_name: str
    value: str | None
    is_valid: bool
    error_code: str | None = None


class FieldValidators:
    """Validate date plausibility and number patterns by field name."""

    def __init__(self, *, reference_date: date | None = None) -> None:
        self._reference_date = reference_date or date.today()

    def validate_date_plausibility(
        self,
        *,
        field_name: str,
        value: str | None,
    ) -> FieldValidationResult:
        """Validate ISO date format and field-specific range plausibility."""
        normalized = _normalize_optional(value)
        if normalized is None:
            return _valid(field_name=field_name, value=None)

        parsed = _parse_iso_date(normalized)
        if parsed is None:
            return _invalid(
                field_name=field_name,
                value=normalized,
                error_code="invalid_date_format",
            )

        min_date, max_date = self._resolve_date_range(field_name)
        if parsed < min_date or parsed > max_date:
            return _invalid(
                field_name=field_name,
                value=normalized,
                error_code="date_out_of_range",
            )

        return _valid(field_name=field_name, value=normalized)

    def validate_number_pattern(
        self,
        *,
        field_name: str,
        value: str | None,
    ) -> FieldValidationResult:
        """Validate number-like fields against schema-aware patterns."""
        normalized = _normalize_optional(value)
        if normalized is None:
            return _valid(field_name=field_name, value=None)

        pattern = _NUMBER_PATTERNS.get(field_name)
        if pattern is None:
            return _valid(field_name=field_name, value=normalized)

        if pattern.fullmatch(normalized) is None:
            return _invalid(
                field_name=field_name,
                value=normalized,
                error_code="invalid_number_pattern",
            )

        return _valid(field_name=field_name, value=normalized)

    def _resolve_date_range(self, field_name: str) -> tuple[date, date]:
        """Resolve plausibility window for one date field."""
        if field_name == "date_of_birth":
            return (_MIN_PLAUSIBLE_DATE, self._reference_date)

        if field_name == "issue_date":
            return (_MIN_PLAUSIBLE_DATE, self._reference_date)

        if field_name == "expiry_date":
            max_expiry = _add_years_safe(self._reference_date, _MAX_FUTURE_YEARS)
            return (_MIN_PLAUSIBLE_DATE, max_expiry)

        if field_name in _DATE_FIELDS:
            max_date = _add_years_safe(self._reference_date, _MAX_FUTURE_YEARS)
            return (_MIN_PLAUSIBLE_DATE, max_date)

        return (_MIN_PLAUSIBLE_DATE, self._reference_date)


def _normalize_optional(value: str | None) -> str | None:
    """Collapse empty and whitespace-only values into ``None``."""
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    return cleaned


def _parse_iso_date(value: str) -> date | None:
    """Parse date in strict ISO format (YYYY-MM-DD)."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _add_years_safe(source: date, years: int) -> date:
    """Add years while preserving leap-day behavior safely."""
    target_year = source.year + years
    try:
        return source.replace(year=target_year)
    except ValueError:
        # Clamp leap-day rollover to the closest valid day.
        return source.replace(year=target_year, day=28)


def _valid(field_name: str, value: str | None) -> FieldValidationResult:
    """Build a successful field validation result."""
    return FieldValidationResult(
        field_name=field_name,
        value=value,
        is_valid=True,
        error_code=None,
    )


def _invalid(
    *,
    field_name: str,
    value: str | None,
    error_code: str,
) -> FieldValidationResult:
    """Build a failed field validation result with machine error code."""
    return FieldValidationResult(
        field_name=field_name,
        value=value,
        is_valid=False,
        error_code=error_code,
    )


__all__ = [
    "FieldValidationResult",
    "FieldValidators",
]
