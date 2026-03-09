"""Unit tests for field-level validation helpers."""

from __future__ import annotations

from datetime import date

import pytest

from app.validation.field_validators import FieldValidators


def _build_validators() -> FieldValidators:
    """Create validator with fixed reference date for deterministic tests."""
    return FieldValidators(reference_date=date(2026, 3, 9))


def test_validate_date_plausibility_accepts_valid_birth_date() -> None:
    """Accept realistic birth dates in normalized ISO format."""
    validators = _build_validators()

    result = validators.validate_date_plausibility(
        field_name="date_of_birth",
        value="1990-10-19",
    )

    assert result.is_valid is True
    assert result.error_code is None


@pytest.mark.parametrize(
    "value",
    [
        "19.10.1990",
        "1990-13-01",
        "1990-02-30",
    ],
)
def test_validate_date_plausibility_rejects_invalid_date_values(
    value: str,
) -> None:
    """Reject non-ISO or impossible calendar dates."""
    validators = _build_validators()

    result = validators.validate_date_plausibility(
        field_name="date_of_birth",
        value=value,
    )

    assert result.is_valid is False
    assert result.error_code == "invalid_date_format"


def test_validate_date_plausibility_rejects_future_birth_date() -> None:
    """Reject birth dates that are later than the reference date."""
    validators = _build_validators()

    result = validators.validate_date_plausibility(
        field_name="date_of_birth",
        value="2027-01-01",
    )

    assert result.is_valid is False
    assert result.error_code == "date_out_of_range"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("card_number", "4532 1234 5678 9010"),
        ("card_number", "4532123456789010"),
        ("document_number", "45 77 695122"),
        ("document_number", "4577695122"),
        ("license_number", "77 28 089628"),
        ("license_number", "7728089628"),
    ],
)
def test_validate_number_pattern_accepts_valid_values(
    field_name: str,
    value: str,
) -> None:
    """Accept known number formats produced by extractors."""
    validators = _build_validators()

    result = validators.validate_number_pattern(
        field_name=field_name,
        value=value,
    )

    assert result.is_valid is True
    assert result.error_code is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("card_number", "1234 5678 9012"),
        ("card_number", "4532-1234-5678-901A"),
        ("document_number", "45 777 695122"),
        ("document_number", "AA BB 123456"),
        ("license_number", "77 2808962"),
        ("license_number", "7728-089628"),
    ],
)
def test_validate_number_pattern_rejects_invalid_values(
    field_name: str,
    value: str,
) -> None:
    """Reject malformed numeric identifiers by field-specific patterns."""
    validators = _build_validators()

    result = validators.validate_number_pattern(
        field_name=field_name,
        value=value,
    )

    assert result.is_valid is False
    assert result.error_code == "invalid_number_pattern"


def test_validate_number_pattern_accepts_missing_values() -> None:
    """Treat missing values as non-blocking for partial extraction flow."""
    validators = _build_validators()

    result = validators.validate_number_pattern(
        field_name="document_number",
        value=None,
    )

    assert result.is_valid is True
    assert result.error_code is None
