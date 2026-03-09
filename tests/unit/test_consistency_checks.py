"""Unit tests for cross-field consistency checks."""

from __future__ import annotations

from app.api.schemas import ExtractedFields
from app.validation.consistency_checks import ConsistencyChecks


def test_generate_flags_returns_empty_list_for_consistent_fields() -> None:
    """Emit no flags when cross-field relationships are consistent."""
    fields = ExtractedFields(
        full_name="Митин Андрей Владимирович",
        date_of_birth="1984-09-04",
        issue_date="2016-07-21",
        expiry_date="2026-07-21",
        license_number="77 28 089628",
    )

    checks = ConsistencyChecks()
    flags = checks.generate_flags(fields)

    assert flags == []


def test_generate_flags_detects_issue_date_after_expiry_date() -> None:
    """Flag inconsistent document validity interval."""
    fields = ExtractedFields(
        issue_date="2027-07-21",
        expiry_date="2026-07-21",
    )

    checks = ConsistencyChecks()
    flags = checks.generate_flags(fields)

    assert "issue_date_after_expiry_date" in flags


def test_generate_flags_detects_birth_date_after_issue_date() -> None:
    """Flag impossible chronology when birth date is after issue date."""
    fields = ExtractedFields(
        date_of_birth="2020-01-01",
        issue_date="2016-07-21",
    )

    checks = ConsistencyChecks()
    flags = checks.generate_flags(fields)

    assert "date_of_birth_after_issue_date" in flags


def test_generate_flags_detects_birth_date_after_expiry_date() -> None:
    """Flag impossible chronology when birth date is after expiry date."""
    fields = ExtractedFields(
        date_of_birth="2030-01-01",
        expiry_date="2026-07-21",
    )

    checks = ConsistencyChecks()
    flags = checks.generate_flags(fields)

    assert "date_of_birth_after_expiry_date" in flags


def test_generate_flags_ignores_invalid_dates_without_crashing() -> None:
    """Skip chronology checks for unparsable dates to keep flags meaningful."""
    fields = ExtractedFields(
        date_of_birth="01.01.2000",
        issue_date="2016-07-21",
        expiry_date="2026-07-21",
    )

    checks = ConsistencyChecks()
    flags = checks.generate_flags(fields)

    assert "date_of_birth_after_issue_date" not in flags
    assert "date_of_birth_after_expiry_date" not in flags
