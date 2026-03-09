"""Cross-field consistency checks for extracted document fields."""

from __future__ import annotations

import re
from datetime import date

from app.api.schemas import ExtractedFields

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ConsistencyChecks:
    """Generate validation flags for date chronology consistency."""

    def generate_flags(self, fields: ExtractedFields) -> list[str]:
        """Return cross-field consistency flags in stable order."""
        flags: list[str] = []

        self._check_date_order(flags=flags, fields=fields)

        return flags

    def _check_date_order(
        self,
        *,
        flags: list[str],
        fields: ExtractedFields,
    ) -> None:
        """Validate chronological consistency between related date fields."""
        date_of_birth = _parse_iso_date(fields.date_of_birth)
        issue_date = _parse_iso_date(fields.issue_date)
        expiry_date = _parse_iso_date(fields.expiry_date)

        if issue_date is not None and expiry_date is not None:
            if issue_date > expiry_date:
                _append_flag(flags, "issue_date_after_expiry_date")

        if date_of_birth is not None and issue_date is not None:
            if date_of_birth > issue_date:
                _append_flag(flags, "date_of_birth_after_issue_date")

        if date_of_birth is not None and expiry_date is not None:
            if date_of_birth > expiry_date:
                _append_flag(flags, "date_of_birth_after_expiry_date")


def _parse_iso_date(value: str | None) -> date | None:
    """Parse strict ``YYYY-MM-DD`` date values for chronology checks."""
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    if _ISO_DATE_PATTERN.fullmatch(cleaned) is None:
        return None

    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _append_flag(flags: list[str], flag: str) -> None:
    """Append one flag only once while preserving insertion order."""
    if flag not in flags:
        flags.append(flag)


__all__ = ["ConsistencyChecks"]
