"""Unit tests for extraction normalization utilities."""

from __future__ import annotations

import pytest

from app.extraction.normalizers import normalize_date
from app.extraction.normalizers import normalize_document_number
from app.extraction.normalizers import normalize_name
from app.extraction.normalizers import DateParseResult
from app.extraction.normalizers import NameNormalizeResult
from app.extraction.normalizers import DocumentNumberResult


class TestNormalizeDate:
    """Tests for date normalization with parse status."""

    def test_normalize_iso_date_with_hyphens(self) -> None:
        """Parse ISO date format YYYY-MM-DD."""
        result = normalize_date("2000-05-15")

        assert result.normalized == "2000-05-15"
        assert result.parse_status == "success"
        assert result.raw_value == "2000-05-15"

    def test_normalize_date_with_dots(self) -> None:
        """Parse DD.MM.YYYY format and convert to ISO."""
        result = normalize_date("19.10.1969")

        assert result.normalized == "1969-10-19"
        assert result.parse_status == "success"

    def test_normalize_date_with_slashes(self) -> None:
        """Parse DD/MM/YYYY format and convert to ISO."""
        result = normalize_date("25/12/1995")

        assert result.normalized == "1995-12-25"
        assert result.parse_status == "success"

    def test_normalize_date_with_zero_padding(self) -> None:
        """Preserve zero-padded dates in DD.MM.YYYY format."""
        result = normalize_date("01.01.2020")

        assert result.normalized == "2020-01-01"
        assert result.parse_status == "success"

    def test_normalize_date_without_zero_padding(self) -> None:
        """Parse dates without zero-padding."""
        result = normalize_date("1.2.2022")

        assert result.normalized == "2022-02-01"
        assert result.parse_status == "success"

    def test_normalize_date_with_two_digit_year(self) -> None:
        """Parse DD.MM.YY format with two-digit year."""
        result = normalize_date("15.03.99")

        assert result.normalized == "1999-03-15"
        assert result.parse_status == "success"

        result = normalize_date("15.03.21")
        assert result.normalized == "2021-03-15"
        assert result.parse_status == "success"

    def test_normalize_date_rejects_invalid_month(self) -> None:
        """Return failure status for invalid month."""
        result = normalize_date("32.13.2000")

        assert result.normalized is None
        assert result.parse_status == "invalid"
        assert result.raw_value == "32.13.2000"

    def test_normalize_date_rejects_invalid_day(self) -> None:
        """Return failure status for invalid day."""
        result = normalize_date("32.01.2000")

        assert result.normalized is None
        assert result.parse_status == "invalid"

    def test_normalize_date_handles_empty_input(self) -> None:
        """Return failure status for empty input."""
        result = normalize_date("")

        assert result.normalized is None
        assert result.parse_status == "invalid"
        assert result.raw_value == ""

    def test_normalize_date_handles_whitespace_padding(self) -> None:
        """Strip and parse dates with surrounding whitespace."""
        result = normalize_date("  19.10.1969  ")

        assert result.normalized == "1969-10-19"
        assert result.parse_status == "success"

    def test_normalize_date_rejects_non_date_text(self) -> None:
        """Return failure status for non-date text."""
        result = normalize_date("ABC.DEF.GHIJ")

        assert result.normalized is None
        assert result.parse_status == "invalid"
        assert result.raw_value == "ABC.DEF.GHIJ"

    def test_normalize_date_handles_partial_date(self) -> None:
        """Parse YYYY-MM-DD format for partial date patterns."""
        result = normalize_date("2000-05")

        assert result.normalized == "2000-05-01"
        assert result.parse_status == "success"

    def test_normalize_date_handles_partial_date_with_slashes(self) -> None:
        """Parse DD/MM/YY format with two-digit year for partial date patterns."""
        result = normalize_date("12/95")
        assert result.normalized == "1995-12-01"
        assert result.parse_status == "success"

        result = normalize_date("12/26")
        assert result.normalized == "2026-12-01"
        assert result.parse_status == "success"


class TestNormalizeName:
    """Tests for name normalization with spacing and casing."""

    def test_normalize_name_with_proper_case(self) -> None:
        """Normalize name with each word capitalized."""
        result = normalize_name("ИВАНОВ ИВАН ИВАНОВИЧ")

        assert result.normalized == "Иванов Иван Иванович"
        assert result.raw_value == "ИВАНОВ ИВАН ИВАНОВИЧ"

    def test_normalize_name_preserves_hyphens(self) -> None:
        """Keep hyphenated name structure intact."""
        result = normalize_name("СИДОРОВ-ПЕТРОВ АЛЕКСЕЙ")

        assert result.normalized == "Сидоров-Петров Алексей"
        assert result.raw_value == "СИДОРОВ-ПЕТРОВ АЛЕКСЕЙ"

    def test_normalize_name_collapses_multiple_spaces(self) -> None:
        """Collapse repeated whitespace into single spaces."""
        result = normalize_name("ИВАНОВ   ИВАН    ИВАНОВИЧ")

        assert result.normalized == "Иванов Иван Иванович"
        assert result.raw_value == "ИВАНОВ   ИВАН    ИВАНОВИЧ"

    def test_normalize_name_strips_leading_trailing_whitespace(self) -> None:
        """Strip edges and normalize internal spacing."""
        result = normalize_name("  ПЕТРОВ ПЕТР  ")

        assert result.normalized == "Петров Петр"
        assert result.raw_value == "  ПЕТРОВ ПЕТР  "

    def test_normalize_name_handles_lowercase_input(self) -> None:
        """Title-case lowercase names."""
        result = normalize_name("иванов иван иванович")

        assert result.normalized == "Иванов Иван Иванович"
        assert result.raw_value == "иванов иван иванович"

    def test_normalize_name_handles_mixed_case_input(self) -> None:
        """Normalize mixed-case names to title case."""
        result = normalize_name("иВаНоВ ИвАн")

        assert result.normalized == "Иванов Иван"
        assert result.raw_value == "иВаНоВ ИвАн"

    def test_normalize_name_handles_latin_names(self) -> None:
        """Normalize Latin script names."""
        result = normalize_name("SMITH JOHN DAVID")

        assert result.normalized == "Smith John David"
        assert result.raw_value == "SMITH JOHN DAVID"

    def test_normalize_name_handles_empty_input(self) -> None:
        """Return empty normalized value for empty input."""
        result = normalize_name("")

        assert result.normalized == ""
        assert result.raw_value == ""

    def test_normalize_name_handles_single_word(self) -> None:
        """Normalize single-word names."""
        result = normalize_name("МАРИЯ")

        assert result.normalized == "Мария"
        assert result.raw_value == "МАРИЯ"

    def test_normalize_name_preserves_apostrophes(self) -> None:
        """Keep apostrophes in names like O'Brien."""
        result = normalize_name("O'BRIEN SEAN")

        assert result.normalized == "O'Brien Sean"
        assert result.raw_value == "O'BRIEN SEAN"


class TestNormalizeDocumentNumber:
    """Tests for document number normalization."""

    def test_normalize_document_number_preserves_hyphens(self) -> None:
        """Preserve already-normalized document numbers."""
        result = normalize_document_number("123-456")

        assert result.normalized == "123-456"
        assert result.raw_value == "123-456"

    def test_normalize_document_number_converts_to_uppercase(self) -> None:
        """Convert alphanumeric document numbers to uppercase."""
        result = normalize_document_number("abc123 def")

        assert result.normalized == "ABC123 DEF"
        assert result.raw_value == "abc123 def"

    def test_normalize_document_number_strips_leading_trailing_space(self) -> None:
        """Strip edge whitespace before normalizing."""
        result = normalize_document_number("  1234567890  ")

        assert result.normalized == "1234567890"
        assert result.raw_value == "  1234567890  "

    def test_normalize_document_number_preserves_valid_format_for_russian_documents(self) -> None:
        """Preserve already-normalized document numbers."""
        result = normalize_document_number("12 34 567890")

        assert result.normalized == "12 34 567890"
        assert result.raw_value == "12 34 567890"

    def test_normalize_document_number_handles_empty_input(self) -> None:
        """Return empty normalized value for empty input."""
        result = normalize_document_number("")

        assert result.normalized == ""
        assert result.raw_value == ""

    def test_normalize_document_number_handles_bank_card_format(self) -> None:
        """Preserve already-normalized bank card number format with spaces."""
        result = normalize_document_number("4532 1234 5678 9010")

        assert result.normalized == "4532 1234 5678 9010"
        assert result.raw_value == "4532 1234 5678 9010"