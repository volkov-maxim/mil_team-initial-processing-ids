"""Unit tests for driver's license extraction mapping logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.drivers_license_extractor import DriversLicenseExtractor
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_line(
    *,
    text: str,
    y: float,
    confidence: float = 0.9,
) -> RecognizedLine:
    """Build one OCR line fixture with deterministic geometry."""
    token = RecognizedToken(
        text=text,
        polygon=[
            (10.0, y),
            (330.0, y),
            (330.0, y + 22.0),
            (10.0, y + 22.0),
        ],
        bounding_box=(10.0, y, 320.0, 22.0),
        confidence=confidence,
    )

    return RecognizedLine(
        text=text,
        tokens=[token],
        bounding_box=(10.0, y, 320.0, 22.0),
        confidence=confidence,
    )


@pytest.mark.parametrize(
    ("relative_path", "ocr_texts", "expected"),
    [
        (
            "images/drivers_licenses/orig.png",
            [
                "1. МИТИН",
                "2. АНДРЕЙ ВЛАДИМИРОВИЧ",
                "3. 04.09.1984",
                "МОСКВА",
                "4a) 21.07.2016    4b) 21.07.2026",
                "4c) ГИБДД 7723",
                "5. 77 28 089628",
                "8. МОСКВА",
                "9. B",
            ],
            {
                "full_name": "Митин Андрей Владимирович",
                "date_of_birth": "1984-09-04",
                "place_of_birth": "Москва",
                "issue_date": "2016-07-21",
                "expiry_date": "2026-07-21",
                "issuing_authority": "ГИБДД 7723",
                "license_number": "77 28 089628",
                "place_of_residence": "Москва",
                "license_class": "B",
            },
        ),
        (
            "images/drivers_licenses/1385315836_1578903606.jpg",
            [
                "1. ЕЛИЗАРЬЕВ",
                "2. МАКСИМ АНДРЕЕВИЧ",
                "3. 21.04.1995",
                "РЕСПУБЛИКА БАШКОРТОСТАН",
                "4a) 26.06.2013    4b) 26.06.2023",
                "4c) ГИБДД 0275",
                "5. 02 12 172900",
                "8. РЕСПУБЛИКА БАШКОРТОСТАН",
                "9. B",
            ],
            {
                "full_name": "Елизарьев Максим Андреевич",
                "date_of_birth": "1995-04-21",
                "place_of_birth": "Республика Башкортостан",
                "issue_date": "2013-06-26",
                "expiry_date": "2023-06-26",
                "issuing_authority": "ГИБДД 0275",
                "license_number": "02 12 172900",
                "place_of_residence": "Республика Башкортостан",
                "license_class": "B",
            },
        ),
    ],
)
def test_drivers_license_extractor_extracts_fields_from_fixture_ocr_lines(
    relative_path: str,
    ocr_texts: list[str],
    expected: dict[str, str],
) -> None:
    """Extract required driver's-license fields from OCR lines."""
    fixture_path = PROJECT_ROOT / relative_path
    assert fixture_path.exists()

    lines = [
        _build_line(text=text, y=10.0 + (index * 26.0))
        for index, text in enumerate(ocr_texts)
    ]

    extractor = DriversLicenseExtractor()
    result = extractor.extract(lines)

    assert result.full_name == expected["full_name"]
    assert result.date_of_birth == expected["date_of_birth"]
    assert result.place_of_birth == expected["place_of_birth"]
    assert result.issue_date == expected["issue_date"]
    assert result.expiry_date == expected["expiry_date"]
    assert result.issuing_authority == expected["issuing_authority"]
    assert result.license_number == expected["license_number"]
    assert result.place_of_residence == expected["place_of_residence"]
    assert result.license_class == expected["license_class"]


def test_drivers_license_extractor_keeps_required_fields_null_when_missing() -> None:
    """Return explicit nulls for required fields when values are unavailable."""
    lines = [
        _build_line(text="RUS", y=10.0),
        _build_line(text="ВОДИТЕЛЬСКОЕ УДОСТОВЕРЕНИЕ", y=36.0),
    ]

    extractor = DriversLicenseExtractor()
    result = extractor.extract(lines)

    assert result.full_name is None
    assert result.date_of_birth is None
    assert result.place_of_birth is None
    assert result.issue_date is None
    assert result.expiry_date is None
    assert result.issuing_authority is None
    assert result.license_number is None
    assert result.place_of_residence is None
    assert result.license_class is None
    assert result.card_number is None


def test_drivers_license_extractor_formats_compact_license_number() -> None:
    """Normalize compact 10-digit license number to canonical spacing."""
    lines = [
        _build_line(
            text="5. 7728089628",
            y=10.0,
        ),
    ]

    extractor = DriversLicenseExtractor()
    result = extractor.extract(lines)

    assert result.license_number == "77 28 089628"
