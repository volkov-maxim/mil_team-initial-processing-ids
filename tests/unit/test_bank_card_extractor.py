"""Unit tests for bank card extraction mapping logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.bank_card_extractor import BankCardExtractor
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
            "images/bank_cards/pr-card.png",
            [
                "МИР",
                "2200 0312 3456 7890",
                "IVAN IVANOV",
                "VALID THRU 01/27",
                "ПСБ",
            ],
            {
                "card_number": "2200 0312 3456 7890",
                "cardholder_name": "Ivan Ivanov",
                "expiry_date": "2027-01-01",
                "issuer_network": "МИР",
                "bank_name": "ПСБ",
            },
        ),
        (
            "images/bank_cards/bank-cards.jpg",
            [
                "VISA",
                "1234 5678 9123 4567",
                "MR. ALIOTH",
                "GOOD THRU 12/26",
            ],
            {
                "card_number": "1234 5678 9123 4567",
                "cardholder_name": "Mr. Alioth",
                "expiry_date": "2026-12-01",
                "issuer_network": "VISA",
                "bank_name": None,
            },
        ),
    ],
)
def test_bank_card_extractor_extracts_fields_from_fixture_ocr_lines(
    relative_path: str,
    ocr_texts: list[str],
    expected: dict[str, str],
) -> None:
    """Extract required and optional bank fields from OCR fixture lines."""
    fixture_path = PROJECT_ROOT / relative_path
    assert fixture_path.exists()

    lines = [
        _build_line(text=text, y=10.0 + (index * 26.0))
        for index, text in enumerate(ocr_texts)
    ]

    extractor = BankCardExtractor()
    result = extractor.extract(lines)

    assert result.card_number == expected["card_number"]
    assert result.cardholder_name == expected["cardholder_name"]
    assert result.expiry_date == expected["expiry_date"]
    assert result.issuer_network == expected["issuer_network"]
    assert result.bank_name == expected["bank_name"]


def test_bank_card_extractor_keeps_required_fields_null_when_unavailable() -> None:
    """Return explicit nulls for required fields that are not detected."""
    lines = [
        _build_line(text="VISA", y=10.0),
        _build_line(text="ALFA BANK", y=36.0),
    ]

    extractor = BankCardExtractor()
    result = extractor.extract(lines)

    assert result.card_number is None
    assert result.cardholder_name is None
    assert result.expiry_date is None
    assert result.issuer_network == "VISA"
    assert result.bank_name == "Alfa Bank"
    assert result.full_name is None
