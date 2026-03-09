"""Unit tests for ID card extraction mapping logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.id_card_extractor import IdCardExtractor
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
            "images/id_cards/passport_min.png",
            [
                "45 77 695122",
                "ФАМИЛИЯ",
                "КОРОКОВ",
                "ИМЯ",
                "ЛЕОНИД",
                "ОТЧЕСТВО",
                "БОРИСОВИЧ",
                "ПОЛ МУЖ",
                "ДАТА РОЖДЕНИЯ 19.10.1969",
                "МЕСТО РОЖДЕНИЯ ГОР. МОСКВА",
                "ПАСПОРТ ВЫДАН ОТДЕЛОМ ВНУТРЕННИХ ДЕЛ",
                "\"ГОЛЬЯНОВО\"",
                "ГОР. МОСКВЫ",
                "ДАТА ВЫДАЧИ 18.02.2000",
                "КОД ПОДРАЗДЕЛЕНИЯ 772-050"
            ],
            {
                "full_name": "Короков Леонид Борисович",
                "date_of_birth": "1969-10-19",
                "sex": "M",
                "place_of_birth": "Гор. Москва",
                "document_number": "45 77 695122",
                "issuing_authority": "Отделом внутренних дел \"Гольяново\" гор. Москвы",
                "issue_date": "2000-02-18",
                "expiry_date": None,
            },
        ),
        (
            "images/id_cards/Pasport_RF.jpg",
            [
                "11 04 000000",
                "ФАМИЛИЯ",
                "ИМЯРЕК",
                "ИМЯ",
                "ЕВГЕНИЙ",
                "ОТЧЕСТВО",
                "АЛЕКСАНДРОВИЧ",
                "ПОЛ",
                "МУЖ.",
                "ДАТА РОЖДЕНИЯ 12.09.1682",
                "МЕСТО РОЖДЕНИЯ ГОР. АРХАНГЕЛЬСК",
            ],
            {
                "full_name": "Имярек Евгений Александрович",
                "date_of_birth": "1682-09-12",
                "sex": "M",
                "place_of_birth": "Гор. Архангельск",
                "document_number": "11 04 000000",
                "issuing_authority": None,
                "issue_date": None,
                "expiry_date": None,
            },
        ),
    ],
)
def test_id_card_extractor_extracts_fields_from_fixture_ocr_lines(
    relative_path: str,
    ocr_texts: list[str],
    expected: dict[str, str | None],
) -> None:
    """Extract required and optional ID-card fields from OCR lines."""
    fixture_path = PROJECT_ROOT / relative_path
    assert fixture_path.exists()

    lines = [
        _build_line(text=text, y=10.0 + (index * 26.0))
        for index, text in enumerate(ocr_texts)
    ]

    extractor = IdCardExtractor()
    result = extractor.extract(lines)

    assert result.full_name == expected["full_name"]
    assert result.date_of_birth == expected["date_of_birth"]
    assert result.sex == expected["sex"]
    assert result.place_of_birth == expected["place_of_birth"]
    assert result.document_number == expected["document_number"]
    assert result.issuing_authority == expected["issuing_authority"]
    assert result.issue_date == expected["issue_date"]
    assert result.expiry_date == expected["expiry_date"]


def test_id_card_extractor_keeps_required_fields_null_when_unavailable() -> None:
    """Return explicit nulls for required fields when not detected."""
    lines = [
        _build_line(text="РОССИЙСКАЯ ФЕДЕРАЦИЯ", y=10.0),
        _build_line(text="ФАМИЛИЯ", y=36.0),
    ]

    extractor = IdCardExtractor()
    result = extractor.extract(lines)

    assert result.full_name is None
    assert result.date_of_birth is None
    assert result.sex is None
    assert result.place_of_birth is None
    assert result.document_number is None
    assert result.issuing_authority is None
    assert result.issue_date is None
    assert result.expiry_date is None
    assert result.card_number is None
