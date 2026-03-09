"""Unit tests for document type dispatcher routing behavior."""

from __future__ import annotations

import pytest

from app.api.schemas import DocumentTypeHint
from app.extraction.bank_card_extractor import BankCardExtractor
from app.extraction.dispatcher import DocumentTypeDispatcher
from app.extraction.drivers_license_extractor import DriversLicenseExtractor
from app.extraction.id_card_extractor import IdCardExtractor
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken


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
    ("hint", "expected_type"),
    [
        (DocumentTypeHint.BANK_CARD, BankCardExtractor),
        (DocumentTypeHint.ID_CARD, IdCardExtractor),
        (DocumentTypeHint.DRIVERS_LICENSE, DriversLicenseExtractor),
    ],
)
def test_dispatcher_routes_explicit_hint_to_matching_extractor(
    hint: DocumentTypeHint,
    expected_type: type,
) -> None:
    """Resolve explicit hints without invoking auto-detection logic."""
    dispatcher = DocumentTypeDispatcher()

    extractor = dispatcher.resolve_extractor(
        document_type_hint=hint,
        ocr_lines=[_build_line(text="placeholder", y=10.0)],
    )

    assert isinstance(extractor, expected_type)


@pytest.mark.parametrize(
    ("ocr_texts", "expected_type"),
    [
        (
            [
                "МИР",
                "2200 0312 3456 7890",
                "VALID THRU 01/27",
            ],
            BankCardExtractor,
        ),
        (
            [
                "ИМЯ",
                "АНДРЕЙ",
                "ОТЧЕСТВО",
                "ВЛАДИМИРОВИЧ",
                "ПОЛ",
                "МУЖ.",
                "ДАТА РОЖДЕНИЯ 19.10.1969",
            ],
            IdCardExtractor,
        ),
        (
            [
                "1. МИТИН",
                "4a) 21.07.2016    4b) 21.07.2026",
                "4c) ГИБДД 7723",
                "9. B",
            ],
            DriversLicenseExtractor,
        ),
        (
            [
                "SOME RANDOM TEXT",
                "UNCLASSIFIED DOCUMENT",
            ],
            IdCardExtractor,
        ),
    ],
)
def test_dispatcher_routes_auto_hint_across_all_detection_branches(
    ocr_texts: list[str],
    expected_type: type,
) -> None:
    """Resolve ``auto`` hint via bank/id/license/fallback branches."""
    dispatcher = DocumentTypeDispatcher()
    lines = [
        _build_line(text=text, y=10.0 + (index * 26.0))
        for index, text in enumerate(ocr_texts)
    ]

    extractor = dispatcher.resolve_extractor(
        document_type_hint=DocumentTypeHint.AUTO,
        ocr_lines=lines,
    )

    assert isinstance(extractor, expected_type)
