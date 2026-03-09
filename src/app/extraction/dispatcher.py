"""Document-type dispatcher for extractor selection."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.api.schemas import DocumentTypeHint
from app.extraction.bank_card_extractor import BankCardExtractor
from app.extraction.base_extractor import BaseExtractor
from app.extraction.drivers_license_extractor import DriversLicenseExtractor
from app.extraction.id_card_extractor import IdCardExtractor
from app.extraction.rules_common import cleanup_text
from app.ocr.recognizer import LineRecognitionResult

_CARD_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{4}(?:[ -]\d{4}){2,3}|\d{13,19})(?!\d)"
)
_CARD_EXPIRY_PATTERN = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])[/-](\d{2,4})(?!\d)")
_PASSPORT_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{2}\s?\d{2}\s?\d{6}|\d{10})(?!\d)"
)
_NUMBERED_LICENSE_FIELD_PATTERN = re.compile(
    r"^\s*(?:[1-9][a-zа-я]?|4[abc])[.)]?\s+",
    flags=re.IGNORECASE,
)

_BANK_KEYWORDS = (
    "visa",
    "mastercard",
    "master card",
    "mir",
    "мир",
    "valid thru",
    "good thru",
    "expiry",
    "card",
    "карта",
)
_ID_KEYWORDS = (
    "паспорт",
    "российская федерация",
    "фамилия",
    "имя",
    "отчество",
    "дата рождения",
    "пол",
    "код подраздел",
    "место рождения",
)
_DRIVERS_KEYWORDS = (
    "водитель",
    "удостовер",
    "гибдд",
    "категор",
    "driver",
    "license",
    "licence",
)

_RESOLVED_HINTS = (
    DocumentTypeHint.BANK_CARD,
    DocumentTypeHint.ID_CARD,
    DocumentTypeHint.DRIVERS_LICENSE,
)
_AUTO_PRIORITY = (
    DocumentTypeHint.BANK_CARD,
    DocumentTypeHint.DRIVERS_LICENSE,
    DocumentTypeHint.ID_CARD,
)


class DocumentTypeDispatcher:
    """Resolve document extractors by explicit hint or OCR auto-detection."""

    def __init__(
        self,
        *,
        bank_card_extractor: BaseExtractor | None = None,
        id_card_extractor: BaseExtractor | None = None,
        drivers_license_extractor: BaseExtractor | None = None,
    ) -> None:
        """Initialize dispatcher with default or injected extractors."""
        self._extractors: dict[DocumentTypeHint, BaseExtractor] = {
            DocumentTypeHint.BANK_CARD: (
                bank_card_extractor
                if bank_card_extractor is not None
                else BankCardExtractor()
            ),
            DocumentTypeHint.ID_CARD: (
                id_card_extractor
                if id_card_extractor is not None
                else IdCardExtractor()
            ),
            DocumentTypeHint.DRIVERS_LICENSE: (
                drivers_license_extractor
                if drivers_license_extractor is not None
                else DriversLicenseExtractor()
            ),
        }

    def resolve_document_type(
        self,
        *,
        document_type_hint: DocumentTypeHint,
        ocr_lines: LineRecognitionResult,
    ) -> DocumentTypeHint:
        """Resolve one concrete document type from hint and OCR text lines."""
        if document_type_hint in _RESOLVED_HINTS:
            return document_type_hint

        return _detect_document_type(ocr_lines)

    def resolve_extractor(
        self,
        *,
        document_type_hint: DocumentTypeHint,
        ocr_lines: LineRecognitionResult,
    ) -> BaseExtractor:
        """Resolve extractor implementation for hint or auto-detected type."""
        resolved_type = self.resolve_document_type(
            document_type_hint=document_type_hint,
            ocr_lines=ocr_lines,
        )
        return self._extractors[resolved_type]


def _detect_document_type(ocr_lines: LineRecognitionResult) -> DocumentTypeHint:
    """Detect document type from OCR lines using heuristic scoring."""
    cleaned_lines = _cleaned_lines(ocr_lines)
    if not cleaned_lines:
        return DocumentTypeHint.ID_CARD

    scores = {
        DocumentTypeHint.BANK_CARD: _score_bank_card(cleaned_lines),
        DocumentTypeHint.ID_CARD: _score_id_card(cleaned_lines),
        DocumentTypeHint.DRIVERS_LICENSE: _score_drivers_license(
            cleaned_lines,
        ),
    }

    best_score = max(scores.values())
    if best_score <= 0:
        return DocumentTypeHint.ID_CARD

    for candidate in _AUTO_PRIORITY:
        if scores[candidate] == best_score:
            return candidate

    return DocumentTypeHint.ID_CARD


def _cleaned_lines(ocr_lines: LineRecognitionResult) -> list[str]:
    """Normalize OCR line text values for dispatcher scoring."""
    return [
        value
        for value in (cleanup_text(line.text).casefold() for line in ocr_lines)
        if value
    ]


def _score_bank_card(cleaned_lines: Iterable[str]) -> int:
    """Score bank-card likelihood from card-specific lexical patterns."""
    score = 0
    for line in cleaned_lines:
        if _CARD_NUMBER_PATTERN.search(line) is not None:
            score += 6

        if _CARD_EXPIRY_PATTERN.search(line) is not None:
            score += 4

        score += _keyword_score(line, _BANK_KEYWORDS, weight=2)

    return score


def _score_id_card(cleaned_lines: Iterable[str]) -> int:
    """Score ID-card likelihood from passport semantics and labels."""
    score = 0
    for line in cleaned_lines:
        if _PASSPORT_NUMBER_PATTERN.search(line) is not None:
            score += 2

        score += _keyword_score(line, _ID_KEYWORDS, weight=2)

    return score


def _score_drivers_license(cleaned_lines: Iterable[str]) -> int:
    """Score driver's-license likelihood from license-specific markers."""
    score = 0
    for line in cleaned_lines:
        if _NUMBERED_LICENSE_FIELD_PATTERN.match(line) is not None:
            score += 3

        if _PASSPORT_NUMBER_PATTERN.search(line) is not None:
            score += 1

        score += _keyword_score(line, _DRIVERS_KEYWORDS, weight=3)

    return score


def _keyword_score(line: str, keywords: tuple[str, ...], *, weight: int) -> int:
    """Return weighted score for keyword hits in one normalized line."""
    hits = sum(1 for keyword in keywords if keyword in line)
    return hits * weight


__all__ = ["DocumentTypeDispatcher"]
