"""Bank card field extractor implementation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.extraction.base_extractor import BaseExtractor
from app.extraction.base_extractor import ExtractedFieldsModel
from app.extraction.normalizers import normalize_date
from app.extraction.normalizers import normalize_document_number
from app.extraction.normalizers import normalize_name
from app.extraction.rules_common import cleanup_text
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine

_EXPIRY_HINT_KEYWORDS = (
    "exp",
    "expiry",
    "expires",
    "valid",
    "thru",
    "through",
    "good",
)

_CARD_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{4}(?:[ -]\d{4}){2,3}|\d{13,19})(?!\d)"
)
_EXPIRY_PATTERN = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])[/-](\d{2,4})(?!\d)")
_NAME_TOKEN_PATTERN = re.compile(
    r"^[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*\.?$"
)
_BANK_ACRONYM_PATTERN = re.compile(r"^[A-ZА-ЯЁ]{2,6}$")
_DIGIT_PATTERN = re.compile(r"\d")
_GROUPED_16_PATTERN = re.compile(r"^\d{4}(?:[ -]\d{4}){3}$")

_NETWORK_ALIASES = (
    ("american express", "AMEX"),
    ("master card", "MASTERCARD"),
    ("mastercard", "MASTERCARD"),
    ("unionpay", "UNIONPAY"),
    ("maestro", "MAESTRO"),
    ("visa", "VISA"),
    ("amex", "AMEX"),
    ("мир", "МИР"),
    ("mir", "MIR"),
)
_NETWORK_KEYWORDS = tuple(alias for alias, _ in _NETWORK_ALIASES)
_BANK_KEYWORDS = (
    "bank",
    "бан",
    "sber",
    "сбер",
    "сбербанк",
    "tinkoff",
    "тинькофф",
    "alfa",
    "альфа",
    "vtb",
    "втб",
    "gazprom",
    "газпром",
    "raiffeisen",
    "райффайзен",
    "росбанк",
    "psb",
    "псб",
    "банк",
)
_NON_NAME_KEYWORDS = _NETWORK_KEYWORDS + (
    "bank",
    "бан",
    "issuer",
    "valid",
    "expiry",
    "exp",
    "card",
    "number",
)


class BankCardExtractor(BaseExtractor):
    """Extract bank-card fields from OCR line-level outputs."""

    def extract_from_lines(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        """Extract required and optional fields for bank card schema."""
        return ExtractedFieldsModel(
            card_number=_extract_card_number(ocr_lines),
            cardholder_name=_extract_cardholder_name(ocr_lines),
            expiry_date=_extract_expiry_date(ocr_lines),
            issuer_network=_extract_issuer_network(ocr_lines),
            bank_name=_extract_bank_name(ocr_lines),
        )


def _extract_card_number(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract best card number candidate in 13-19 digit range."""
    best_value: str | None = None
    best_key: tuple[int, int, int, int, int, int] | None = None
    min_y, max_y = _y_bounds(ocr_lines)

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        vertical_ratio = _vertical_ratio(line, min_y=min_y, max_y=max_y)

        for candidate in _iter_card_number_candidates(cleaned):
            normalized = normalize_document_number(candidate).normalized
            digit_count = _count_digits(normalized)
            if digit_count < 13 or digit_count > 19:
                continue

            compact_digits = _digits_only(normalized)
            likely_prefix = compact_digits[:1] in {"2", "3", "4", "5", "6"}

            candidate_key = (
                1 if digit_count == 16 else 0,
                1 if _GROUPED_16_PATTERN.fullmatch(candidate) else 0,
                1 if 0.20 <= vertical_ratio <= 0.80 else 0,
                1 if likely_prefix else 0,
                int(round(line.confidence * 1000.0)),
                -index,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_value = normalized

    return best_value


def _iter_card_number_candidates(text: str) -> Iterable[str]:
    """Yield card-number candidates directly from OCR line content."""
    yielded: set[str] = set()

    for match in _CARD_NUMBER_PATTERN.finditer(text):
        candidate = cleanup_text(match.group(0))
        if candidate in yielded:
            continue
        yielded.add(candidate)
        yield candidate


def _extract_cardholder_name(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract cardholder value using line semantics and card layout."""
    best_value: str | None = None
    best_key: tuple[int, int, int, int] | None = None
    min_y, max_y = _y_bounds(ocr_lines)

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        if not _is_name_like_text(cleaned):
            continue

        if _CARD_NUMBER_PATTERN.search(cleaned) is not None:
            continue

        if _EXPIRY_PATTERN.search(cleaned) is not None:
            continue

        if _contains_any_keyword(cleaned, _NON_NAME_KEYWORDS):
            continue

        vertical_ratio = _vertical_ratio(line, min_y=min_y, max_y=max_y)
        candidate_key = (
            1 if vertical_ratio >= 0.30 else 0,
            1 if _is_uppercase_dominant(cleaned) else 0,
            int(round(line.confidence * 1000.0)),
            -index,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_value = normalize_name(cleaned).normalized

    return best_value


def _extract_expiry_date(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract and normalize card expiry date from date-like patterns."""
    best_value: str | None = None
    best_key: tuple[int, int, int, int] | None = None
    min_y, max_y = _y_bounds(ocr_lines)

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        vertical_ratio = _vertical_ratio(line, min_y=min_y, max_y=max_y)
        has_hint = _contains_any_keyword(cleaned, _EXPIRY_HINT_KEYWORDS)

        for candidate in _iter_expiry_candidates(cleaned):
            normalized = normalize_date(candidate).normalized
            if normalized is None:
                continue

            candidate_key = (
                1 if has_hint else 0,
                1 if vertical_ratio >= 0.20 else 0,
                int(round(line.confidence * 1000.0)),
                -index,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_value = normalized

    return best_value


def _iter_expiry_candidates(text: str) -> Iterable[str]:
    """Yield expiry candidates as ``MM/YY`` or ``MM/YYYY`` strings."""
    yielded: set[str] = set()

    for match in _EXPIRY_PATTERN.finditer(text):
        candidate = f"{match.group(1)}/{match.group(2)}"
        if candidate in yielded:
            continue
        yielded.add(candidate)
        yield candidate


def _extract_issuer_network(ocr_lines: LineRecognitionResult) -> str | None:
    """Detect issuer network from known aliases in OCR lines."""
    best_value: str | None = None
    best_score: tuple[int, int] | None = None

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        lowered = cleaned.casefold()
        for alias, canonical in _NETWORK_ALIASES:
            if not _contains_keyword(lowered, alias):
                continue

            score = (int(round(line.confidence * 1000.0)), -index)
            if best_score is None or score > best_score:
                best_score = score
                best_value = canonical

    return best_value


def _extract_bank_name(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract issuer bank name from semantic line-level evidence."""
    best_value: str | None = None
    best_key: tuple[int, int, int, int] | None = None
    min_y, max_y = _y_bounds(ocr_lines)

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned or _DIGIT_PATTERN.search(cleaned):
            continue

        if _contains_any_keyword(cleaned, _NETWORK_KEYWORDS):
            continue

        keyword_hit = _contains_any_keyword(cleaned, _BANK_KEYWORDS)
        acronym_hit = _is_bank_acronym(cleaned)
        if not keyword_hit and not acronym_hit:
            continue

        # Exclude likely person names unless the line contains explicit bank
        # semantics such as "BANK"/"БАНК".
        if _is_name_like_text(cleaned) and not keyword_hit:
            continue

        vertical_ratio = _vertical_ratio(line, min_y=min_y, max_y=max_y)

        normalized = _normalize_bank_name(cleaned)
        candidate_key = (
            1 if keyword_hit else 0,
            1 if acronym_hit else 0,
            1 if vertical_ratio <= 0.90 else 0,
            int(round(line.confidence * 1000.0)),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_value = normalized

    return best_value


def _contains_keyword(text: str, keyword: str) -> bool:
    """Check keyword match with word boundaries when applicable."""
    escaped = re.escape(keyword.casefold())
    if keyword.isalnum():
        pattern = rf"(?<!\w){escaped}(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    return escaped in text


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether line contains at least one keyword alias."""
    lowered = text.casefold()
    return any(_contains_keyword(lowered, keyword) for keyword in keywords)


def _y_bounds(ocr_lines: LineRecognitionResult) -> tuple[float, float]:
    """Return min top and max bottom coordinates for OCR line layout."""
    if not ocr_lines:
        return 0.0, 1.0

    min_y = min(line.bounding_box[1] for line in ocr_lines)
    max_y = max(line.bounding_box[1] + line.bounding_box[3] for line in ocr_lines)
    if max_y <= min_y:
        return min_y, min_y + 1.0

    return min_y, max_y


def _vertical_ratio(
    line: RecognizedLine,
    *,
    min_y: float,
    max_y: float,
) -> float:
    """Map line vertical center to a normalized 0..1 layout coordinate."""
    center_y = line.bounding_box[1] + (line.bounding_box[3] / 2.0)
    return (center_y - min_y) / (max_y - min_y)


def _count_digits(value: str) -> int:
    """Count decimal digits in normalized numeric text."""
    return sum(character.isdigit() for character in value)


def _digits_only(value: str) -> str:
    """Keep only decimal digits in one string."""
    return "".join(character for character in value if character.isdigit())


def _is_name_like_text(value: str) -> bool:
    """Heuristic: 2-4 alpha tokens without digits for person names."""
    cleaned = cleanup_text(value)
    if not cleaned or _DIGIT_PATTERN.search(cleaned):
        return False

    tokens = cleaned.split(" ")
    if len(tokens) < 2 or len(tokens) > 4:
        return False

    return all(_NAME_TOKEN_PATTERN.fullmatch(token) is not None for token in tokens)


def _is_uppercase_dominant(value: str) -> bool:
    """Return whether most alphabetic characters are uppercase."""
    letters = [character for character in value if character.isalpha()]
    if not letters:
        return False

    uppercase = sum(character.isupper() for character in letters)
    return (uppercase / len(letters)) >= 0.60


def _is_bank_like_text(value: str) -> bool:
    """Return whether text appears to represent a bank issuer name."""
    cleaned = cleanup_text(value)
    if not cleaned:
        return False

    if _contains_any_keyword(cleaned, _BANK_KEYWORDS):
        return True

    return _is_bank_acronym(cleaned)


def _is_bank_acronym(value: str) -> bool:
    """Return whether value looks like a short uppercase bank acronym."""
    compact = value.replace(" ", "")
    return _BANK_ACRONYM_PATTERN.fullmatch(compact) is not None


def _normalize_bank_name(value: str) -> str:
    """Normalize bank name while preserving all-uppercase acronyms."""
    compact = value.replace(" ", "")
    if _BANK_ACRONYM_PATTERN.fullmatch(compact) is not None:
        return value.upper()

    return normalize_name(value).normalized


__all__ = ["BankCardExtractor"]
