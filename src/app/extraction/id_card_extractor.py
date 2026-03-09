"""ID card (Russian passport) field extractor implementation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.extraction.base_extractor import BaseExtractor
from app.extraction.base_extractor import ExtractedFieldsModel
from app.extraction.normalizers import normalize_date
from app.extraction.normalizers import normalize_document_number
from app.extraction.normalizers import normalize_name
from app.extraction.rules_common import cleanup_text
from app.extraction.rules_common import extract_value_after_label
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine

_SURNAME_LABELS = ("фамилия",)
_NAME_LABELS = ("имя",)
_PATRONYMIC_LABELS = ("отчество",)
_DATE_OF_BIRTH_LABELS = ("дата рождения", "рождения")
_SEX_LABELS = ("пол",)
_PLACE_OF_BIRTH_LABELS = ("место рождения", "рождения")
_ISSUING_AUTHORITY_LABELS = ("паспорт выдан", "выдан")
_ISSUE_DATE_LABELS = ("дата выдачи",)
_EXPIRY_DATE_LABELS = ("дата окончания срока действия", "date of expiry")

_DATE_PATTERN = re.compile(r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\d)")
_DOCUMENT_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{2}\s?\d{2}\s?\d{6}|\d{10})(?!\d)"
)
_DIGITS_ONLY_PATTERN = re.compile(r"^[\d\s]+$")
_DIGIT_PATTERN = re.compile(r"\d")
_NAME_TOKEN_PATTERN = re.compile(
    r"^[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*\.?$"
)
_MALE_PATTERN = re.compile(r"\b(муж|male|m)\b", flags=re.IGNORECASE)
_FEMALE_PATTERN = re.compile(r"\b(жен|female|f)\b", flags=re.IGNORECASE)

_STOP_LABEL_HINTS = (
    "дата выдачи",
    "дата рождения",
    "место рождения",
    "фамилия",
    "имя",
    "отчество",
)
_NAME_EXCLUSION_HINTS = (
    "паспорт",
    "рождения",
    "выдан",
    "дата",
    "место",
    "российская",
    "федерация",
)
_DOCUMENT_NUMBER_EXCLUSION_HINTS = (
    "код",
    "подраздел",
    "дата",
    "рожд",
    "выдан",
    "место",
    "пол",
)


class IdCardExtractor(BaseExtractor):
    """Extract Russian passport fields from OCR line-level outputs."""

    def extract_from_lines(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        """Extract required and optional fields for ID card schema."""
        return ExtractedFieldsModel(
            full_name=_extract_full_name(ocr_lines),
            date_of_birth=_extract_date_of_birth(ocr_lines),
            sex=_extract_sex(ocr_lines),
            place_of_birth=_extract_place_of_birth(ocr_lines),
            document_number=_extract_document_number(ocr_lines),
            issuing_authority=_extract_issuing_authority(ocr_lines),
            issue_date=_extract_issue_date(ocr_lines),
            expiry_date=_extract_expiry_date(ocr_lines),
        )


def _extract_document_number(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract passport number using shape, semantics, and layout."""
    best_value: str | None = None
    best_key: tuple[int, int, int, int, int, int] | None = None
    min_y, max_y = _y_bounds(ocr_lines)

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        vertical_ratio = _vertical_ratio(line, min_y=min_y, max_y=max_y)
        numeric_only_line = _DIGITS_ONLY_PATTERN.fullmatch(cleaned) is not None
        has_exclusion_hint = _contains_any_label(
            cleaned,
            _DOCUMENT_NUMBER_EXCLUSION_HINTS,
        )

        for candidate in _iter_document_number_candidates(cleaned):
            normalized = normalize_document_number(candidate).normalized
            digit_count = _count_digits(normalized)
            if digit_count != 10:
                continue

            formatted = _format_passport_number(normalized)
            has_space_layout = " " in formatted
            covers_line = len(candidate) >= (len(cleaned) * 0.60)

            candidate_key = (
                1 if has_space_layout else 0,
                1 if numeric_only_line else 0,
                1 if covers_line else 0,
                1 if vertical_ratio <= 0.80 else 0,
                0 if has_exclusion_hint else 1,
                -index,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_value = formatted

    return best_value


def _iter_document_number_candidates(text: str) -> Iterable[str]:
    """Yield document number candidates from one cleaned OCR line."""
    yielded: set[str] = set()

    for match in _DOCUMENT_NUMBER_PATTERN.finditer(text):
        candidate = cleanup_text(match.group(0))
        if candidate in yielded:
            continue

        yielded.add(candidate)
        yield candidate


def _extract_full_name(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract full name from labeled parts or name-like fallback lines."""
    surname = _extract_person_name_part(ocr_lines, _SURNAME_LABELS)
    given_name = _extract_person_name_part(ocr_lines, _NAME_LABELS)
    patronymic = _extract_person_name_part(ocr_lines, _PATRONYMIC_LABELS)

    if surname is not None and given_name is not None:
        parts = [surname, given_name]
        if patronymic is not None:
            parts.append(patronymic)
        return normalize_name(" ".join(parts)).normalized

    best_name: tuple[int, str] | None = None
    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not _is_full_name_like_text(cleaned):
            continue

        if _contains_any_label(cleaned, _NAME_EXCLUSION_HINTS):
            continue

        candidate = normalize_name(cleaned).normalized
        key = (int(round(line.confidence * 1000.0)), -index)
        if best_name is None or key > (best_name[0], 0):
            best_name = (key[0], candidate)

    if best_name is None:
        return None

    return best_name[1]


def _extract_person_name_part(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> str | None:
    """Extract one name part (surname/name/patronymic) from label context."""
    line_values = _extract_labeled_values(ocr_lines, labels)
    for value in line_values:
        if _is_name_part(value):
            return normalize_name(value).normalized

    return None


def _extract_date_of_birth(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract date of birth from labeled lines or date-only fallback."""
    labeled = _extract_labeled_date(ocr_lines, _DATE_OF_BIRTH_LABELS)
    if labeled is not None:
        return labeled

    best_value: str | None = None
    best_index: int | None = None
    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        if "рожд" not in cleaned.casefold():
            continue

        parsed = _extract_first_date(cleaned)
        if parsed is None:
            continue

        if best_index is None or index < best_index:
            best_index = index
            best_value = parsed

    return best_value


def _extract_issue_date(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract issue date from issue-date labeled lines."""
    return _extract_labeled_date(ocr_lines, _ISSUE_DATE_LABELS)


def _extract_expiry_date(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract optional ID expiry date when explicit marker exists."""
    return _extract_labeled_date(ocr_lines, _EXPIRY_DATE_LABELS)


def _extract_labeled_date(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> str | None:
    """Extract first parseable date from label-tail or neighbor lines."""
    for value in _extract_labeled_values(ocr_lines, labels):
        parsed = _extract_first_date(value)
        if parsed is not None:
            return parsed

    return None


def _extract_sex(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract canonical sex marker: ``M`` or ``F``."""
    for line in ocr_lines:
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        value = extract_value_after_label(cleaned, _SEX_LABELS) or cleaned
        lowered = value.casefold()

        if _FEMALE_PATTERN.search(lowered) is not None:
            return "F"

        if _MALE_PATTERN.search(lowered) is not None:
            return "M"

    return None


def _extract_place_of_birth(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract place of birth from labeled line values."""
    for value in _extract_labeled_values(ocr_lines, _PLACE_OF_BIRTH_LABELS):
        if _DIGIT_PATTERN.search(value) is not None:
            continue

        cleaned = _strip_trailing_date_fragment(value)
        if not cleaned:
            continue

        return normalize_name(cleaned).normalized

    return None


def _extract_issuing_authority(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract issuing authority text from label context."""
    cleaned_lines = [cleanup_text(line.text) for line in ocr_lines]
    for index, cleaned in enumerate(cleaned_lines):
        if not cleaned:
            continue

        if not _contains_any_label(cleaned, _ISSUING_AUTHORITY_LABELS):
            continue

        parts: list[str] = []
        tail = extract_value_after_label(cleaned, _ISSUING_AUTHORITY_LABELS)
        if tail is None:
            tail = _extract_value_after_label_flexible(
                cleaned,
                _ISSUING_AUTHORITY_LABELS,
            )

        if tail:
            parts.append(tail)

        for next_index in range(index + 1, len(cleaned_lines)):
            next_value = cleaned_lines[next_index]
            if not next_value:
                continue

            truncated = _truncate_at_stop_label(next_value)
            if truncated is not None:
                if truncated:
                    parts.append(truncated)
                break

            if _contains_any_label(next_value, _ISSUING_AUTHORITY_LABELS):
                break

            parts.append(next_value)

        authority_raw = cleanup_text(" ".join(parts))
        authority_raw = _strip_trailing_date_fragment(authority_raw)
        authority_raw = _truncate_at_stop_label(authority_raw) or authority_raw
        authority_raw = cleanup_text(authority_raw)
        if not authority_raw:
            continue

        return _normalize_authority_text(authority_raw)

    return None


def _extract_labeled_values(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> list[str]:
    """Collect candidate values from label tails and next lines."""
    values: list[str] = []
    allow_flexible = any(" " in label for label in labels)

    cleaned_lines = [cleanup_text(line.text) for line in ocr_lines]
    for index, cleaned in enumerate(cleaned_lines):
        if not cleaned:
            continue

        value = extract_value_after_label(cleaned, labels)
        if value is not None:
            values.append(value)
            continue

        if allow_flexible:
            flexible_value = _extract_value_after_label_flexible(
                cleaned,
                labels,
            )
            if flexible_value is not None:
                values.append(flexible_value)
                continue

        if not _contains_any_label(cleaned, labels):
            continue

        next_index = index + 1
        if next_index >= len(cleaned_lines):
            continue

        next_value = cleanup_text(cleaned_lines[next_index])
        if next_value:
            values.append(next_value)

    return values


def _extract_first_date(value: str) -> str | None:
    """Extract first date-like token and normalize to ISO form."""
    for match in _DATE_PATTERN.finditer(value):
        parsed = normalize_date(match.group(0)).normalized
        if parsed is not None:
            return parsed

    return None


def _contains_any_label(text: str, labels: tuple[str, ...]) -> bool:
    """Return whether text includes at least one semantic label token."""
    lowered = text.casefold()
    for label in labels:
        lowered_label = label.casefold()
        escaped = re.escape(lowered_label)
        if any(character.isalnum() for character in lowered_label):
            pattern = rf"(?<!\w){escaped}(?!\w)"
            if re.search(pattern, lowered, flags=re.IGNORECASE) is not None:
                return True
            continue

        if lowered_label in lowered:
            return True

    return False


def _is_name_part(value: str) -> bool:
    """Return whether value looks like one person-name token."""
    cleaned = cleanup_text(value)
    if not cleaned or _DIGIT_PATTERN.search(cleaned):
        return False

    tokens = cleaned.split(" ")
    if len(tokens) != 1:
        return False

    return _NAME_TOKEN_PATTERN.fullmatch(tokens[0]) is not None


def _is_full_name_like_text(value: str) -> bool:
    """Return whether value resembles a full name in one line."""
    cleaned = cleanup_text(value)
    if not cleaned or _DIGIT_PATTERN.search(cleaned):
        return False

    tokens = cleaned.split(" ")
    if len(tokens) < 2 or len(tokens) > 3:
        return False

    return all(_NAME_TOKEN_PATTERN.fullmatch(token) is not None for token in tokens)


def _strip_trailing_date_fragment(value: str) -> str:
    """Remove date substrings from mixed semantic lines."""
    cleaned = cleanup_text(value)
    cleaned = _DATE_PATTERN.sub("", cleaned)
    return cleanup_text(cleaned)


def _extract_value_after_label_flexible(
    text: str,
    labels: tuple[str, ...],
) -> str | None:
    """Extract tail after label even when OCR misses word boundaries."""
    lowered = text.casefold()
    for label in labels:
        lowered_label = label.casefold()
        search_from = 0

        while True:
            start = lowered.find(lowered_label, search_from)
            if start == -1:
                break

            end = start + len(lowered_label)
            is_single_word = " " not in lowered_label
            if is_single_word:
                prev_ok = start == 0 or not lowered[start - 1].isalnum()
                next_ok = end >= len(lowered) or not lowered[end].isalnum()
                if not prev_ok or not next_ok:
                    search_from = start + 1
                    continue

            if end >= len(text):
                break

            tail = text[end:].strip(" :-)\t")
            tail = cleanup_text(tail)
            if tail:
                return tail

            search_from = start + 1

    return None


def _truncate_at_stop_label(value: str) -> str | None:
    """Return prefix before stop labels, supporting glued OCR tokens."""
    lowered = value.casefold()
    stop_positions: list[int] = []
    for label in _STOP_LABEL_HINTS:
        pos = lowered.find(label.casefold())
        if pos > 0:
            stop_positions.append(pos)

    if not stop_positions:
        return None

    return cleanup_text(value[: min(stop_positions)])


def _normalize_authority_text(value: str) -> str:
    """Normalize authority while preserving common lowercase function words."""
    normalized = normalize_name(value).normalized
    tokens = normalized.split(" ")
    if not tokens:
        return normalized

    lowered_tokens: list[str] = []
    for index, token in enumerate(tokens):
        token_lower = token.lower()
        if index == 0:
            lowered_tokens.append(token)
            continue

        if token_lower in {"внутренних", "дел", "гор."}:
            lowered_tokens.append(token_lower)
            continue

        lowered_tokens.append(token)

    normalized_text = " ".join(lowered_tokens)
    return re.sub(
        r'"([A-Za-zА-Яа-яЁё])([A-Za-zА-Яа-яЁё]*)"',
        lambda match: (
            f'"{match.group(1).upper()}{match.group(2).lower()}"'
        ),
        normalized_text,
    )


def _count_digits(value: str) -> int:
    """Count decimal digits in one text value."""
    return sum(character.isdigit() for character in value)


def _digits_only(value: str) -> str:
    """Return one compact string containing only decimal digits."""
    return "".join(character for character in value if character.isdigit())


def _format_passport_number(value: str) -> str:
    """Format 10-digit passport number as ``XX XX XXXXXX``."""
    digits = _digits_only(value)
    if len(digits) != 10:
        return cleanup_text(value)

    return f"{digits[:2]} {digits[2:4]} {digits[4:]}"


def _y_bounds(ocr_lines: LineRecognitionResult) -> tuple[float, float]:
    """Return min top and max bottom coordinates across OCR lines."""
    if not ocr_lines:
        return 0.0, 1.0

    min_y = min(line.bounding_box[1] for line in ocr_lines)
    max_y = max(
        line.bounding_box[1] + line.bounding_box[3]
        for line in ocr_lines
    )
    if max_y <= min_y:
        return min_y, min_y + 1.0

    return min_y, max_y


def _vertical_ratio(
    line: RecognizedLine,
    *,
    min_y: float,
    max_y: float,
) -> float:
    """Return normalized vertical center position in 0..1 range."""
    center_y = line.bounding_box[1] + (line.bounding_box[3] / 2.0)
    return (center_y - min_y) / (max_y - min_y)


__all__ = ["IdCardExtractor"]
