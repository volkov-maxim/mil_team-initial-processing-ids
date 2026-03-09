"""Driver's license field extractor implementation."""

from __future__ import annotations

import re

from app.extraction.base_extractor import BaseExtractor
from app.extraction.base_extractor import ExtractedFieldsModel
from app.extraction.normalizers import normalize_date
from app.extraction.normalizers import normalize_document_number
from app.extraction.normalizers import normalize_name
from app.extraction.rules_common import cleanup_text
from app.extraction.rules_common import extract_value_after_label
from app.ocr.recognizer import LineRecognitionResult

_SURNAME_LABELS = ("фамилия", "surname")
_NAME_LABELS = ("имя", "name")
_PATRONYMIC_LABELS = ("отчество", "patronymic")
_DATE_OF_BIRTH_LABELS = ("дата рождения", "birth date")
_PLACE_OF_BIRTH_LABELS = ("место рождения", "birth place")
_ISSUE_DATE_LABELS = ("дата выдачи", "date of issue")
_EXPIRY_DATE_LABELS = (
    "действительно до",
    "дата окончания срока действия",
    "expiry date",
    "date of expiry",
    "valid until",
)
_ISSUING_AUTHORITY_LABELS = ("кем выдано", "выдан", "issued by")
_LICENSE_NUMBER_LABELS = (
    "номер водительского удостоверения",
    "номер удостоверения",
    "license no",
    "licence no",
    "license number",
)
_PLACE_OF_RESIDENCE_LABELS = (
    "место жительства",
    "место проживания",
    "place of residence",
    "residence",
    "address",
)
_LICENSE_CLASS_LABELS = ("категории", "категория", "categories", "category")

_DATE_PATTERN = re.compile(r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\d)")
_LICENSE_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{2}\s?\d{2}\s?\d{6}|\d{10})(?!\d)"
)
_DIGIT_PATTERN = re.compile(r"\d")
_NAME_TOKEN_PATTERN = re.compile(
    r"^[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*\.?$"
)
_AUTHORITY_ACRONYM_PATTERN = re.compile(r"^[A-ZА-ЯЁ]{2,8}$")

_NUMBERED_SURNAME_PATTERN = re.compile(r"^\s*1[.)]?\s*(.+)$")
_NUMBERED_NAME_PATTERN = re.compile(r"^\s*2[.)]?\s*(.+)$")
_NUMBERED_DOB_PATTERN = re.compile(r"^\s*3[.)]?\s*(.+)$")
_NUMBERED_ISSUE_EXPIRY_PATTERN = re.compile(
    r"4a\)?\s*([0-9./-]{6,10}).*?4b\)?\s*([0-9./-]{6,10})",
    flags=re.IGNORECASE,
)
_NUMBERED_AUTHORITY_PATTERN = re.compile(r"^\s*4c[.)]?\s*(.+)$")
_NUMBERED_LICENSE_NUMBER_PATTERN = re.compile(r"^\s*5[.)]?\s*(.+)$")
_NUMBERED_RESIDENCE_PATTERN = re.compile(r"^\s*8[.)]?\s*(.+)$")
_NUMBERED_CLASS_PATTERN = re.compile(r"^\s*9[.)]?\s*(.+)$")
_FIELD_MARKER_PATTERN = re.compile(r"^\s*\d+[A-Za-zА-Яа-я]?[.)]?\s*")

_NAME_EXCLUSION_HINTS = (
    "дата",
    "выдан",
    "номер",
    "катег",
    "место",
    "license",
    "birth",
    "issue",
    "expiry",
    "issued",
    "водитель",
    "водительское",
    "удостовер",
    "удостоверение",
    "driver",
    "licence",
    "republic",
    "республика",
    "область",
    "край",
    "город",
)
_AUTHORITY_HINTS = ("гибдд", "gibdd", "мрэо", "mreo")
_NON_AUTHORITY_HINTS = (
    "дата",
    "рожд",
    "birth",
    "issue",
    "expiry",
    "номер",
    "license",
    "катег",
    "residence",
    "address",
)

_CYRILLIC_TO_LATIN_CLASS_MAP = {
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "У": "Y",
    "Х": "X",
}


class DriversLicenseExtractor(BaseExtractor):
    """Extract driver's-license fields from OCR line-level outputs."""

    def extract_from_lines(
        self,
        ocr_lines: LineRecognitionResult,
    ) -> ExtractedFieldsModel:
        """Extract required driver's-license fields from OCR lines."""
        return ExtractedFieldsModel(
            full_name=_extract_full_name(ocr_lines),
            date_of_birth=_extract_date_of_birth(ocr_lines),
            place_of_birth=_extract_place_of_birth(ocr_lines),
            issue_date=_extract_issue_date(ocr_lines),
            expiry_date=_extract_expiry_date(ocr_lines),
            issuing_authority=_extract_issuing_authority(ocr_lines),
            license_number=_extract_license_number(ocr_lines),
            place_of_residence=_extract_place_of_residence(ocr_lines),
            license_class=_extract_license_class(ocr_lines),
        )


def _extract_full_name(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract full name from labeled name parts or fallback line text."""
    numbered_surname = _extract_numbered_value(
        ocr_lines,
        _NUMBERED_SURNAME_PATTERN,
    )
    numbered_name = _extract_numbered_value(
        ocr_lines,
        _NUMBERED_NAME_PATTERN,
    )
    if numbered_surname is not None and numbered_name is not None:
        raw_full_name = f"{numbered_surname} {numbered_name}"
        return normalize_name(raw_full_name).normalized

    surname = _extract_person_name_part(ocr_lines, _SURNAME_LABELS)
    given_name = _extract_person_name_part(ocr_lines, _NAME_LABELS)
    patronymic = _extract_person_name_part(ocr_lines, _PATRONYMIC_LABELS)

    if surname is not None and given_name is not None:
        parts = [surname, given_name]
        if patronymic is not None:
            parts.append(patronymic)
        return normalize_name(" ".join(parts)).normalized

    best_value: str | None = None
    best_key: tuple[int, int] | None = None

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not _is_full_name_like_text(cleaned):
            continue

        if _contains_any_label(cleaned, _NAME_EXCLUSION_HINTS):
            continue

        candidate = normalize_name(cleaned).normalized
        candidate_key = (int(round(line.confidence * 1000.0)), -index)

        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_value = candidate

    return best_value


def _extract_person_name_part(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> str | None:
    """Extract one name token from lines associated with semantic labels."""
    for value in _extract_labeled_values(ocr_lines, labels):
        if _is_name_part(value):
            return normalize_name(value).normalized

    return None


def _extract_date_of_birth(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract date of birth from labeled or birth-hint lines."""
    numbered = _extract_numbered_value(ocr_lines, _NUMBERED_DOB_PATTERN)
    if numbered is not None:
        parsed_numbered = _extract_first_date(numbered)
        if parsed_numbered is not None:
            return parsed_numbered

    labeled_value = _extract_labeled_date(ocr_lines, _DATE_OF_BIRTH_LABELS)
    if labeled_value is not None:
        return labeled_value

    for line in ocr_lines:
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        lowered = cleaned.casefold()
        if "рожд" not in lowered and "birth" not in lowered:
            continue

        parsed = _extract_first_date(cleaned)
        if parsed is not None:
            return parsed

    return None


def _extract_issue_date(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract issue date from date-of-issue label context."""
    numbered_pair = _extract_numbered_issue_expiry_dates(ocr_lines)
    if numbered_pair is not None and numbered_pair[0] is not None:
        return numbered_pair[0]

    return _extract_labeled_date(ocr_lines, _ISSUE_DATE_LABELS)


def _extract_expiry_date(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract expiry date from validity/expiry label context."""
    numbered_pair = _extract_numbered_issue_expiry_dates(ocr_lines)
    if numbered_pair is not None and numbered_pair[1] is not None:
        return numbered_pair[1]

    return _extract_labeled_date(ocr_lines, _EXPIRY_DATE_LABELS)


def _extract_place_of_birth(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract place of birth from labeled place-of-birth lines."""
    numbered_place = _extract_numbered_place_of_birth(ocr_lines)
    if numbered_place is not None:
        return numbered_place

    return _extract_place_value(ocr_lines, _PLACE_OF_BIRTH_LABELS)


def _extract_place_of_residence(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract place of residence from labeled residence lines."""
    numbered_residence = _extract_numbered_value(
        ocr_lines,
        _NUMBERED_RESIDENCE_PATTERN,
    )
    if numbered_residence is not None:
        cleaned_numbered = _strip_trailing_date_fragment(numbered_residence)
        if cleaned_numbered:
            return normalize_name(cleaned_numbered).normalized

    return _extract_place_value(ocr_lines, _PLACE_OF_RESIDENCE_LABELS)


def _extract_place_value(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> str | None:
    """Extract a normalized place string from semantic label context."""
    for value in _extract_labeled_values(ocr_lines, labels):
        cleaned = _strip_trailing_date_fragment(value)
        if not cleaned:
            continue

        return normalize_name(cleaned).normalized

    return None


def _extract_issuing_authority(
    ocr_lines: LineRecognitionResult,
) -> str | None:
    """Extract issuing authority from labels or authority-specific hints."""
    numbered_authority = _extract_numbered_value(
        ocr_lines,
        _NUMBERED_AUTHORITY_PATTERN,
    )
    if numbered_authority is not None:
        return _normalize_authority_text(numbered_authority)

    for value in _extract_labeled_values(ocr_lines, _ISSUING_AUTHORITY_LABELS):
        cleaned = _strip_trailing_date_fragment(value)
        if not cleaned:
            continue

        if _contains_any_label(cleaned, _NON_AUTHORITY_HINTS):
            continue

        return _normalize_authority_text(cleaned)

    best_value: str | None = None
    best_key: tuple[int, int] | None = None
    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        lowered = cleaned.casefold()
        if not any(hint in lowered for hint in _AUTHORITY_HINTS):
            continue

        if _contains_any_label(cleaned, _NON_AUTHORITY_HINTS):
            continue

        candidate = _normalize_authority_text(_strip_leading_field_marker(cleaned))
        candidate_key = (int(round(line.confidence * 1000.0)), -index)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_value = candidate

    return best_value


def _extract_license_number(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract driver's-license number using label and format scoring."""
    numbered_value = _extract_numbered_value(
        ocr_lines,
        _NUMBERED_LICENSE_NUMBER_PATTERN,
    )
    if numbered_value is not None:
        for candidate in _iter_license_number_candidates(numbered_value):
            normalized = normalize_document_number(candidate).normalized
            if _count_digits(normalized) == 10:
                return _format_license_number(normalized)

    best_value: str | None = None
    best_key: tuple[int, int, int, int] | None = None

    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        has_label = _contains_any_label(cleaned, _LICENSE_NUMBER_LABELS)
        has_exclusion_hint = _contains_any_label(cleaned, _NAME_EXCLUSION_HINTS)

        for candidate in _iter_license_number_candidates(cleaned):
            normalized = normalize_document_number(candidate).normalized
            digit_count = _count_digits(normalized)
            if digit_count != 10:
                continue

            formatted = _format_license_number(normalized)
            candidate_key = (
                1 if has_label else 0,
                0 if has_exclusion_hint else 1,
                int(round(line.confidence * 1000.0)),
                -index,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_value = formatted

    return best_value


def _iter_license_number_candidates(text: str) -> list[str]:
    """Yield de-duplicated license number candidates from one OCR line."""
    yielded: set[str] = set()
    candidates: list[str] = []

    for match in _LICENSE_NUMBER_PATTERN.finditer(text):
        candidate = cleanup_text(match.group(0))
        if candidate in yielded:
            continue

        yielded.add(candidate)
        candidates.append(candidate)

    return candidates


def _extract_license_class(ocr_lines: LineRecognitionResult) -> str | None:
    """Extract class categories and normalize as comma-separated tokens."""
    numbered_classes = _extract_numbered_value(
        ocr_lines,
        _NUMBERED_CLASS_PATTERN,
    )
    if numbered_classes is not None:
        normalized_numbered = _normalize_license_classes(numbered_classes)
        if normalized_numbered is not None:
            return normalized_numbered

    best_value: str | None = None
    best_key: tuple[int, int, int] | None = None

    cleaned_lines = [cleanup_text(line.text) for line in ocr_lines]
    for index, cleaned in enumerate(cleaned_lines):
        if not cleaned:
            continue

        values: list[str] = []
        tail = extract_value_after_label(cleaned, _LICENSE_CLASS_LABELS)
        if tail is not None:
            values.append(tail)

        if not values:
            flexible = _extract_value_after_label_flexible(
                cleaned,
                _LICENSE_CLASS_LABELS,
            )
            if flexible is not None:
                values.append(flexible)

        if not values and _contains_any_label(cleaned, _LICENSE_CLASS_LABELS):
            next_index = index + 1
            if next_index < len(cleaned_lines) and cleaned_lines[next_index]:
                values.append(cleaned_lines[next_index])

        for value in values:
            normalized = _normalize_license_classes(value)
            if normalized is None:
                continue

            class_count = normalized.count(",") + 1
            candidate_key = (
                class_count,
                int(round(ocr_lines[index].confidence * 1000.0)),
                -index,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_value = normalized

    return best_value


def _normalize_license_classes(value: str) -> str | None:
    """Normalize one class string into ``X``/``X1`` comma-separated form."""
    cleaned = cleanup_text(value)
    if not cleaned:
        return None

    raw_tokens = re.split(r"[,;/\s]+", cleaned)
    classes: list[str] = []
    seen: set[str] = set()

    for raw_token in raw_tokens:
        token = raw_token.strip().upper()
        if not token:
            continue

        normalized_token = _normalize_license_class_token(token)
        if normalized_token is None:
            continue

        if normalized_token in seen:
            continue

        seen.add(normalized_token)
        classes.append(normalized_token)

    if not classes:
        return None

    return ", ".join(classes)


def _normalize_license_class_token(token: str) -> str | None:
    """Normalize one class token and map cyrillic look-alike symbols."""
    translated = "".join(_CYRILLIC_TO_LATIN_CLASS_MAP.get(ch, ch) for ch in token)
    if re.fullmatch(r"[A-Z]\d?", translated) is None:
        return None

    return translated


def _extract_labeled_date(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> str | None:
    """Extract first parseable date from label-tail or neighboring lines."""
    for value in _extract_labeled_values(ocr_lines, labels):
        parsed = _extract_first_date(value)
        if parsed is not None:
            return parsed

    return None


def _extract_numbered_issue_expiry_dates(
    ocr_lines: LineRecognitionResult,
) -> tuple[str | None, str | None] | None:
    """Extract issue/expiry pair from ``4a`` and ``4b`` one-line layout."""
    for line in ocr_lines:
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        match = _NUMBERED_ISSUE_EXPIRY_PATTERN.search(cleaned)
        if match is None:
            continue

        issue = normalize_date(match.group(1)).normalized
        expiry = normalize_date(match.group(2)).normalized
        return issue, expiry

    return None


def _extract_numbered_place_of_birth(
    ocr_lines: LineRecognitionResult,
) -> str | None:
    """Extract place of birth from line following numbered ``3`` field."""
    for index, line in enumerate(ocr_lines):
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        if _NUMBERED_DOB_PATTERN.match(cleaned) is None:
            continue

        for follow_index in range(index + 1, len(ocr_lines)):
            follow_value = cleanup_text(ocr_lines[follow_index].text)
            if not follow_value:
                continue

            if _starts_with_field_marker(follow_value):
                break

            stripped = _strip_leading_field_marker(follow_value)
            stripped = _strip_trailing_date_fragment(stripped)
            if not stripped:
                continue

            return normalize_name(stripped).normalized

    return None


def _extract_numbered_value(
    ocr_lines: LineRecognitionResult,
    pattern: re.Pattern[str],
) -> str | None:
    """Extract one value from numbered-field OCR line pattern."""
    for line in ocr_lines:
        cleaned = cleanup_text(line.text)
        if not cleaned:
            continue

        match = pattern.match(cleaned)
        if match is None:
            continue

        value = cleanup_text(match.group(1))
        if value:
            return value

    return None


def _extract_labeled_values(
    ocr_lines: LineRecognitionResult,
    labels: tuple[str, ...],
) -> list[str]:
    """Collect candidate values from label tails and adjacent lines."""
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
    """Extract and normalize the first date-like token from text."""
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


def _extract_value_after_label_flexible(
    text: str,
    labels: tuple[str, ...],
) -> str | None:
    """Extract label tail when OCR glues neighboring symbols together."""
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


def _starts_with_field_marker(value: str) -> bool:
    """Return whether line begins with numbered field marker prefix."""
    return _FIELD_MARKER_PATTERN.match(value) is not None


def _strip_leading_field_marker(value: str) -> str:
    """Remove numbered field prefixes like ``4c)`` or ``5.``."""
    return cleanup_text(_FIELD_MARKER_PATTERN.sub("", value, count=1))


def _is_name_part(value: str) -> bool:
    """Return whether value resembles one valid person-name token."""
    cleaned = cleanup_text(value)
    if not cleaned or _DIGIT_PATTERN.search(cleaned):
        return False

    tokens = cleaned.split(" ")
    if len(tokens) != 1:
        return False

    return _NAME_TOKEN_PATTERN.fullmatch(tokens[0]) is not None


def _is_full_name_like_text(value: str) -> bool:
    """Return whether value resembles a full name in a single OCR line."""
    cleaned = cleanup_text(value)
    if not cleaned or _DIGIT_PATTERN.search(cleaned):
        return False

    tokens = cleaned.split(" ")
    if len(tokens) < 2 or len(tokens) > 3:
        return False

    return all(_NAME_TOKEN_PATTERN.fullmatch(token) is not None for token in tokens)


def _strip_trailing_date_fragment(value: str) -> str:
    """Drop date fragments from semantic text lines."""
    cleaned = cleanup_text(value)
    cleaned = _DATE_PATTERN.sub("", cleaned)
    return cleanup_text(cleaned)


def _count_digits(value: str) -> int:
    """Count decimal digits in one text value."""
    return sum(character.isdigit() for character in value)


def _digits_only(value: str) -> str:
    """Return one compact string containing only decimal digits."""
    return "".join(character for character in value if character.isdigit())


def _format_license_number(value: str) -> str:
    """Format 10-digit license number as ``XX XX XXXXXX``."""
    digits = _digits_only(value)
    if len(digits) != 10:
        return cleanup_text(value)

    return f"{digits[:2]} {digits[2:4]} {digits[4:]}"


def _normalize_authority_text(value: str) -> str:
    """Normalize authority text while preserving acronym tokens."""
    cleaned = _strip_leading_field_marker(cleanup_text(value))
    if not cleaned:
        return cleaned

    normalized_tokens: list[str] = []
    for token in cleaned.split(" "):
        if not token:
            continue

        lowered = token.casefold()
        if lowered in {"гибдд", "gibdd", "мрэо", "mreo"}:
            normalized_tokens.append(token.upper())
            continue

        if _AUTHORITY_ACRONYM_PATTERN.fullmatch(token) is not None:
            normalized_tokens.append(token)
            continue

        if any(character.isalpha() for character in token):
            normalized_tokens.append(normalize_name(token).normalized)
            continue

        normalized_tokens.append(token)

    return cleanup_text(" ".join(normalized_tokens))


__all__ = ["DriversLicenseExtractor"]
