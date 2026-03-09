"""Confidence aggregation for extracted document fields."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

from app.api.schemas import DocumentTypeDetected
from app.api.schemas import ExtractedFields

_REQUIRED_FIELDS_BY_TYPE: dict[DocumentTypeDetected, tuple[str, ...]] = {
    DocumentTypeDetected.BANK_CARD: (
        "card_number",
        "cardholder_name",
        "expiry_date",
    ),
    DocumentTypeDetected.ID_CARD: (
        "full_name",
        "date_of_birth",
        "sex",
        "place_of_birth",
        "document_number",
    ),
    DocumentTypeDetected.DRIVERS_LICENSE: (
        "full_name",
        "date_of_birth",
        "place_of_birth",
        "issue_date",
        "expiry_date",
        "issuing_authority",
        "license_number",
        "place_of_residence",
        "license_class",
    ),
}

_FLAG_FIELD_IMPACT: dict[str, tuple[str, ...]] = {
    "issue_date_after_expiry_date": ("issue_date", "expiry_date"),
    "date_of_birth_after_issue_date": ("date_of_birth", "issue_date"),
    "date_of_birth_after_expiry_date": ("date_of_birth", "expiry_date"),
}


@dataclass(frozen=True)
class ConfidenceScoreResult:
    """Per-field and aggregate confidence outputs."""

    field_confidence: dict[str, float]
    aggregate_confidence: float


class ConfidenceScorer:
    """Compute bounded confidence scores for extracted fields."""

    def __init__(
        self,
        *,
        default_present_confidence: float = 0.85,
        validation_flag_penalty: float = 0.20,
    ) -> None:
        self._default_present_confidence = _clamp(default_present_confidence)
        self._validation_flag_penalty = _clamp(validation_flag_penalty)

    def score_fields(
        self,
        *,
        fields: ExtractedFields,
        evidence_scores: Mapping[str, float] | None = None,
        validation_flags: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Score each extracted field with optional evidence overrides."""
        payload = fields.model_dump()
        resolved_evidence = evidence_scores or {}
        field_confidence: dict[str, float] = {}

        for field_name, value in payload.items():
            if _has_value(value):
                score = resolved_evidence.get(
                    field_name,
                    self._default_present_confidence,
                )
                field_confidence[field_name] = _clamp(score)
                continue

            field_confidence[field_name] = 0.0

        if validation_flags:
            self._apply_flag_penalties(
                field_confidence=field_confidence,
                validation_flags=validation_flags,
            )

        return field_confidence

    def aggregate_confidence(
        self,
        *,
        field_confidence: Mapping[str, float],
        fields: ExtractedFields,
        document_type: DocumentTypeDetected,
    ) -> float:
        """Aggregate field confidence with schema-aware required fields."""
        required_fields = _REQUIRED_FIELDS_BY_TYPE.get(document_type)
        if required_fields is not None:
            required_scores = [
                _clamp(field_confidence.get(field_name, 0.0))
                for field_name in required_fields
            ]
            return _mean(required_scores)

        present_fields = _present_field_names(fields)
        if not present_fields:
            return 0.0

        scores = [
            _clamp(field_confidence.get(field_name, 0.0))
            for field_name in present_fields
        ]
        return _mean(scores)

    def score(
        self,
        *,
        fields: ExtractedFields,
        document_type: DocumentTypeDetected,
        evidence_scores: Mapping[str, float] | None = None,
        validation_flags: Sequence[str] | None = None,
    ) -> ConfidenceScoreResult:
        """Compute per-field and aggregate confidence in one call."""
        field_confidence = self.score_fields(
            fields=fields,
            evidence_scores=evidence_scores,
            validation_flags=validation_flags,
        )
        aggregate_confidence = self.aggregate_confidence(
            field_confidence=field_confidence,
            fields=fields,
            document_type=document_type,
        )
        return ConfidenceScoreResult(
            field_confidence=field_confidence,
            aggregate_confidence=aggregate_confidence,
        )

    def _apply_flag_penalties(
        self,
        *,
        field_confidence: dict[str, float],
        validation_flags: Sequence[str],
    ) -> None:
        """Apply consistency-flag penalties to impacted fields."""
        for flag in validation_flags:
            impacted_fields = _FLAG_FIELD_IMPACT.get(flag)
            if impacted_fields is None:
                continue

            for field_name in impacted_fields:
                current_score = field_confidence.get(field_name)
                if current_score is None:
                    continue

                adjusted = current_score - self._validation_flag_penalty
                field_confidence[field_name] = _clamp(adjusted)


def _has_value(value: object) -> bool:
    """Return whether extracted field value should be treated as present."""
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True


def _present_field_names(fields: ExtractedFields) -> list[str]:
    """List names for fields that have non-empty extracted values."""
    payload = fields.model_dump()
    return [
        field_name
        for field_name, value in payload.items()
        if _has_value(value)
    ]


def _mean(scores: Sequence[float]) -> float:
    """Return arithmetic mean for non-empty confidence score lists."""
    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))


def _clamp(score: float) -> float:
    """Clamp confidence score into the inclusive ``[0.0, 1.0]`` range."""
    if score <= 0.0:
        return 0.0

    if score >= 1.0:
        return 1.0

    return float(score)


__all__ = [
    "ConfidenceScoreResult",
    "ConfidenceScorer",
]
