"""Unit tests for validation confidence aggregation."""

from __future__ import annotations

import pytest

from app.api.schemas import DocumentTypeDetected
from app.api.schemas import ExtractedFields
from app.validation.confidence import ConfidenceScorer


def test_score_fields_uses_evidence_default_and_missing_zero() -> None:
    """Use explicit evidence, default score, and zero for missing fields."""
    fields = ExtractedFields(
        card_number="4532 1234 5678 9010",
        cardholder_name="Ivan Ivanov",
    )

    scorer = ConfidenceScorer(default_present_confidence=0.83)
    scores = scorer.score_fields(
        fields=fields,
        evidence_scores={"card_number": 0.91},
    )

    assert scores["card_number"] == pytest.approx(0.91)
    assert scores["cardholder_name"] == pytest.approx(0.83)
    assert scores["expiry_date"] == pytest.approx(0.0)
    assert scores["full_name"] == pytest.approx(0.0)


def test_score_fields_applies_validation_flag_penalty() -> None:
    """Reduce impacted field confidence when consistency flags are present."""
    fields = ExtractedFields(
        date_of_birth="1984-09-04",
        issue_date="2027-07-21",
        expiry_date="2026-07-21",
    )

    scorer = ConfidenceScorer(
        default_present_confidence=0.90,
        validation_flag_penalty=0.15,
    )
    scores = scorer.score_fields(
        fields=fields,
        validation_flags=["issue_date_after_expiry_date"],
    )

    assert scores["issue_date"] == pytest.approx(0.75)
    assert scores["expiry_date"] == pytest.approx(0.75)
    assert scores["date_of_birth"] == pytest.approx(0.90)


def test_aggregate_confidence_for_bank_card_uses_required_fields() -> None:
    """Compute aggregate from required schema fields for known document type."""
    fields = ExtractedFields(card_number="4532 1234 5678 9010")

    scorer = ConfidenceScorer(default_present_confidence=0.90)
    field_scores = scorer.score_fields(fields=fields)

    aggregate = scorer.aggregate_confidence(
        field_confidence=field_scores,
        fields=fields,
        document_type=DocumentTypeDetected.BANK_CARD,
    )

    assert aggregate == pytest.approx((0.90 + 0.0 + 0.0) / 3.0)


def test_aggregate_confidence_for_unknown_uses_present_fields_only() -> None:
    """Compute aggregate from present fields when type is unknown."""
    fields = ExtractedFields(
        full_name="Иванов Иван",
        place_of_birth="Москва",
    )

    scorer = ConfidenceScorer(default_present_confidence=0.80)
    field_scores = scorer.score_fields(
        fields=fields,
        evidence_scores={
            "full_name": 0.90,
            "place_of_birth": 0.60,
        },
    )

    aggregate = scorer.aggregate_confidence(
        field_confidence=field_scores,
        fields=fields,
        document_type=DocumentTypeDetected.UNKNOWN,
    )

    assert aggregate == pytest.approx(0.75)


def test_score_returns_per_field_and_aggregate() -> None:
    """Return consistent per-field and aggregate confidence outputs."""
    fields = ExtractedFields(
        full_name="Иванов Иван Иванович",
        date_of_birth="1984-09-04",
        sex="M",
        place_of_birth="Москва",
        document_number="45 77 695122",
    )

    scorer = ConfidenceScorer(default_present_confidence=0.80)
    result = scorer.score(
        fields=fields,
        document_type=DocumentTypeDetected.ID_CARD,
        evidence_scores={
            "full_name": 0.90,
            "document_number": 0.70,
        },
    )

    assert result.field_confidence["full_name"] == pytest.approx(0.90)
    assert result.field_confidence["document_number"] == pytest.approx(0.70)
    assert result.aggregate_confidence == pytest.approx(0.80)
