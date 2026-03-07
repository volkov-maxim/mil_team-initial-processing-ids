"""Unit tests for preprocessing type/size validation."""

from app.preprocessing.document_preprocessor import DocumentPreprocessor


def test_validate_type_size_readability_accepts_allowed_type_and_size() -> None:
    """Accept payloads with supported media type and valid byte size."""
    preprocessor = DocumentPreprocessor(max_file_size_bytes=16)

    outcome = preprocessor.validate_type_size_readability(
        b"12345678",
        content_type="image/png",
    )

    assert outcome.is_valid is True
    assert outcome.failure_reason is None
    assert outcome.message is None
    assert outcome.details["content_type"] == "image/png"
    assert outcome.details["file_size_bytes"] == 8


def test_validate_type_size_readability_rejects_unsupported_media_type() -> None:
    """Block unsupported payload content types with typed failure details."""
    preprocessor = DocumentPreprocessor(max_file_size_bytes=16)

    outcome = preprocessor.validate_type_size_readability(
        b"1234",
        content_type="application/pdf",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "unsupported_media_type"
    assert outcome.message == "Unsupported media type."
    assert outcome.details["content_type"] == "application/pdf"


def test_validate_type_size_readability_rejects_payload_above_size_limit() -> None:
    """Block payloads that exceed configured maximum file size."""
    preprocessor = DocumentPreprocessor(max_file_size_bytes=4)

    outcome = preprocessor.validate_type_size_readability(
        b"12345",
        content_type="image/png",
    )

    assert outcome.is_valid is False
    assert outcome.failure_reason == "file_too_large"
    assert outcome.message == "File size exceeds the configured limit."
    assert outcome.details["file_size_bytes"] == 5
    assert outcome.details["max_file_size_bytes"] == 4


def test_validate_type_size_readability_normalizes_content_type_header() -> None:
    """Accept content types that include optional header parameters."""
    preprocessor = DocumentPreprocessor(max_file_size_bytes=16)

    outcome = preprocessor.validate_type_size_readability(
        b"1234",
        content_type="image/jpeg; charset=binary",
    )

    assert outcome.is_valid is True
    assert outcome.details["content_type"] == "image/jpeg"
