"""Unit tests for typed alignment failure diagnostics."""

import numpy as np
import pytest

from app.core.exceptions import UnprocessableDocumentError
from app.preprocessing.document_preprocessor import AlignmentFailureDiagnostic
from app.preprocessing.document_preprocessor import BoundaryDetectionResult
from app.preprocessing.document_preprocessor import DocumentPreprocessor


def test_align_image_returns_typed_boundary_failure_diagnostic() -> None:
    """Wrap boundary failures with a structured alignment diagnostic payload."""
    preprocessor = DocumentPreprocessor()
    image = np.empty((0, 0, 3), dtype=np.uint8)

    with pytest.raises(UnprocessableDocumentError) as exc_info:
        preprocessor.align_image(image)

    error = exc_info.value
    assert error.message == "Document alignment failed."
    assert error.details is not None
    assert error.details["reason"] == "alignment_failure"

    diagnostic_payload = error.details["alignment_diagnostic"]
    diagnostic = AlignmentFailureDiagnostic.model_validate(diagnostic_payload)

    assert diagnostic.stage == "boundary_detection"
    assert diagnostic.reason == "empty_image"
    assert diagnostic.message == "Cannot detect boundary on an empty image."


def test_align_image_returns_typed_perspective_failure_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap perspective failures with stage-specific diagnostic details."""
    preprocessor = DocumentPreprocessor()
    image = np.full((32, 48, 3), 220, dtype=np.uint8)

    boundary = BoundaryDetectionResult(
        corners=[(0, 0), (47, 0), (47, 31), (0, 31)],
        bounds=(0, 0, 48, 32),
        contour_area=1504.0,
        area_ratio=0.98,
    )

    def _mock_detect_document_boundary(_: np.ndarray) -> BoundaryDetectionResult:
        return boundary

    def _mock_perspective_failure(
        _: np.ndarray,
        __: BoundaryDetectionResult,
    ) -> np.ndarray:
        raise UnprocessableDocumentError(
            message="Perspective transform failed.",
            details={
                "reason": "perspective_transform_failed",
                "opencv_error": "mock_error",
            },
        )

    monkeypatch.setattr(
        preprocessor,
        "detect_document_boundary",
        _mock_detect_document_boundary,
    )
    monkeypatch.setattr(
        preprocessor,
        "apply_perspective_correction",
        _mock_perspective_failure,
    )

    with pytest.raises(UnprocessableDocumentError) as exc_info:
        preprocessor.align_image(image)

    error = exc_info.value
    assert error.message == "Document alignment failed."
    assert error.details is not None
    assert error.details["reason"] == "alignment_failure"

    diagnostic_payload = error.details["alignment_diagnostic"]
    diagnostic = AlignmentFailureDiagnostic.model_validate(diagnostic_payload)

    assert diagnostic.stage == "perspective_correction"
    assert diagnostic.reason == "perspective_transform_failed"
    assert diagnostic.message == "Perspective transform failed."
    assert diagnostic.details["opencv_error"] == "mock_error"
