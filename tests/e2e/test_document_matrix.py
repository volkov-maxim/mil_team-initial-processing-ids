"""End-to-end matrix tests for document type and quality variants."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
import pytest

import app.pipeline.processing as pipeline_processing
from app.api.schemas import DocumentTypeHint
from app.main import app
from app.ocr.detector import DetectionResult
from app.ocr.detector import TextRegion
from app.ocr.recognizer import LineRecognitionResult
from app.ocr.recognizer import RecognizedLine
from app.ocr.recognizer import RecognizedToken
from app.ocr.recognizer import TokenRecognitionResult
from app.pipeline.context import PipelineContext
from app.storage.artifacts import ArtifactStorageManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_BASE_FIXTURES_BY_HINT: dict[DocumentTypeHint, Path] = {
    DocumentTypeHint.BANK_CARD: (
        PROJECT_ROOT / "images" / "bank_cards" / "bank-cards.jpg"
    ),
    DocumentTypeHint.ID_CARD: (
        PROJECT_ROOT / "images" / "id_cards" / "passport_min.png"
    ),
    DocumentTypeHint.DRIVERS_LICENSE: (
        PROJECT_ROOT / "images" / "drivers_licenses" / "orig.png"
    ),
}

_OCR_TEXTS_BY_HINT: dict[DocumentTypeHint, list[str]] = {
    DocumentTypeHint.BANK_CARD: [
        "VISA",
        "1234 5678 9123 4567",
        "MR. ALIOTH",
        "GOOD THRU 12/26",
    ],
    DocumentTypeHint.ID_CARD: [
        "45 77 695122",
        "КОРОКОВ ЛЕОНИД БОРИСОВИЧ",
        "19.10.1969",
    ],
    DocumentTypeHint.DRIVERS_LICENSE: [
        "1. МИТИН",
        "2. АНДРЕЙ ВЛАДИМИРОВИЧ",
        "3. 04.09.1984",
        "МОСКВА",
        "4a) 21.07.2016 4b) 21.07.2026",
        "4c) ГИБДД 7723",
        "5. 77 28 089628",
        "8. МОСКВА",
        "9. B",
    ],
}

_QUALITY_CONDITIONS = (
    "clean",
    "rotated",
    "perspective",
    "low_quality",
)


class _ScenarioTextDetector:
    """Return one deterministic text region for OCR-stage smoke coverage."""

    def detect(self, aligned_image: np.ndarray) -> DetectionResult:
        """Return one region anchored inside aligned image bounds."""
        height, width = aligned_image.shape[:2]
        region_width = float(max(1, width - 20))
        region_height = float(max(1, height - 20))

        return [
            TextRegion(
                polygon=[
                    (10.0, 10.0),
                    (10.0 + region_width, 10.0),
                    (10.0 + region_width, 10.0 + region_height),
                    (10.0, 10.0 + region_height),
                ],
                bounding_box=(10.0, 10.0, region_width, region_height),
                confidence=0.95,
            )
        ]


class _ScenarioTextRecognizer:
    """Return deterministic OCR tokens and grouped lines per scenario."""

    def __init__(self, lines: LineRecognitionResult) -> None:
        """Store prepared line outputs for deterministic OCR behavior."""
        self._lines = lines

    def recognize(
        self,
        aligned_image: np.ndarray,
        regions: DetectionResult,
    ) -> TokenRecognitionResult:
        """Emit one token per prepared OCR line."""
        assert aligned_image.size > 0
        assert len(regions) > 0
        return [line.tokens[0] for line in self._lines]

    def group_tokens_to_lines(
        self,
        tokens: TokenRecognitionResult,
    ) -> LineRecognitionResult:
        """Return prepared grouped lines after token recognition."""
        assert len(tokens) == len(self._lines)
        return self._lines


def _build_ocr_lines(texts: list[str]) -> LineRecognitionResult:
    """Build deterministic OCR line fixtures from plain text rows."""
    lines: list[RecognizedLine] = []
    for index, text in enumerate(texts):
        y_coord = 12.0 + (index * 26.0)
        token_width = float(min(250, max(80, len(text) * 8)))
        token = RecognizedToken(
            text=text,
            polygon=[
                (10.0, y_coord),
                (10.0 + token_width, y_coord),
                (10.0 + token_width, y_coord + 22.0),
                (10.0, y_coord + 22.0),
            ],
            bounding_box=(10.0, y_coord, token_width, 22.0),
            confidence=0.90,
        )
        lines.append(
            RecognizedLine(
                text=text,
                tokens=[token],
                bounding_box=(10.0, y_coord, token_width, 22.0),
                confidence=0.90,
            )
        )

    return lines


def _patch_e2e_stage_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
) -> None:
    """Patch optional OCR and artifact storage for deterministic E2E runs."""

    def _resolve_storage_manager(
        _context: PipelineContext,
    ) -> ArtifactStorageManager:
        return ArtifactStorageManager(artifacts_root=tmp_path)

    def _resolve_text_detector(
        _context: PipelineContext,
    ) -> _ScenarioTextDetector:
        return _ScenarioTextDetector()

    def _resolve_text_recognizer(
        context: PipelineContext,
    ) -> _ScenarioTextRecognizer:
        line_texts = _OCR_TEXTS_BY_HINT.get(
            context.document_type_hint,
            _OCR_TEXTS_BY_HINT[DocumentTypeHint.ID_CARD],
        )
        return _ScenarioTextRecognizer(_build_ocr_lines(line_texts))

    monkeypatch.setattr(
        pipeline_processing,
        "_resolve_artifact_storage_manager",
        _resolve_storage_manager,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "_resolve_text_detector",
        _resolve_text_detector,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "_resolve_text_recognizer",
        _resolve_text_recognizer,
    )


def _load_fixture_image(image_path: Path) -> np.ndarray:
    """Load one fixture image and assert it is readable."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.size > 0
    return image


def _rotate_image_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image while expanding bounds to avoid clipping."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    abs_cos = abs(transform[0, 0])
    abs_sin = abs(transform[0, 1])
    bound_width = int(round((height * abs_sin) + (width * abs_cos)))
    bound_height = int(round((height * abs_cos) + (width * abs_sin)))

    transform[0, 2] += (bound_width / 2.0) - center[0]
    transform[1, 2] += (bound_height / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        transform,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _apply_perspective_skew(image: np.ndarray) -> np.ndarray:
    """Apply a mild perspective warp to emulate skewed captures."""
    height, width = image.shape[:2]
    offset_x = max(2.0, width * 0.06)
    offset_y = max(2.0, height * 0.05)

    source = np.float32(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ]
    )
    destination = np.float32(
        [
            [offset_x, offset_y],
            [float(width - 1) - offset_x, 0.0],
            [float(width - 1), float(height - 1) - offset_y],
            [0.0, float(height - 1)],
        ]
    )

    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_low_quality_degradation(image: np.ndarray) -> np.ndarray:
    """Degrade image using resize, blur, and lossy compression cycle."""
    height, width = image.shape[:2]
    reduced_width = max(96, int(round(width * 0.65)))
    reduced_height = max(96, int(round(height * 0.65)))

    resized_down = cv2.resize(
        image,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )
    restored = cv2.resize(
        resized_down,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    blurred = cv2.GaussianBlur(restored, (3, 3), 0.0)

    was_encoded, jpeg_bytes = cv2.imencode(
        ".jpg",
        blurred,
        [int(cv2.IMWRITE_JPEG_QUALITY), 45],
    )
    if not was_encoded:
        return blurred

    decoded = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if decoded is None:
        return blurred

    return decoded


def _build_variant_image(
    base_image: np.ndarray,
    quality_condition: str,
) -> np.ndarray:
    """Build one image variant for matrix quality-condition coverage."""
    if quality_condition == "clean":
        return base_image.copy()
    if quality_condition == "rotated":
        return _rotate_image_bound(base_image, angle_deg=11.0)
    if quality_condition == "perspective":
        return _apply_perspective_skew(base_image)
    if quality_condition == "low_quality":
        return _apply_low_quality_degradation(base_image)

    raise ValueError(f"Unknown quality condition: {quality_condition}")


def _encode_upload_image(image: np.ndarray) -> bytes:
    """Encode an upload image as PNG bytes for multipart requests."""
    was_encoded, encoded = cv2.imencode(".png", image)
    assert was_encoded
    return encoded.tobytes()


def _resolve_artifact_path(path_value: str) -> Path:
    """Resolve absolute or project-relative artifact path values."""
    artifact_path = Path(path_value)
    if artifact_path.is_absolute():
        return artifact_path

    return PROJECT_ROOT / artifact_path


@pytest.mark.parametrize(
    ("document_type_hint", "quality_condition"),
    [
        pytest.param(
            document_type_hint,
            quality_condition,
            id=f"{document_type_hint.value}-{quality_condition}",
        )
        for document_type_hint in (
            DocumentTypeHint.BANK_CARD,
            DocumentTypeHint.ID_CARD,
            DocumentTypeHint.DRIVERS_LICENSE,
        )
        for quality_condition in _QUALITY_CONDITIONS
    ],
)
def test_process_document_e2e_matrix_by_document_type_and_quality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    document_type_hint: DocumentTypeHint,
    quality_condition: str,
) -> None:
    """Validate E2E matrix coverage and both mandatory output artifacts."""
    _patch_e2e_stage_dependencies(monkeypatch, tmp_path=tmp_path)

    fixture_path = _BASE_FIXTURES_BY_HINT[document_type_hint]
    assert fixture_path.exists()

    base_image = _load_fixture_image(fixture_path)
    variant_image = _build_variant_image(base_image, quality_condition)
    upload_bytes = _encode_upload_image(variant_image)

    client = TestClient(app)
    response = client.post(
        "/v1/process-document",
        files={
            "image": (
                f"{document_type_hint.value}-{quality_condition}.png",
                upload_bytes,
                "image/png",
            )
        },
        data={
            "document_type_hint": document_type_hint.value,
            "use_external_fallback": "false",
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["document_type_detected"] == document_type_hint.value
    assert len(payload["detections"]) > 0
    assert any(value is not None for value in payload["fields"].values())

    metadata = payload["processing_metadata"]
    assert metadata["executed_stages"] == list(
        pipeline_processing.STAGE_SEQUENCE
    )

    aligned_artifact = metadata["aligned_artifact"]
    overlay_artifact = metadata["overlay_artifact"]

    assert aligned_artifact["path"] == payload["aligned_image"]
    assert aligned_artifact["height"] > 0
    assert aligned_artifact["width"] > 0
    assert aligned_artifact["channels"] in {1, 3}

    assert overlay_artifact["height"] > 0
    assert overlay_artifact["width"] > 0
    assert overlay_artifact["channels"] in {1, 3}

    aligned_path = _resolve_artifact_path(aligned_artifact["path"])
    overlay_path = _resolve_artifact_path(overlay_artifact["path"])

    assert aligned_path.exists()
    assert aligned_path.is_file()
    assert overlay_path.exists()
    assert overlay_path.is_file()

    aligned_image = cv2.imread(str(aligned_path), cv2.IMREAD_UNCHANGED)
    overlay_image = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)

    assert aligned_image is not None
    assert aligned_image.size > 0
    assert overlay_image is not None
    assert overlay_image.size > 0