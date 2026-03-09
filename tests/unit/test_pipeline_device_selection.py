"""Unit tests for OCR runtime device routing and safe fallback behavior."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import AppSettings
from app.ocr.detector import DetectionResult
from app.pipeline.context import PipelineContext
import app.pipeline.processing as pipeline_processing


class _RecordingDetector:
    """Detector test double that records GPU flag at construction."""

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("ru", "en"),
        gpu: bool = False,
        reader=None,
    ) -> None:
        self.languages = languages
        self.gpu = gpu
        self.reader = reader

    def detect(self, aligned_image: np.ndarray) -> DetectionResult:
        return []


class _RecordingRecognizer:
    """Recognizer test double that records GPU flag at construction."""

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("ru",),
        gpu: bool = False,
        reader=None,
    ) -> None:
        self.languages = languages
        self.gpu = gpu
        self.reader = reader

    def recognize(self, aligned_image: np.ndarray, regions: DetectionResult):
        return []

    def group_tokens_to_lines(self, tokens):
        return []


@pytest.mark.parametrize(
    ("device_mode", "cuda_available", "expected_device"),
    [
        ("cpu", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
    ],
)
def test_select_runtime_device_resolves_expected_target(
    device_mode: str,
    cuda_available: bool,
    expected_device: str,
) -> None:
    """Resolve runtime device from requested mode and CUDA availability."""
    resolved = pipeline_processing._select_runtime_device(
        device_mode=device_mode,
        cuda_available=cuda_available,
    )

    assert resolved == expected_device


def test_select_runtime_device_falls_back_to_cpu_for_unavailable_cuda() -> None:
    """Fall back to CPU when mode is cuda but CUDA is not available."""
    resolved = pipeline_processing._select_runtime_device(
        device_mode="cuda",
        cuda_available=False,
    )

    assert resolved == "cpu"


def test_resolve_text_detector_uses_cpu_fallback_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct detector with GPU disabled for safe CUDA fallback."""
    monkeypatch.setattr(
        pipeline_processing,
        "load_settings",
        lambda: AppSettings(device_mode="cuda"),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "_is_cuda_available",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "EasyOCRTextDetector",
        _RecordingDetector,
    )

    context = PipelineContext(request_id="req-detector-device", image_bytes=b"1")

    detector = pipeline_processing._resolve_text_detector(context)

    assert isinstance(detector, _RecordingDetector)
    assert detector.gpu is False
    assert context.stage_outputs["ocr_device"] == "cpu"
    assert context.metadata["ocr_device"] == {
        "requested": "cuda",
        "resolved": "cpu",
        "cuda_available": False,
    }


def test_resolve_text_recognizer_uses_cuda_when_auto_and_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct recognizer with GPU enabled when auto resolves to CUDA."""
    monkeypatch.setattr(
        pipeline_processing,
        "load_settings",
        lambda: AppSettings(device_mode="auto"),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "_is_cuda_available",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "EasyOCRTextRecognizer",
        _RecordingRecognizer,
    )

    context = PipelineContext(
        request_id="req-recognizer-device",
        image_bytes=b"1",
    )

    recognizer = pipeline_processing._resolve_text_recognizer(context)

    assert isinstance(recognizer, _RecordingRecognizer)
    assert recognizer.gpu is True
    assert context.stage_outputs["ocr_device"] == "cuda"
    assert context.metadata["ocr_device"] == {
        "requested": "auto",
        "resolved": "cuda",
        "cuda_available": True,
    }
