"""Unit tests for OCR runtime device routing and safe fallback behavior."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import AppSettings
from app.core.exceptions import InternalProcessingError
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
    metadata = context.metadata["ocr_device"]
    assert metadata["requested"] == "auto"
    assert metadata["resolved"] == "cuda"
    assert metadata["cuda_available"] is True


def test_resolve_runtime_device_rejects_budget_above_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast when configured GPU budget exceeds the hard guardrail."""
    monkeypatch.setattr(
        pipeline_processing,
        "load_settings",
        lambda: AppSettings(
            device_mode="cuda",
            gpu_memory_budget_gb=11.0,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_processing,
        "_is_cuda_available",
        lambda: True,
        raising=False,
    )

    context = PipelineContext(
        request_id="req-budget-over-limit",
        image_bytes=b"1",
    )

    with pytest.raises(InternalProcessingError) as exc_info:
        pipeline_processing._resolve_ocr_runtime_device(context)

    error = exc_info.value
    assert error.error_code == "ocr_gpu_memory_budget_over_limit"
    assert error.details is not None
    assert error.details["configured_budget_gb"] == 11.0
    assert error.details["max_budget_gb"] == 10.0


def test_resolve_runtime_device_rejects_model_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast when selected OCR model requires more GPU memory."""
    monkeypatch.setattr(
        pipeline_processing,
        "load_settings",
        lambda: AppSettings(
            device_mode="cuda",
            gpu_memory_budget_gb=6.0,
        ),
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
        "_resolve_ocr_gpu_memory_requirement_gb",
        lambda: 8.0,
        raising=False,
    )

    context = PipelineContext(
        request_id="req-model-over-budget",
        image_bytes=b"1",
    )

    with pytest.raises(InternalProcessingError) as exc_info:
        pipeline_processing._resolve_ocr_runtime_device(context)

    error = exc_info.value
    assert error.error_code == "ocr_gpu_memory_budget_exceeded"
    assert error.details is not None
    assert error.details["configured_budget_gb"] == 6.0
    assert error.details["required_budget_gb"] == 8.0


def test_resolve_runtime_device_allows_cuda_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow CUDA execution when model requirement is within budget."""
    monkeypatch.setattr(
        pipeline_processing,
        "load_settings",
        lambda: AppSettings(
            device_mode="cuda",
            gpu_memory_budget_gb=8.0,
        ),
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
        "_resolve_ocr_gpu_memory_requirement_gb",
        lambda: 7.5,
        raising=False,
    )

    context = PipelineContext(
        request_id="req-model-within-budget",
        image_bytes=b"1",
    )

    resolved = pipeline_processing._resolve_ocr_runtime_device(context)

    assert resolved == "cuda"
    metadata = context.metadata["ocr_device"]
    assert metadata["requested"] == "cuda"
    assert metadata["resolved"] == "cuda"
    assert metadata["cuda_available"] is True
    assert metadata["configured_budget_gb"] == 8.0
    assert metadata["required_budget_gb"] == 7.5
