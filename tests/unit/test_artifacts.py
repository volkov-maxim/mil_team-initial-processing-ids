"""Unit tests for request-scoped artifact storage manager behavior."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.storage.artifacts import ArtifactStorageManager


def _build_detection_payload() -> list[Mapping[str, Any]]:
    """Build one synthetic detection payload for overlay rendering tests."""
    return [
        {
            "text": "АНДРЕЙ",
            "confidence": 0.91,
            "polygon": [
                [8.0, 10.0],
                [56.0, 10.0],
                [56.0, 28.0],
                [8.0, 28.0],
            ],
            "bounding_box": [8.0, 10.0, 48.0, 18.0],
        }
    ]


def test_draw_detections_returns_annotated_image() -> None:
    """Render visible detection markup over an aligned source image."""
    manager = ArtifactStorageManager(artifacts_root=Path("artifacts"))
    source = np.zeros((64, 128, 3), dtype=np.uint8)

    overlay = manager.draw_detections(
        aligned_image=source,
        detections=_build_detection_payload(),
    )

    assert overlay.shape == source.shape
    assert overlay.dtype == source.dtype
    assert not np.array_equal(overlay, source)


def test_persist_detection_overlay_writes_readable_artifact(
    tmp_path: Path,
) -> None:
    """Persist an overlay image and return readable artifact metadata."""
    manager = ArtifactStorageManager(artifacts_root=tmp_path)
    overlay = np.full((32, 64, 3), 255, dtype=np.uint8)

    artifact = manager.persist_detection_overlay(
        request_id="req-overlay-unit",
        overlay_image=overlay,
    )

    artifact_path = Path(artifact.path)
    assert artifact_path.exists()
    assert artifact_path.is_file()
    assert artifact.height == 32
    assert artifact.width == 64
    assert artifact.channels == 3

    persisted = cv2.imread(
        str(artifact_path),
        cv2.IMREAD_UNCHANGED,
    )
    assert persisted is not None
    assert persisted.size > 0
