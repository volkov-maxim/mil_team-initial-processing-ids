"""Artifact storage manager for request-scoped pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any
from typing import Mapping

import cv2
import numpy as np

from app.core.config import load_settings
from app.core.exceptions import InternalProcessingError

_ALLOWED_REQUEST_ID_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_"
)
_ALIGNED_IMAGE_FILENAME = "aligned-image.png"
_OVERLAY_IMAGE_FILENAME = "detections-overlay.png"
_OVERLAY_LINE_COLOR = (0, 255, 0)
_OVERLAY_TEXT_COLOR = (255, 255, 255)
_OVERLAY_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_OVERLAY_TEXT_SCALE = 0.45
_OVERLAY_TEXT_THICKNESS = 1


@dataclass(frozen=True, slots=True)
class StoredAlignedArtifact:
    """Metadata describing one persisted aligned-image artifact."""

    path: str
    height: int
    width: int
    channels: int

    def as_metadata(self) -> dict[str, str | int]:
        """Return a JSON-serializable payload for response metadata."""
        return {
            "path": self.path,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
        }


@dataclass(frozen=True, slots=True)
class StoredOverlayArtifact:
    """Metadata describing one persisted detection-overlay artifact."""

    path: str
    height: int
    width: int
    channels: int

    def as_metadata(self) -> dict[str, str | int]:
        """Return a JSON-serializable payload for response metadata."""
        return {
            "path": self.path,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
        }


class ArtifactStorageManager:
    """Persist request-scoped artifacts and return stable local references."""

    def __init__(self, *, artifacts_root: Path) -> None:
        """Initialize manager with a configurable artifact root directory."""
        self.artifacts_root = Path(artifacts_root)

    @classmethod
    def from_settings(cls) -> "ArtifactStorageManager":
        """Build manager from application settings."""
        settings = load_settings()
        return cls(artifacts_root=settings.artifacts_root)

    def create_request_directory(self, request_id: str) -> Path:
        """Create and return one request-scoped artifact directory."""
        safe_request_id = _sanitize_request_id(request_id)
        request_dir = self.artifacts_root / safe_request_id

        try:
            request_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise InternalProcessingError(
                message="Failed to create request artifact directory.",
                error_code="artifact_directory_creation_failed",
                details={
                    "request_id": request_id,
                    "path": request_dir.as_posix(),
                    "os_error": str(error),
                },
            ) from error

        return request_dir

    def persist_aligned_image(
        self,
        *,
        request_id: str,
        aligned_image: np.ndarray,
    ) -> StoredAlignedArtifact:
        """Persist aligned image and return metadata for API serialization."""
        if aligned_image.size == 0:
            raise InternalProcessingError(
                message="Aligned image is empty and cannot be persisted.",
                error_code="aligned_artifact_empty",
                details={"request_id": request_id},
            )

        request_dir = self.create_request_directory(request_id)
        artifact_path = request_dir / _ALIGNED_IMAGE_FILENAME

        try:
            was_written = cv2.imwrite(str(artifact_path), aligned_image)
        except cv2.error as error:
            raise InternalProcessingError(
                message="Failed to write aligned image artifact.",
                error_code="aligned_artifact_write_failed",
                details={
                    "request_id": request_id,
                    "path": artifact_path.as_posix(),
                    "opencv_error": str(error),
                },
            ) from error

        if not was_written:
            raise InternalProcessingError(
                message="Failed to write aligned image artifact.",
                error_code="aligned_artifact_write_failed",
                details={
                    "request_id": request_id,
                    "path": artifact_path.as_posix(),
                },
            )

        persisted_image = cv2.imread(
            str(artifact_path),
            cv2.IMREAD_UNCHANGED,
        )
        if persisted_image is None or persisted_image.size == 0:
            raise InternalProcessingError(
                message="Persisted aligned artifact is unreadable.",
                error_code="aligned_artifact_unreadable",
                details={
                    "request_id": request_id,
                    "path": artifact_path.as_posix(),
                },
            )

        height, width = persisted_image.shape[:2]
        channels = 1
        if persisted_image.ndim == 3:
            channels = int(persisted_image.shape[2])

        return StoredAlignedArtifact(
            path=artifact_path.as_posix(),
            height=int(height),
            width=int(width),
            channels=channels,
        )

    def draw_detections(
        self,
        *,
        aligned_image: np.ndarray,
        detections: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        """Render OCR detections over aligned image and return annotation."""
        if aligned_image.size == 0:
            raise InternalProcessingError(
                message=(
                    "Aligned image is empty and cannot be used for overlay."
                ),
                error_code="overlay_source_image_empty",
            )

        overlay_image = _ensure_drawable_image(aligned_image)

        for detection in detections:
            if not isinstance(detection, Mapping):
                continue

            top_left = _draw_detection_geometry(
                overlay_image,
                detection=detection,
            )
            _draw_detection_label(
                overlay_image,
                detection=detection,
                top_left=top_left,
            )

        return overlay_image

    def persist_detection_overlay(
        self,
        *,
        request_id: str,
        overlay_image: np.ndarray,
    ) -> StoredOverlayArtifact:
        """Persist detection overlay image and return artifact metadata."""
        if overlay_image.size == 0:
            raise InternalProcessingError(
                message="Detection overlay image is empty.",
                error_code="overlay_artifact_empty",
                details={"request_id": request_id},
            )

        request_dir = self.create_request_directory(request_id)
        artifact_path = request_dir / _OVERLAY_IMAGE_FILENAME

        try:
            was_written = cv2.imwrite(str(artifact_path), overlay_image)
        except cv2.error as error:
            raise InternalProcessingError(
                message="Failed to write detection overlay artifact.",
                error_code="overlay_artifact_write_failed",
                details={
                    "request_id": request_id,
                    "path": artifact_path.as_posix(),
                    "opencv_error": str(error),
                },
            ) from error

        if not was_written:
            raise InternalProcessingError(
                message="Failed to write detection overlay artifact.",
                error_code="overlay_artifact_write_failed",
                details={
                    "request_id": request_id,
                    "path": artifact_path.as_posix(),
                },
            )

        persisted_image = cv2.imread(
            str(artifact_path),
            cv2.IMREAD_UNCHANGED,
        )
        if persisted_image is None or persisted_image.size == 0:
            raise InternalProcessingError(
                message="Persisted overlay artifact is unreadable.",
                error_code="overlay_artifact_unreadable",
                details={
                    "request_id": request_id,
                    "path": artifact_path.as_posix(),
                },
            )

        height, width = persisted_image.shape[:2]
        channels = 1
        if persisted_image.ndim == 3:
            channels = int(persisted_image.shape[2])

        return StoredOverlayArtifact(
            path=artifact_path.as_posix(),
            height=int(height),
            width=int(width),
            channels=channels,
        )


def _ensure_drawable_image(source: np.ndarray) -> np.ndarray:
    """Copy source image into a drawable array for OpenCV overlays."""
    if source.ndim == 2:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    if source.ndim == 3 and source.shape[2] == 1:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    return source.copy()


def _draw_detection_geometry(
    image: np.ndarray,
    *,
    detection: Mapping[str, Any],
) -> tuple[int, int]:
    """Draw polygon/bounding-box geometry and return label anchor point."""
    polygon = _normalize_polygon(detection.get("polygon"))
    if polygon is not None:
        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=_OVERLAY_LINE_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        x_values = polygon[:, 0, 0]
        y_values = polygon[:, 0, 1]
        return int(np.min(x_values)), int(np.min(y_values))

    bbox = _normalize_bounding_box(detection.get("bounding_box"))
    if bbox is not None:
        x, y, width, height = bbox
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            color=_OVERLAY_LINE_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return top_left

    return 0, 0


def _draw_detection_label(
    image: np.ndarray,
    *,
    detection: Mapping[str, Any],
    top_left: tuple[int, int],
) -> None:
    """Draw one compact label for a detection near the geometry anchor."""
    raw_text = detection.get("text")
    if not isinstance(raw_text, str):
        return

    text = raw_text.strip()
    if text == "":
        return

    anchor_x, anchor_y = top_left
    text_x = max(0, anchor_x)
    text_y = max(14, anchor_y - 4)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        _OVERLAY_TEXT_FONT,
        _OVERLAY_TEXT_SCALE,
        _OVERLAY_TEXT_COLOR,
        _OVERLAY_TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def _normalize_polygon(raw_value: object) -> np.ndarray | None:
    """Normalize polygon payload into OpenCV polyline point format."""
    if not isinstance(raw_value, list):
        return None

    vertices: list[list[int]] = []
    for vertex in raw_value:
        if not isinstance(vertex, list | tuple):
            return None
        if len(vertex) != 2:
            return None

        x_value, y_value = vertex
        if not isinstance(x_value, int | float):
            return None
        if not isinstance(y_value, int | float):
            return None

        vertices.append([int(round(float(x_value))), int(round(float(y_value)))])

    if len(vertices) < 4:
        return None

    return np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))


def _normalize_bounding_box(
    raw_value: object,
) -> tuple[int, int, int, int] | None:
    """Normalize bounding-box payload into integer pixel dimensions."""
    if not isinstance(raw_value, list | tuple):
        return None
    if len(raw_value) != 4:
        return None

    x_value, y_value, width_value, height_value = raw_value
    if not isinstance(x_value, int | float):
        return None
    if not isinstance(y_value, int | float):
        return None
    if not isinstance(width_value, int | float):
        return None
    if not isinstance(height_value, int | float):
        return None

    width = int(round(float(width_value)))
    height = int(round(float(height_value)))
    if width <= 0 or height <= 0:
        return None

    x = int(round(float(x_value)))
    y = int(round(float(y_value)))
    return (x, y, width, height)


def _sanitize_request_id(request_id: str) -> str:
    """Normalize request ID for safe filesystem usage."""
    normalized = request_id.strip()
    if normalized == "":
        return "unknown-request"

    sanitized = "".join(
        character
        if character in _ALLOWED_REQUEST_ID_CHARACTERS
        else "_"
        for character in normalized
    )
    return sanitized or "unknown-request"


__all__ = [
    "ArtifactStorageManager",
    "StoredAlignedArtifact",
    "StoredOverlayArtifact",
]
