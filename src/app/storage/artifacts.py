"""Artifact storage manager for request-scoped pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


__all__ = ["ArtifactStorageManager", "StoredAlignedArtifact"]
