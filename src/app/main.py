"""Application bootstrap and service-level routes."""

from fastapi import FastAPI

from app.core.config import load_settings
from app.core.logging import get_configured_app_logger

settings = load_settings()
logger = get_configured_app_logger()

app = FastAPI(title="mil_team-initial-processing-ids")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal liveness payload for health checks."""
    logger.info(
        "health_check",
        extra={"device_mode": settings.device_mode},
    )
    return {"status": "ok"}
