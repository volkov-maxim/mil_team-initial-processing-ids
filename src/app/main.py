"""Application bootstrap and service-level routes."""

from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from app.core.config import load_settings
from app.core.logging import get_configured_app_logger

settings = load_settings()
logger = get_configured_app_logger()

app = FastAPI(title="mil_team-initial-processing-ids")

REQUEST_ID_HEADER = "X-Request-ID"


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach request ID to request context and response header."""
    incoming_request_id = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming_request_id or str(uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Return a minimal liveness payload for health checks."""
    logger.info(
        "health_check",
        extra={
            "device_mode": settings.device_mode,
            "request_id": request.state.request_id,
        },
    )
    return {"status": "ok"}
