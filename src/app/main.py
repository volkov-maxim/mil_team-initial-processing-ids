"""Application bootstrap and service-level routes."""

from collections.abc import Awaitable, Callable
from typing import TypeAlias
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api.errors import ApiErrorResponse
from app.api.errors import BadRequestErrorResponse
from app.api.errors import InternalServerErrorResponse
from app.api.errors import UnprocessableEntityErrorResponse
from app.core.config import load_settings
from app.core.exceptions import AppCoreError
from app.core.logging import get_configured_app_logger

settings = load_settings()
logger = get_configured_app_logger()

app = FastAPI(title="mil_team-initial-processing-ids")

REQUEST_ID_HEADER = "X-Request-ID"
INTERNAL_SERVER_STATUS = 500

ErrorResponseModel: TypeAlias = type[ApiErrorResponse]


def _resolve_error_model(status_code: int) -> ErrorResponseModel:
    """Resolve typed error response model by status code."""
    status_to_model: dict[int, ErrorResponseModel] = {
        400: BadRequestErrorResponse,
        422: UnprocessableEntityErrorResponse,
        INTERNAL_SERVER_STATUS: InternalServerErrorResponse,
    }
    return status_to_model.get(status_code, InternalServerErrorResponse)


@app.exception_handler(AppCoreError)
async def app_core_error_handler(
    request: Request,
    exc: AppCoreError,
) -> JSONResponse:
    """Map core typed exceptions to API error envelopes."""
    request_id = getattr(request.state, "request_id", "unknown")
    response_model = _resolve_error_model(exc.status_code)
    payload = response_model(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )

    logger.warning(
        "handled_app_core_error",
        extra={
            "request_id": request_id,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(exclude_none=True),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return stable internal error envelopes for unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    payload = InternalServerErrorResponse()

    logger.exception(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=INTERNAL_SERVER_STATUS,
        content=payload.model_dump(exclude_none=True),
    )


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
