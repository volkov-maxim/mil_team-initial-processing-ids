"""API routes for document processing endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request

from app.api.errors import BadRequestErrorResponse
from app.api.errors import InternalServerErrorResponse
from app.api.errors import UnprocessableEntityErrorResponse
from app.api.schemas import ProcessDocumentRequest
from app.api.schemas import ProcessDocumentResponse
from app.api.schemas import parse_process_document_request
from app.pipeline.context import PipelineContext
from app.pipeline.processing import process_document_pipeline

router = APIRouter(prefix="/v1", tags=["document-processing"])


@router.post(
    "/process-document",
    response_model=ProcessDocumentResponse,
    responses={
        400: {"model": BadRequestErrorResponse},
        422: {"model": UnprocessableEntityErrorResponse},
        500: {"model": InternalServerErrorResponse},
    },
)
async def process_document(
    request: Request,
    payload: Annotated[
        ProcessDocumentRequest,
        Depends(parse_process_document_request),
    ],
) -> ProcessDocumentResponse:
    """Process a single document image and return placeholder output."""
    request_id = getattr(request.state, "request_id", "unknown")
    image_bytes = await payload.image.read()

    context = PipelineContext(
        request_id=request_id,
        image_bytes=image_bytes,
        document_type_hint=payload.document_type_hint,
        use_external_fallback=payload.use_external_fallback,
        metadata={"content_type": payload.image.content_type},
    )
    pipeline_result = process_document_pipeline(context)

    aligned_image = pipeline_result.aligned_image
    if aligned_image is None:
        aligned_image = f"artifacts/{request_id}/aligned-placeholder.png"

    return ProcessDocumentResponse(
        request_id=pipeline_result.request_id,
        document_type_detected=pipeline_result.document_type_detected,
        aligned_image=aligned_image,
        detections=pipeline_result.detections,
        fields=pipeline_result.fields,
        field_confidence=pipeline_result.field_confidence,
        validation_flags=pipeline_result.validation_flags,
        processing_metadata=pipeline_result.processing_metadata,
    )


__all__ = ["router"]
