from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

from .errors import ApiError
from .logging_config import configure_logging, get_request_id, request_id_context
from .model_service import ModelService
from .schemas import (
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    SegmentRequest,
    SegmentResponse,
)
from .settings import Settings, get_settings
from .validation import decode_base64_image, decode_image, mask_as_png_base64, validate_points


logger = logging.getLogger("simpleclick.api")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "")
    return supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex


def _error_payload(
    status_code: int,
    code: str,
    message: str,
    details: object = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details),
        request_id=get_request_id(),
    ).model_dump(exclude_none=True)
    response = JSONResponse(status_code=status_code, content=body)
    response.headers["X-Request-ID"] = get_request_id()
    return response


def _validation_details(exc: RequestValidationError) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        details.append(
            {
                "field": location,
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "validation_error"),
            }
        )
    return details


async def _require_api_key(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    expected_value = expected.get_secret_value()
    if not api_key or not secrets.compare_digest(api_key, expected_value):
        raise ApiError(401, "authentication_required", "a valid X-API-Key header is required")


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    settings = settings or get_settings()
    service = service or ModelService(settings)
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if settings.api_key is None or not settings.api_key.get_secret_value().strip():
            logger.critical("SIMPLECLICK_API_KEY is not set; refusing to start")
            raise RuntimeError(
                "SIMPLECLICK_API_KEY is required and must not be empty. "
                "Set it in the environment or .env before starting the API."
            )
        logger.info("loading SimpleClick model")
        try:
            await asyncio.to_thread(service.load)
        except Exception:
            logger.exception("failed to load SimpleClick model")
            raise
        logger.info("SimpleClick model ready", extra={"device": service.device})
        yield

    application = FastAPI(
        title=settings.app_name,
        description="Interactive foreground/background segmentation powered by SimpleClick.",
        version="0.1.0",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.service = service

    allowed_origins = settings.allowed_cors_origins()
    if allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @application.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = _request_id(request)
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        response = None
        try:
            content_length = request.headers.get("content-length")
            if (
                content_length
                and content_length.isdigit()
                and int(content_length) > settings.max_request_body_bytes
            ):
                response = _error_payload(
                    413,
                    "request_too_large",
                    f"request body exceeds the {settings.max_request_body_bytes:,}-byte limit",
                )
                return response
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code if response else 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            request_id_context.reset(token)

    @application.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError):
        return _error_payload(exc.status_code, exc.code, exc.message, exc.details)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return _error_payload(
            422,
            "validation_error",
            "request validation failed",
            _validation_details(exc),
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed"
        return _error_payload(exc.status_code, f"http_{exc.status_code}", message)

    @application.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception):
        logger.exception("unhandled application error", exc_info=exc)
        return _error_payload(500, "internal_error", "an unexpected internal error occurred")

    health_router = APIRouter(tags=["health"])

    @health_router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
    async def liveness() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            model="simpleclick",
            model_state=service.state,
            device=service.device,
        )

    @health_router.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={503: {"model": ErrorResponse}},
        summary="Readiness probe",
    )
    async def readiness() -> HealthResponse:
        if service.state != "ready":
            raise ApiError(503, "model_not_ready", "SimpleClick model is not ready")
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            model="simpleclick",
            model_state=service.state,
            device=service.device,
        )

    application.include_router(health_router)

    segment_router = APIRouter(
        prefix="/v1",
        tags=["segmentation"],
        dependencies=[Depends(_require_api_key)],
    )

    @segment_router.post(
        "/segment",
        response_model=SegmentResponse,
        summary="Segment an image from interactive points",
        responses={
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def segment(payload: SegmentRequest) -> SegmentResponse:
        if service.state != "ready":
            raise ApiError(503, "model_not_ready", "SimpleClick model is not ready")

        image_bytes = decode_base64_image(payload.image, settings.max_image_bytes)
        image_rgb = decode_image(image_bytes, settings.max_image_pixels)
        height, width = image_rgb.shape[:2]
        positive_points = validate_points(
            payload.positive_points,
            width,
            height,
            "positive_points",
            settings.max_points_used,
            settings.max_input_points,
        )
        negative_points = validate_points(
            payload.negative_points,
            width,
            height,
            "negative_points",
            settings.max_points_used,
            settings.max_input_points,
        )
        threshold = (
            float(payload.threshold)
            if payload.threshold is not None
            else settings.default_threshold
        )

        try:
            mask = await asyncio.to_thread(
                service.segment,
                image_rgb,
                positive_points,
                negative_points,
                threshold,
            )
            # OpenCV's encoder is kept on the event-loop thread. In some
            # OpenCV/Python combinations, invoking it from an asyncio worker
            # thread can deadlock during native library initialization.
            encoded_mask = mask_as_png_base64(mask)
        except ApiError:
            raise
        except Exception as exc:
            logger.exception("SimpleClick inference failed")
            raise ApiError(500, "inference_failed", "SimpleClick inference failed") from exc

        return SegmentResponse(
            request_id=get_request_id(),
            mask=encoded_mask,
            mask_format="png",
            mask_shape=[height, width],
            positive_points_used=positive_points,
            negative_points_used=negative_points,
            threshold=threshold,
        )

    application.include_router(segment_router)
    return application


app = create_app()
