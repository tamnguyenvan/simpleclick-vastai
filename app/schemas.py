from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


Point = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]


class SegmentRequest(BaseModel):
    """One interactive segmentation request.

    Points use image pixel coordinates with the origin at the top-left. The
    API accepts a standard base64 payload or a ``data:image/...;base64,...``
    data URL.
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        min_length=1,
        max_length=36_000_000,
        description="Base64-encoded PNG/JPEG image or a base64 image data URL.",
    )
    positive_points: list[Point] = Field(
        min_length=1,
        max_length=4096,
        description="At least one foreground point, each represented as [x, y].",
    )
    negative_points: list[Point] = Field(
        default_factory=list,
        max_length=4096,
        description="Optional background points, each represented as [x, y].",
    )
    threshold: FiniteFloat | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Probability threshold in the open interval (0, 1).",
    )


class SegmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    mask: str = Field(description="Base64-encoded binary PNG mask.")
    mask_format: Literal["png"]
    mask_shape: list[int] = Field(min_length=2, max_length=2)
    positive_points_used: list[list[int]]
    negative_points_used: list[list[int]]
    threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    service: str
    model: str
    model_state: str
    device: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, object]] | dict[str, object] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str
