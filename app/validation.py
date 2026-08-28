from __future__ import annotations

import base64
import binascii
import math
from typing import Any

from .errors import ApiError


def sample_points(points: list[list[float]], limit: int) -> list[list[int]]:
    """Keep points distributed across a stroke while bounding model work."""

    if limit < 1:
        raise ValueError("point sampling limit must be positive")
    if len(points) <= limit:
        selected = points
    elif limit == 1:
        selected = [points[0]]
    else:
        step = (len(points) - 1) / (limit - 1)
        selected = [points[round(index * step)] for index in range(limit)]
    return [[int(round(point[0])), int(round(point[1]))] for point in selected]


def decode_base64_image(value: str, max_image_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise ApiError(422, "invalid_image", "image must be a non-empty base64 string")

    encoded = value
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ApiError(
                422,
                "invalid_image",
                "image data URL must contain a base64 payload",
            )

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(422, "invalid_image", "image is not valid base64") from exc

    if not image_bytes:
        raise ApiError(422, "invalid_image", "image is empty")
    if len(image_bytes) > max_image_bytes:
        limit_mb = max_image_bytes / (1024 * 1024)
        raise ApiError(
            413,
            "image_too_large",
            f"image exceeds the {limit_mb:g} MB limit",
        )
    return image_bytes


def decode_image(image_bytes: bytes, max_image_pixels: int) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment configuration error
        raise RuntimeError("OpenCV and NumPy are required for image decoding") from exc

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiError(422, "invalid_image", "image could not be decoded; use PNG or JPEG")

    height, width = image.shape[:2]
    if width * height > max_image_pixels:
        raise ApiError(
            413,
            "image_too_large",
            f"image dimensions exceed the {max_image_pixels:,}-pixel limit",
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_points(
    value: list[list[float]],
    width: int,
    height: int,
    field_name: str,
    limit: int,
    max_input_points: int,
) -> list[list[int]]:
    if len(value) > max_input_points:
        raise ApiError(
            422,
            "too_many_points",
            f"{field_name} cannot contain more than {max_input_points} points",
        )

    points: list[list[float]] = []
    for index, point in enumerate(value):
        if len(point) != 2:
            raise ApiError(422, "invalid_points", f"{field_name}[{index}] must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ApiError(
                422,
                "invalid_points",
                f"{field_name}[{index}] coordinates must be finite numbers",
            )
        if not (0 <= x < width and 0 <= y < height):
            raise ApiError(
                422,
                "point_out_of_bounds",
                f"{field_name}[{index}] is outside the image bounds",
                {"image_width": width, "image_height": height},
            )
        points.append([x, y])

    sampled = sample_points(points, limit)
    return [
        [
            min(width - 1, max(0, point[0])),
            min(height - 1, max(0, point[1])),
        ]
        for point in sampled
    ]


def mask_as_png_base64(mask: Any) -> str:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment configuration error
        raise RuntimeError("OpenCV and NumPy are required for mask encoding") from exc

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise RuntimeError("model returned a mask with an invalid shape")

    mask_image = np.where(mask_array, 255, 0).astype(np.uint8)
    encoded, buffer = cv2.imencode(
        ".png", mask_image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not encoded:
        raise RuntimeError("failed to encode segmentation mask")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
