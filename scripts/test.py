#!/usr/bin/env python
"""Smoke-test a deployed SimpleClick API over HTTP.

Usage:
    python scripts/test-deployed.py [BASE_URL] [--api-key KEY] [--image IMG.png] [--x X] [--y Y]

Defaults:
    BASE_URL   from SIMPLECLICK_BASE_URL or http://localhost:8000
    API key    from SIMPLECLICK_API_KEY (only sent if set)
    --image    when omitted, a synthetic test image is generated
    --x / --y  click coordinates; default to the image centre

Dependencies: httpx, numpy, opencv-python-headless (runtime deps of the service).
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

SEGMENT_ENDPOINT = "/v1/segment"
HEALTH_ENDPOINT = "/health/ready"


def generate_test_image(size: int = 256) -> np.ndarray:
    """Return an RGB image: a bright circle on a dark background."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    centre = size // 2
    radius = size // 4
    cv2.circle(image, (centre, centre), radius, (255, 255, 255), thickness=-1)
    return image


def encode_image(image_rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode test image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_payload(image_b64: str, x: int, y: int) -> dict:
    return {
        "image": image_b64,
        "positive_points": [[x, y]],
        "negative_points": [],
    }


def check_ready(client: httpx.Client, base_url: str, api_key: str | None) -> None:
    headers = {"X-API-Key": api_key} if api_key else {}
    ready = client.get(f"{base_url}{HEALTH_ENDPOINT}", headers=headers)
    if ready.status_code != 200:
        print(f"FAIL: {HEALTH_ENDPOINT} returned {ready.status_code}: {ready.text}")
        sys.exit(1)
    print(f"ready: status={ready.status_code} body={ready.json()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_url",
        nargs="?",
        default=os.getenv("SIMPLECLICK_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument("--api-key", default=os.getenv("SIMPLECLICK_API_KEY"))
    parser.add_argument("--image", default=None, help="path to a PNG/JPEG to segment")
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.image:
        image_rgb = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image_rgb is None:
            print(f"FAIL: could not read image: {args.image}")
            return 1
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = generate_test_image()

    height, width = image_rgb.shape[:2]
    click_x = args.x if args.x is not None else width // 2
    click_y = args.y if args.y is not None else height // 2
    if not (0 <= click_x < width and 0 <= click_y < height):
        print(f"FAIL: click ({click_x}, {click_y}) is outside image {width}x{height}")
        return 1

    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(timeout=60.0) as client:
        check_ready(client, base_url, args.api_key)

        payload = build_payload(encode_image(image_rgb), click_x, click_y)
        print(f"POST {base_url}{SEGMENT_ENDPOINT} "
              f"image={width}x{height} click=({click_x},{click_y})")
        response = client.post(
            f"{base_url}{SEGMENT_ENDPOINT}", json=payload, headers=headers
        )

        if response.status_code != 200:
            print(f"FAIL: segment returned {response.status_code}: {response.text}")
            return 1

        body = response.json()
        print(
            f"ok: request_id={body['request_id']} shape={body['mask_shape']} "
            f"threshold={body['threshold']} points_used={body['positive_points_used']}"
        )

        if not body["positive_points_used"]:
            print("FAIL: empty positive_points_used")
            return 1

        mask_bytes = base64.b64decode(body["mask"])
        mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print("FAIL: returned mask is not a decodable image")
            return 1
        mask_h, mask_w = mask.shape[:2]
        if (mask_w, mask_h) != tuple(body["mask_shape"]):
            print(f"FAIL: mask size ({mask_w},{mask_h}) != {body['mask_shape']}")
            return 1

    print("PASS: deployed SimpleClick API responded correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
