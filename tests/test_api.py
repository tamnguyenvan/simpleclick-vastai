import base64
import asyncio

import httpx

from app import main as app_module
from app.main import create_app
from app.settings import Settings


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class FakeService:
    state = "ready"
    device = "cpu"

    def load(self):
        return None

    def segment(self, image, positive_points, negative_points, threshold):
        assert image.shape[:2] == (1, 1)
        assert positive_points == [[0, 0]]
        assert negative_points == []
        assert threshold == 0.49
        return [[True]]


def make_app(api_key=None):
    settings = Settings(
        environment="test",
        log_json=False,
        api_key=api_key,
        max_image_bytes=1024,
        max_image_pixels=100,
    )
    return create_app(settings, FakeService())


def request(app, method, path, **kwargs):
    async def direct_to_thread(function, *args, **call_kwargs):
        return function(*args, **call_kwargs)

    async def send_request():
        original_to_thread = app_module.asyncio.to_thread
        app_module.asyncio.to_thread = direct_to_thread
        try:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    return await client.request(method, path, **kwargs)
        finally:
            app_module.asyncio.to_thread = original_to_thread

    return asyncio.run(send_request())


def post(app, **kwargs):
    return request(app, "POST", "/v1/segment", **kwargs)


def test_health_endpoints_report_liveness_and_readiness():
    app = make_app()
    live = request(app, "GET", "/health/live")
    ready = request(app, "GET", "/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["model_state"] == "ready"


def test_segment_returns_mask_and_request_id():
    response = post(
        make_app(),
        json={
            "image": PNG_1X1,
            "positive_points": [[0, 0]],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mask_format"] == "png"
    assert payload["mask_shape"] == [1, 1]
    assert len(base64.b64decode(payload["mask"])) > 0
    assert response.headers["x-request-id"] == payload["request_id"]


def test_validation_errors_use_stable_error_shape():
    response = post(
        make_app(),
        json={"image": PNG_1X1, "positive_points": []},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["request_id"]


def test_api_key_is_required_when_configured():
    app = make_app("secret")
    response = post(
        app,
        json={"image": PNG_1X1, "positive_points": [[0, 0]]},
    )
    authorized = post(
        app,
        headers={"X-API-Key": "secret"},
        json={"image": PNG_1X1, "positive_points": [[0, 0]]},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"
    assert authorized.status_code == 200
