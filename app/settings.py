from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings.

    Every setting can be overridden with the ``SIMPLECLICK_`` prefix, for
    example ``SIMPLECLICK_API_KEY`` or ``SIMPLECLICK_LOG_LEVEL``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIMPLECLICK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "SimpleClick Segmentation API"
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: str = ""
    api_key: SecretStr | None = None

    simpleclick_root: Path = Path("/opt/SimpleClick")
    checkpoint_path: Path = Path(
        "/opt/SimpleClick/weights/simpleclick_models/cocolvis_vit_huge.pth"
    )
    checkpoint_id: str = "1GXk6q5fwKo2twkY5ZZGjVKCgJv7XeLAW"
    model_device: str = "auto"
    default_threshold: float = Field(default=0.49, gt=0, lt=1)
    max_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=40_000_000, ge=1)
    max_input_points: int = Field(default=4096, ge=1, le=100_000)
    max_points_used: int = Field(default=24, ge=1, le=10_000)
    max_longest_size: int = Field(default=800, ge=1)
    model_input_size: int = Field(default=448, ge=1)
    model_with_flip: bool = True
    enable_docs: bool = True

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log_level must be a standard Python logging level")
        return value

    @field_validator("api_key", mode="before")
    @classmethod
    def empty_api_key_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("model_device")
    @classmethod
    def normalize_model_device(cls, value: str) -> str:
        value = value.lower()
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("model_device must be one of: auto, cpu, cuda")
        return value

    @property
    def max_request_body_bytes(self) -> int:
        """Allow base64 expansion plus a small JSON/point overhead."""

        encoded_image_bytes = ((self.max_image_bytes + 2) // 3) * 4
        return encoded_image_bytes + 1 * 1024 * 1024

    def allowed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
