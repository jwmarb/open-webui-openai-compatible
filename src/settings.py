"""Application settings loaded from environment variables or .env file."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8")

    open_webui_url: str
    user_token: str
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout: int = Field(default=300, ge=10, le=3600)
    stream_empty_retry_max: int = Field(default=3, ge=0, le=10)
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        v = value.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = f"Invalid LOG_LEVEL: {value!r}. Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
            raise ValueError(msg)
        return v

    @field_validator("open_webui_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


# type: ignore[call-arg]  # pydantic-settings injects from env
settings = Settings()
