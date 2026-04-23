"""Application settings loaded from environment variables or .env file."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    open_webui_url: str
    user_token: str
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("open_webui_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


settings = Settings()
