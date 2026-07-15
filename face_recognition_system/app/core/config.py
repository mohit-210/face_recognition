from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Laravel Face Engine"
    face_engine_key: str = "change-face-engine-key"
    arcface_model_name: str = "buffalo_s"
    arcface_onnx_providers: str = "CPUExecutionProvider"
    arcface_ctx_id: int = Field(default=-1, ge=-1, le=8)
    arcface_det_size: int = Field(default=320, ge=320, le=1280)
    engine_max_image_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
