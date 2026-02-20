from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Face Recognition System"
    api_v1_prefix: str = "/api/v1"

    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "face_db"
    db_user: str = "face_user"
    db_password: str = "face_pass"

    jwt_secret_key: str = "change-me"
    jwt_refresh_secret_key: str = "change-me-refresh"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_minutes: int = 60 * 24 * 7

    session_secret_key: str = "change-admin-session-secret"

    recognition_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    liveness_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    attendance_liveness_threshold: float = Field(default=0.48, ge=0.0, le=1.0)
    attendance_min_motion_diff: float = Field(default=1.6, ge=0.1, le=20.0)
    attendance_auto_checkout_after_minutes: int = Field(default=480, ge=1, le=1440)

    model_path: str = "./models"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    @property
    def sqlalchemy_database_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
