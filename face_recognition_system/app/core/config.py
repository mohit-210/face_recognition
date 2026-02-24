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
    liveness_threshold: float = Field(default=0.77, ge=0.0, le=1.0)
    attendance_liveness_threshold: float = Field(default=0.77, ge=0.0, le=1.0)
    attendance_recognition_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    fallback_liveness_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    attendance_min_motion_score: float = Field(default=0.52, ge=0.0, le=1.0)
    attendance_min_motion_diff: float = Field(default=1.6, ge=0.1, le=20.0)
    attendance_auto_checkout_after_minutes: int = Field(default=480, ge=1, le=1440)
    passive_antispoof_model_path: str = "./models/passive_antispoof_mini_fasnet.keras"
    passive_antispoof_onnx_path: str = "./models/passive_antispoof.onnx"
    passive_antispoof_backend: str = "onnx"
    passive_antispoof_onnx_providers: str = "OpenVINOExecutionProvider,CPUExecutionProvider"
    passive_antispoof_onnx_input_size: int = Field(default=128, ge=64, le=256)
    passive_antispoof_onnx_preprocess: str = "imagenet"
    passive_antispoof_onnx_live_index: int = Field(default=1, ge=0, le=4)
    passive_antispoof_onnx_temperature: float = Field(default=6.0, ge=0.5, le=8.0)
    passive_antispoof_debug_scores: bool = Field(default=False)
    passive_antispoof_calibration_path: str = "./models/passive_antispoof_calibration.json"
    passive_quality_gate_enabled: bool = Field(default=False)
    passive_calibration_max_eer: float = Field(default=0.75, ge=0.0, le=1.0)
    passive_calibration_min_auc: float = Field(default=0.45, ge=0.0, le=1.0)
    passive_blur_min_variance: float = Field(default=32.0, ge=1.0, le=5000.0)
    passive_lighting_min: float = Field(default=55.0, ge=0.0, le=255.0)
    passive_lighting_max: float = Field(default=205.0, ge=0.0, le=255.0)
    arcface_model_name: str = "buffalo_s"
    arcface_onnx_providers: str = "CPUExecutionProvider"
    arcface_ctx_id: int = Field(default=-1, ge=-1, le=8)
    arcface_det_size: int = Field(default=320, ge=320, le=1280)
    retinaface_min_score: float = Field(default=0.70, ge=0.0, le=1.0)

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
