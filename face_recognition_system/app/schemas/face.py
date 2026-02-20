from datetime import datetime

from pydantic import BaseModel, Field


class FaceRegisterRequest(BaseModel):
    user_id: int
    images_base64: list[str] = Field(min_length=3)


class FaceRegisterResponse(BaseModel):
    user_id: int
    embeddings_saved: int


class FaceVerifyRequest(BaseModel):
    company_id: int
    user_id: int
    image_base64: str
    mark_attendance: bool = True
    expected_challenge: str | None = None
    challenge_response: str | None = None
    previous_image_base64: str | None = None
    device_id: str | None = None


class FaceVerifyResponse(BaseModel):
    verified: bool
    confidence: float
    liveness_score: float
    reason: str


class VerificationLogRead(BaseModel):
    id: int
    user_id: int | None
    company_id: int
    confidence: float
    liveness_score: float
    success: bool
    timestamp: datetime
    device_id: str | None

    model_config = {"from_attributes": True}
