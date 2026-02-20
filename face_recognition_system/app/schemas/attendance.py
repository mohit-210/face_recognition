from datetime import date, datetime

from pydantic import BaseModel, Field


class AttendanceRead(BaseModel):
    id: int
    company_id: int
    user_id: int
    attendance_date: date
    first_check_in_at: datetime | None
    last_check_out_at: datetime | None
    last_seen_at: datetime | None
    verification_count: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AttendanceMarkRead(BaseModel):
    action: str
    record: AttendanceRead


class AttendanceScanRequest(BaseModel):
    image_base64: str
    previous_image_base64: str | None = None
    device_id: str | None = None
    mark_attendance: bool = True


class AttendanceBurstScanRequest(BaseModel):
    images_base64: list[str] = Field(min_length=2, max_length=5)
    device_id: str | None = None
    mark_attendance: bool = True
    min_verified_samples: int = Field(default=2, ge=1, le=5)
    min_consensus_ratio: float = Field(default=0.67, ge=0.5, le=1.0)


class AttendanceScanResponse(BaseModel):
    verified: bool
    user_id: int | None
    name: str | None
    confidence: float
    liveness_score: float
    reason: str
    attendance: AttendanceMarkRead | None = None
    samples_evaluated: int | None = None
    samples_verified: int | None = None
    consensus_ratio: float | None = None
