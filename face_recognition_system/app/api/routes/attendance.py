from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.schemas.attendance import AttendanceMarkRead, AttendanceRead, AttendanceScanRequest, AttendanceScanResponse
from app.services.attendance_service import AttendanceService
from app.services.face_service import FaceService

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.post("/scan-face", response_model=AttendanceScanResponse)
def scan_face_for_attendance(
    payload: AttendanceScanRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    identified = FaceService(db).identify_in_company(
        company_id=current_user.company_id,
        image_base64=payload.image_base64,
        device_id=payload.device_id,
        enforce_liveness=True,
        previous_image_base64=payload.previous_image_base64,
        require_live_motion=True,
    )
    attendance = None
    if payload.mark_attendance and identified.get("verified") and identified.get("user_id") is not None:
        marked = AttendanceService(db).mark_verified_face(
            company_id=current_user.company_id,
            user_id=int(identified["user_id"]),
        )
        attendance = AttendanceMarkRead.model_validate(marked)

    return AttendanceScanResponse(
        verified=bool(identified.get("verified")),
        user_id=identified.get("user_id"),
        name=identified.get("name"),
        confidence=float(identified.get("confidence", 0.0)),
        liveness_score=float(identified.get("liveness_score", 0.0)),
        reason=str(identified.get("reason", "")),
        attendance=attendance,
    )


@router.get("", response_model=list[AttendanceRead])
def list_attendance(
    attendance_date: date | None = Query(default=None),
    user_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return AttendanceService(db).list(
        company_id=current_user.company_id,
        attendance_date=attendance_date,
        user_id=user_id,
    )


@router.post("/users/{user_id}/checkout", response_model=AttendanceMarkRead)
def manual_checkout(
    user_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    marked = AttendanceService(db).checkout(
        company_id=current_user.company_id,
        user_id=user_id,
    )
    return AttendanceMarkRead.model_validate(marked)
