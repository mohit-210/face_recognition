from datetime import date
from collections import Counter

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.schemas.attendance import (
    AttendanceBurstScanRequest,
    AttendanceMarkRead,
    AttendanceRead,
    AttendanceScanRequest,
    AttendanceScanResponse,
)
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


@router.post("/scan-face-burst", response_model=AttendanceScanResponse)
def scan_face_burst_for_attendance(
    payload: AttendanceBurstScanRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    face_service = FaceService(db)
    attempts: list[dict] = []
    previous_image_base64: str | None = None

    for image_base64 in payload.images_base64:
        result = face_service.identify_in_company(
            company_id=current_user.company_id,
            image_base64=image_base64,
            device_id=payload.device_id,
            enforce_liveness=True,
            previous_image_base64=previous_image_base64,
            require_live_motion=True,
        )
        attempts.append(result)
        previous_image_base64 = image_base64

    verified_attempts = [r for r in attempts if r.get("verified") and r.get("user_id") is not None]
    total_samples = len(attempts)
    verified_samples = len(verified_attempts)
    if not verified_attempts:
        fallback_reason = "No stable live match"
        reason_counts = Counter(str(r.get("reason", "")) for r in attempts if r.get("reason"))
        if reason_counts:
            fallback_reason = reason_counts.most_common(1)[0][0]
        return AttendanceScanResponse(
            verified=False,
            user_id=None,
            name=None,
            confidence=0.0,
            liveness_score=0.0,
            reason=fallback_reason,
            attendance=None,
            samples_evaluated=total_samples,
            samples_verified=0,
            consensus_ratio=0.0,
        )

    user_counts = Counter(int(r["user_id"]) for r in verified_attempts)
    consensus_user_id, user_hit_count = user_counts.most_common(1)[0]
    consensus_ratio = float(user_hit_count / float(total_samples))
    min_verified_samples = max(1, int(payload.min_verified_samples))
    min_consensus_ratio = float(payload.min_consensus_ratio)

    if verified_samples < min_verified_samples or consensus_ratio < min_consensus_ratio:
        return AttendanceScanResponse(
            verified=False,
            user_id=None,
            name=None,
            confidence=0.0,
            liveness_score=0.0,
            reason="Low confidence across burst; retry with better lighting and slight head movement",
            attendance=None,
            samples_evaluated=total_samples,
            samples_verified=verified_samples,
            consensus_ratio=consensus_ratio,
        )

    consensus_attempts = [r for r in verified_attempts if int(r["user_id"]) == consensus_user_id]
    best_attempt = max(consensus_attempts, key=lambda r: float(r.get("confidence", 0.0)))
    avg_confidence = float(sum(float(r.get("confidence", 0.0)) for r in consensus_attempts) / len(consensus_attempts))
    avg_liveness = float(
        sum(float(r.get("liveness_score", 0.0)) for r in consensus_attempts) / len(consensus_attempts)
    )

    attendance = None
    if payload.mark_attendance:
        marked = AttendanceService(db).mark_verified_face(
            company_id=current_user.company_id,
            user_id=consensus_user_id,
        )
        attendance = AttendanceMarkRead.model_validate(marked)

    return AttendanceScanResponse(
        verified=True,
        user_id=consensus_user_id,
        name=best_attempt.get("name"),
        confidence=avg_confidence,
        liveness_score=avg_liveness,
        reason=str(best_attempt.get("reason", "Verified")),
        attendance=attendance,
        samples_evaluated=total_samples,
        samples_verified=verified_samples,
        consensus_ratio=consensus_ratio,
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
