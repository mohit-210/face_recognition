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


def _avg_debug_timings(samples: list[dict | None]) -> dict[str, float] | None:
    rows = [row for row in samples if isinstance(row, dict) and row]
    if not rows:
        return None
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = round(sum(vals) / float(len(vals)), 2)
    return out or None


@router.post("/scan-face", response_model=AttendanceScanResponse)
def scan_face_for_attendance(
    payload: AttendanceScanRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    strict_attendance = bool(payload.mark_attendance)
    if strict_attendance and not payload.previous_image_base64:
        return AttendanceScanResponse(
            verified=False,
            user_id=None,
            name=None,
            confidence=0.0,
            liveness_score=0.0,
            liveness_threshold_used=None,
            model_used=None,
            reason="Need consecutive live frames; provide previous frame",
            attendance=None,
            debug_timings=None,
        )
    # Allow fast embedding path even in strict attendance mode to keep mobile latency low.
    effective_fast_mode = bool(payload.fast_mode)
    identified = FaceService(db).identify_in_company(
        company_id=current_user.company_id,
        image_base64=payload.image_base64,
        device_id=payload.device_id,
        enforce_liveness=True,
        previous_image_base64=payload.previous_image_base64,
        require_live_motion=strict_attendance,
        fast_mode=effective_fast_mode,
        debug_timing=payload.debug_timing,
        strict_attendance=strict_attendance,
        # Reuse bbox across consecutive frames to avoid repeated heavy detection.
        allow_bbox_reuse=True,
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
        liveness_threshold_used=(
            float(identified["liveness_threshold_used"])
            if identified.get("liveness_threshold_used") is not None
            else None
        ),
        model_used=identified.get("model_used"),
        reason=str(identified.get("reason", "")),
        attendance=attendance,
        debug_timings=identified.get("debug_timings"),
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
    min_verified_samples = max(1, int(payload.min_verified_samples))
    min_consensus_ratio = float(payload.min_consensus_ratio)
    total_samples = len(payload.images_base64)
    verified_so_far = 0
    previous_bbox: list[int] | None = None
    strict_attendance = bool(payload.mark_attendance)
    # Allow fast embedding path even in strict attendance mode to keep mobile latency low.
    effective_fast_mode = bool(payload.fast_mode)

    for idx, image_base64 in enumerate(payload.images_base64):
        result = face_service.identify_in_company(
            company_id=current_user.company_id,
            image_base64=image_base64,
            device_id=payload.device_id,
            enforce_liveness=True,
            previous_image_base64=previous_image_base64,
            require_live_motion=strict_attendance,
            previous_bbox=previous_bbox,
            fast_mode=effective_fast_mode,
            debug_timing=payload.debug_timing,
            strict_attendance=strict_attendance,
            # Reuse bbox across consecutive frames to avoid repeated heavy detection.
            allow_bbox_reuse=True,
        )
        attempts.append(result)
        if isinstance(result.get("_bbox"), list):
            previous_bbox = result.get("_bbox")
        if result.get("verified") and result.get("user_id") is not None:
            verified_so_far += 1
        previous_image_base64 = image_base64

        remaining = total_samples - (idx + 1)
        if verified_so_far + remaining < min_verified_samples:
            break

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
            liveness_threshold_used=next(
                (
                    float(r["liveness_threshold_used"])
                    for r in attempts
                    if r.get("liveness_threshold_used") is not None
                ),
                None,
            ),
            model_used=next((r.get("model_used") for r in attempts if r.get("model_used")), None),
            reason=fallback_reason,
            attendance=None,
            samples_evaluated=total_samples,
            samples_verified=0,
            consensus_ratio=0.0,
            debug_timings=_avg_debug_timings([r.get("debug_timings") for r in attempts]),
        )

    user_counts = Counter(int(r["user_id"]) for r in verified_attempts)
    consensus_user_id, user_hit_count = user_counts.most_common(1)[0]
    consensus_ratio = float(user_hit_count / float(verified_samples))

    if verified_samples < min_verified_samples or consensus_ratio < min_consensus_ratio:
        return AttendanceScanResponse(
            verified=False,
            user_id=None,
            name=None,
            confidence=0.0,
            liveness_score=0.0,
            liveness_threshold_used=next(
                (
                    float(r["liveness_threshold_used"])
                    for r in attempts
                    if r.get("liveness_threshold_used") is not None
                ),
                None,
            ),
            model_used=next((r.get("model_used") for r in attempts if r.get("model_used")), None),
            reason="Low confidence across burst; retry with better lighting and slight head movement",
            attendance=None,
            samples_evaluated=total_samples,
            samples_verified=verified_samples,
            consensus_ratio=consensus_ratio,
            debug_timings=_avg_debug_timings([r.get("debug_timings") for r in attempts]),
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
        liveness_threshold_used=(
            float(best_attempt["liveness_threshold_used"])
            if best_attempt.get("liveness_threshold_used") is not None
            else None
        ),
        model_used=best_attempt.get("model_used"),
        reason=str(best_attempt.get("reason", "Verified")),
        attendance=attendance,
        samples_evaluated=total_samples,
        samples_verified=verified_samples,
        consensus_ratio=consensus_ratio,
        debug_timings=_avg_debug_timings([r.get("debug_timings") for r in attempts]),
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
