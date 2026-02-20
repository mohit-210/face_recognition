from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.schemas.face import FaceRegisterRequest, FaceRegisterResponse, FaceVerifyRequest, FaceVerifyResponse
from app.services.attendance_service import AttendanceService
from app.services.face_service import FaceService
from app.services.user_service import UserService

router = APIRouter(prefix="/face", tags=["Face"])


@router.post("/register", response_model=FaceRegisterResponse)
def register_face(payload: FaceRegisterRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    user = UserService(db).get(payload.user_id)
    if user.company_id != current_user.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-company access denied")
    saved = FaceService(db).register_embeddings(payload.user_id, payload.images_base64)
    return FaceRegisterResponse(user_id=payload.user_id, embeddings_saved=saved)


@router.post("/verify", response_model=FaceVerifyResponse)
def verify_face(payload: FaceVerifyRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if payload.company_id != current_user.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-company access denied")
    result = FaceService(db).verify(
        company_id=payload.company_id,
        user_id=payload.user_id,
        image_base64=payload.image_base64,
        expected_challenge=payload.expected_challenge,
        challenge_response=payload.challenge_response,
        previous_image_base64=payload.previous_image_base64,
        device_id=payload.device_id,
    )
    if payload.mark_attendance and result.get("verified"):
        AttendanceService(db).mark_verified_face(
            company_id=payload.company_id,
            user_id=payload.user_id,
        )
    return FaceVerifyResponse(**result)
