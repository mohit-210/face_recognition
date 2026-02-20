from fastapi import APIRouter, Depends, HTTPException
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_access_token
from app.database.session import get_db
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse
from app.services.auth_service import AuthService

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    tokens = AuthService(db).login(payload.company_id, payload.employee_code, payload.password)
    return TokenResponse(**tokens)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest):
    try:
        decoded = jwt.decode(payload.refresh_token, settings.jwt_refresh_secret_key, algorithms=[settings.jwt_algorithm])
        if decoded.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        subject = decoded["sub"]
        company_id = decoded["company_id"]
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from exc

    return TokenResponse(
        access_token=create_access_token(subject=subject, company_id=company_id),
        refresh_token=payload.refresh_token,
    )
