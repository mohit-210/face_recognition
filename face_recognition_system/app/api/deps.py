from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.repositories.user_repository import UserRepository

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_v1_prefix}/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            raise credentials_exception
        employee_code = payload.get("sub")
        company_id = payload.get("company_id")
        if employee_code is None or company_id is None:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc

    user = UserRepository(db).get_by_employee_code(company_id=company_id, employee_code=employee_code)
    if not user:
        raise credentials_exception
    return user


def enforce_company_scope(target_company_id: int, current_user=Depends(get_current_user)):
    if current_user.company_id != target_company_id:
        raise HTTPException(status_code=403, detail="Cross-company access denied")
    return current_user
