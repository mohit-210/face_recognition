from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.security import create_access_token, create_refresh_token, verify_password
from app.repositories.user_repository import UserRepository


class AuthService:
    def __init__(self, db: Session) -> None:
        self.users = UserRepository(db)

    def login(self, company_id: int, employee_code: str, password: str) -> dict:
        user = self.users.get_by_employee_code(company_id=company_id, employee_code=employee_code)
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if user.status != "active":
            raise HTTPException(status_code=403, detail="User inactive")

        return {
            "access_token": create_access_token(subject=user.employee_code, company_id=user.company_id),
            "refresh_token": create_refresh_token(subject=user.employee_code, company_id=user.company_id),
        }
