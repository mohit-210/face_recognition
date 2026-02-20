from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.repositories.company_repository import CompanyRepository
from app.repositories.user_repository import UserRepository


class UserService:
    def __init__(self, db: Session) -> None:
        self.users = UserRepository(db)
        self.companies = CompanyRepository(db)

    def create(self, company_id: int, name: str, employee_code: str, password: str):
        if not self.companies.get(company_id):
            raise HTTPException(status_code=404, detail="Company not found")
        if self.users.get_by_employee_code(company_id, employee_code):
            raise HTTPException(status_code=409, detail="Employee code already exists")
        return self.users.create(
            company_id=company_id,
            name=name,
            employee_code=employee_code,
            password_hash=get_password_hash(password),
            status="active",
        )

    def update(self, user_id: int, name: str | None, status: str | None, password: str | None):
        user = self.users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        update_data = {"name": name, "status": status}
        if password:
            update_data["password_hash"] = get_password_hash(password)
        return self.users.update(user, **update_data)

    def delete(self, user_id: int):
        user = self.users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        self.users.delete(user)

    def list_by_company(self, company_id: int):
        return self.users.list_by_company(company_id)

    def get(self, user_id: int):
        user = self.users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
