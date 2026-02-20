from datetime import datetime

from pydantic import BaseModel


class UserCreate(BaseModel):
    company_id: int
    name: str
    employee_code: str
    password: str


class UserUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    password: str | None = None


class UserRead(BaseModel):
    id: int
    company_id: int
    name: str
    employee_code: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
