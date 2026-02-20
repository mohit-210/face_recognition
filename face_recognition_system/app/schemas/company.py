from datetime import datetime

from pydantic import BaseModel


class CompanyCreate(BaseModel):
    name: str


class CompanyRead(BaseModel):
    id: int
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}
