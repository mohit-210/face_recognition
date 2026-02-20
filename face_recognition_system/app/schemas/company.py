from datetime import datetime

from pydantic import BaseModel


class CompanyCreate(BaseModel):
    name: str
    organization_type: str = "company"


class CompanyRead(BaseModel):
    id: int
    name: str
    organization_type: str
    created_at: datetime

    model_config = {"from_attributes": True}
