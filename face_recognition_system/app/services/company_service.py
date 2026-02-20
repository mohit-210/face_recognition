from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories.company_repository import CompanyRepository


class CompanyService:
    def __init__(self, db: Session) -> None:
        self.repo = CompanyRepository(db)

    @staticmethod
    def _normalize_org_type(organization_type: str) -> str:
        value = organization_type.strip().lower()
        if value not in {"company", "society"}:
            raise HTTPException(status_code=400, detail="organization_type must be 'company' or 'society'")
        return value

    def create(self, name: str, organization_type: str = "company"):
        org_type = self._normalize_org_type(organization_type)
        existing = [c for c in self.repo.list_all() if c.name.lower() == name.lower()]
        if existing:
            raise HTTPException(status_code=409, detail="Company already exists")
        return self.repo.create(name=name, organization_type=org_type)

    def list_all(self):
        return self.repo.list_all()

    def ensure_exists(self, company_id: int):
        company = self.repo.get(company_id)
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        return company
