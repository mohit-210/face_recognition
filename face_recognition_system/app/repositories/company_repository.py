from sqlalchemy.orm import Session

from app.models.company import Company


class CompanyRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, name: str, organization_type: str = "company") -> Company:
        company = Company(name=name, organization_type=organization_type)
        self.db.add(company)
        self.db.commit()
        self.db.refresh(company)
        return company

    def list_all(self) -> list[Company]:
        return self.db.query(Company).order_by(Company.id.desc()).all()

    def get(self, company_id: int) -> Company | None:
        return self.db.query(Company).filter(Company.id == company_id).first()
