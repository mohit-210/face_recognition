from datetime import datetime

from sqlalchemy.orm import Session

from app.repositories.log_repository import VerificationLogRepository


class LogService:
    def __init__(self, db: Session) -> None:
        self.repo = VerificationLogRepository(db)

    def write(self, **kwargs):
        return self.repo.create(**kwargs)

    def list(self, company_id: int, user_id: int | None = None, date: datetime | None = None):
        return self.repo.list(company_id=company_id, user_id=user_id, date=date)
