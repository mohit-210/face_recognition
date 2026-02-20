from datetime import datetime

from sqlalchemy.orm import Session

from app.models.verification_log import VerificationLog


class VerificationLogRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs) -> VerificationLog:
        log = VerificationLog(**kwargs)
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def list(self, company_id: int, user_id: int | None = None, date: datetime | None = None) -> list[VerificationLog]:
        query = self.db.query(VerificationLog).filter(VerificationLog.company_id == company_id)
        if user_id is not None:
            query = query.filter(VerificationLog.user_id == user_id)
        if date is not None:
            query = query.filter(VerificationLog.timestamp >= date)
        return query.order_by(VerificationLog.timestamp.desc()).all()
