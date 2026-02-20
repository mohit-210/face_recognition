from datetime import date

from sqlalchemy.orm import Session

from app.models.attendance_record import AttendanceRecord


class AttendanceRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_for_day(self, company_id: int, user_id: int, attendance_date: date) -> AttendanceRecord | None:
        return (
            self.db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.company_id == company_id,
                AttendanceRecord.user_id == user_id,
                AttendanceRecord.attendance_date == attendance_date,
            )
            .first()
        )

    def get_latest_open(self, company_id: int, user_id: int) -> AttendanceRecord | None:
        return (
            self.db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.company_id == company_id,
                AttendanceRecord.user_id == user_id,
                AttendanceRecord.status.in_(["present", "in"]),
            )
            .order_by(AttendanceRecord.attendance_date.desc())
            .first()
        )

    def create(self, **kwargs) -> AttendanceRecord:
        record = AttendanceRecord(**kwargs)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def update(self, record: AttendanceRecord, **kwargs) -> AttendanceRecord:
        for key, value in kwargs.items():
            setattr(record, key, value)
        self.db.commit()
        self.db.refresh(record)
        return record

    def list(self, company_id: int, attendance_date: date | None = None, user_id: int | None = None) -> list[AttendanceRecord]:
        query = self.db.query(AttendanceRecord).filter(AttendanceRecord.company_id == company_id)
        if attendance_date is not None:
            query = query.filter(AttendanceRecord.attendance_date == attendance_date)
        if user_id is not None:
            query = query.filter(AttendanceRecord.user_id == user_id)
        return query.order_by(AttendanceRecord.attendance_date.desc(), AttendanceRecord.id.desc()).all()
