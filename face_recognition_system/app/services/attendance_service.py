from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories.attendance_repository import AttendanceRepository
from app.repositories.user_repository import UserRepository


class AttendanceService:
    def __init__(self, db: Session) -> None:
        self.repo = AttendanceRepository(db)
        self.users = UserRepository(db)

    def mark_verified_face(
        self,
        company_id: int,
        user_id: int,
        event_time: datetime | None = None,
    ) -> dict:
        ts = event_time or datetime.now(timezone.utc)
        attendance_date = ts.date()
        record = self.repo.get_for_day(company_id=company_id, user_id=user_id, attendance_date=attendance_date)

        if record is None:
            created = self.repo.create(
                company_id=company_id,
                user_id=user_id,
                attendance_date=attendance_date,
                first_check_in_at=ts,
                last_check_out_at=None,
                last_seen_at=ts,
                verification_count=1,
                status="in",
            )
            return {"action": "check_in", "record": created}

        verification_count = int(record.verification_count or 0) + 1
        updates: dict = {
            "last_seen_at": ts,
            "verification_count": verification_count,
        }
        current_status = (record.status or "").strip().lower()

        if current_status in {"present", "in"}:
            updates["last_check_out_at"] = ts
            updates["status"] = "out"
            action = "check_out"
        else:
            updates["status"] = "in"
            updates["last_check_out_at"] = None
            if record.first_check_in_at is None:
                updates["first_check_in_at"] = ts
                action = "check_in"
            else:
                action = "check_in_again"

        updated = self.repo.update(record, **updates)
        return {"action": action, "record": updated}

    def checkout(self, company_id: int, user_id: int, event_time: datetime | None = None) -> dict:
        user = self.users.get(user_id)
        if not user or user.company_id != company_id:
            raise HTTPException(status_code=404, detail="User not found in company")

        ts = event_time or datetime.now(timezone.utc)
        open_record = self.repo.get_latest_open(company_id=company_id, user_id=user_id)
        if open_record is None:
            raise HTTPException(status_code=409, detail="No open attendance session found")

        updated = self.repo.update(
            open_record,
            last_check_out_at=ts,
            last_seen_at=ts,
            verification_count=int(open_record.verification_count or 0) + 1,
            status="out",
        )
        return {"action": "check_out", "record": updated}

    def list(self, company_id: int, attendance_date=None, user_id: int | None = None):
        return self.repo.list(company_id=company_id, attendance_date=attendance_date, user_id=user_id)
