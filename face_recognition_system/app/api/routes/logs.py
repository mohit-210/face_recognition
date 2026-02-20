from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.schemas.face import VerificationLogRead
from app.services.log_service import LogService

router = APIRouter(prefix="/logs", tags=["Logs"])


@router.get("", response_model=list[VerificationLogRead])
def get_logs(
    user_id: int | None = Query(default=None),
    date: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return LogService(db).list(company_id=current_user.company_id, user_id=user_id, date=date)
