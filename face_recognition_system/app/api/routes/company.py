from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.schemas.company import CompanyCreate, CompanyRead
from app.services.company_service import CompanyService

router = APIRouter(prefix="/company", tags=["Company"])


@router.post("", response_model=CompanyRead)
def create_company(payload: CompanyCreate, db: Session = Depends(get_db), _=Depends(get_current_user)):
    return CompanyService(db).create(payload.name)


@router.get("", response_model=list[CompanyRead])
def list_companies(db: Session = Depends(get_db), _=Depends(get_current_user)):
    return CompanyService(db).list_all()
