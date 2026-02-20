from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.schemas.user import UserCreate, UserRead, UserUpdate
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("", response_model=UserRead)
def create_user(payload: UserCreate, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if payload.company_id != current_user.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-company access denied")
    return UserService(db).create(payload.company_id, payload.name, payload.employee_code, payload.password)


@router.put("/{id}", response_model=UserRead)
def update_user(id: int, payload: UserUpdate, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    service = UserService(db)
    user = service.get(id)
    if user.company_id != current_user.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-company access denied")
    return service.update(id, payload.name, payload.status, payload.password)


@router.delete("/{id}", status_code=204)
def delete_user(id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    service = UserService(db)
    user = service.get(id)
    if user.company_id != current_user.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-company access denied")
    service.delete(id)


@router.get("", response_model=list[UserRead])
def list_users(company_id: int = Query(...), db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if company_id != current_user.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-company access denied")
    return UserService(db).list_by_company(company_id)
