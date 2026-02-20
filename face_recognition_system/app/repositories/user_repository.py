from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, **kwargs) -> User:
        user = User(**kwargs)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get(self, user_id: int) -> User | None:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_employee_code(self, company_id: int, employee_code: str) -> User | None:
        return (
            self.db.query(User)
            .filter(User.company_id == company_id, User.employee_code == employee_code)
            .first()
        )

    def list_by_company(self, company_id: int) -> list[User]:
        return self.db.query(User).filter(User.company_id == company_id).order_by(User.id.desc()).all()

    def update(self, user: User, **kwargs) -> User:
        for key, value in kwargs.items():
            if value is not None:
                setattr(user, key, value)
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user: User) -> None:
        self.db.delete(user)
        self.db.commit()
