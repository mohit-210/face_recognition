from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    organization_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="company",
        server_default=text("'company'"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    users = relationship("User", back_populates="company", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="company", cascade="all, delete-orphan")
