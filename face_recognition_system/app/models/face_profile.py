from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="face_profiles")
