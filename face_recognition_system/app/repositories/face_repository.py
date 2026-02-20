import numpy as np
from sqlalchemy.orm import Session

from app.models.face_profile import FaceProfile


class FaceProfileRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_embedding(self, user_id: int, embedding: np.ndarray) -> FaceProfile:
        profile = FaceProfile(user_id=user_id, embedding_vector=embedding.astype(np.float32).tobytes())
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def get_user_embeddings(self, user_id: int) -> list[np.ndarray]:
        profiles = self.db.query(FaceProfile).filter(FaceProfile.user_id == user_id).all()
        return [np.frombuffer(item.embedding_vector, dtype=np.float32) for item in profiles]

    def get_embeddings_by_user_ids(self, user_ids: list[int]) -> dict[int, list[np.ndarray]]:
        if not user_ids:
            return {}
        profiles = self.db.query(FaceProfile).filter(FaceProfile.user_id.in_(user_ids)).all()
        grouped: dict[int, list[np.ndarray]] = {}
        for item in profiles:
            grouped.setdefault(item.user_id, []).append(np.frombuffer(item.embedding_vector, dtype=np.float32))
        return grouped

    def clear_user_embeddings(self, user_id: int) -> None:
        self.db.query(FaceProfile).filter(FaceProfile.user_id == user_id).delete()
        self.db.commit()
