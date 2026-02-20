from functools import lru_cache
import time

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories.face_repository import FaceProfileRepository
from app.services.log_service import LogService
from app.services.user_service import UserService
from app.vision.detector import FaceDetector
from app.vision.recognition import RecognitionEngine

COMPANY_INDEX_TTL_SECONDS = 20.0
_company_index_cache: dict[int, dict] = {}
MAX_REGISTER_IMAGES = 12
REGISTER_DUPLICATE_THRESHOLD_FLOOR = 0.55
REGISTER_DUPLICATE_THRESHOLD_MARGIN = 0.05


@lru_cache(maxsize=1)
def get_recognition_engine() -> RecognitionEngine:
    # Cache model objects to avoid loading deep models on every API request.
    return RecognitionEngine()


class FaceService:
    def __init__(self, db: Session) -> None:
        self.user_service = UserService(db)
        self.face_repo = FaceProfileRepository(db)
        self.log_service = LogService(db)
        self.engine = get_recognition_engine()
        self.detector = FaceDetector()

    @staticmethod
    def _invalidate_company_index(company_id: int) -> None:
        _company_index_cache.pop(company_id, None)

    def _get_company_index(self, company_id: int) -> list[dict]:
        now = time.monotonic()
        cached = _company_index_cache.get(company_id)
        if cached and (now - float(cached.get("ts", 0.0))) < COMPANY_INDEX_TTL_SECONDS:
            return cached["entries"]

        users = [u for u in self.user_service.list_by_company(company_id) if u.status == "active"]
        user_ids = [u.id for u in users]
        embeddings_by_user = self.face_repo.get_embeddings_by_user_ids(user_ids)

        entries = []
        for user in users:
            embeddings = embeddings_by_user.get(user.id, [])
            if not embeddings:
                continue
            entries.append({"user_id": user.id, "name": user.name, "embeddings": embeddings})

        _company_index_cache[company_id] = {"ts": now, "entries": entries}
        return entries

    def _find_duplicate_user_in_company(
        self,
        company_id: int,
        user_id: int,
        candidate_embeddings: list,
    ) -> tuple[int | None, str | None, float]:
        users = self.user_service.list_by_company(company_id)
        other_users = [u for u in users if u.id != user_id]
        if not other_users:
            return None, None, 0.0

        embeddings_by_user = self.face_repo.get_embeddings_by_user_ids([u.id for u in other_users])
        if not embeddings_by_user:
            return None, None, 0.0

        duplicate_threshold = max(
            REGISTER_DUPLICATE_THRESHOLD_FLOOR,
            self.engine.settings.recognition_threshold + REGISTER_DUPLICATE_THRESHOLD_MARGIN,
        )
        best_similarity = 0.0
        best_user = None

        for other_user in other_users:
            enrolled = embeddings_by_user.get(other_user.id, [])
            if not enrolled:
                continue
            similarity = max(
                self.engine.cosine_similarity(candidate, saved)
                for candidate in candidate_embeddings
                for saved in enrolled
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_user = other_user
            if similarity >= duplicate_threshold:
                return other_user.id, other_user.name, float(similarity)

        if best_user and best_similarity >= duplicate_threshold:
            return best_user.id, best_user.name, float(best_similarity)
        return None, None, float(best_similarity)

    def register_embeddings(self, user_id: int, images_base64: list[str]) -> int:
        user = self.user_service.get(user_id)
        if user.status != "active":
            raise HTTPException(status_code=400, detail="User inactive")

        embeddings: list = []
        # Limit processing for responsive UX when users capture too many frames.
        for image_b64 in images_base64[:MAX_REGISTER_IMAGES]:
            image = self.detector.decode_image(image_b64)
            try:
                emb, face_count = self.engine.embedder.get_best_embedding(image)
            except ValueError:
                continue
            if face_count != 1:
                continue
            embeddings.append(emb)

        if not embeddings:
            raise HTTPException(status_code=400, detail="No valid face embedding could be generated from provided images")

        duplicate_user_id, duplicate_user_name, duplicate_similarity = self._find_duplicate_user_in_company(
            company_id=user.company_id,
            user_id=user_id,
            candidate_embeddings=embeddings,
        )
        if duplicate_user_id is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Face already enrolled to {duplicate_user_name} "
                    f"(user_id={duplicate_user_id}, confidence={duplicate_similarity:.3f})"
                ),
            )

        self.face_repo.clear_user_embeddings(user_id)
        count = self.face_repo.add_embeddings_bulk(user_id=user_id, embeddings=embeddings)
        self._invalidate_company_index(user.company_id)
        return count

    def verify(
        self,
        company_id: int,
        user_id: int,
        image_base64: str,
        expected_challenge: str | None,
        challenge_response: str | None,
        previous_image_base64: str | None,
        device_id: str | None,
    ) -> dict:
        user = self.user_service.get(user_id)
        if user.company_id != company_id:
            raise HTTPException(status_code=403, detail="User does not belong to company")
        if user.status != "active":
            raise HTTPException(status_code=403, detail="User inactive")

        enrolled_embeddings = self.face_repo.get_user_embeddings(user_id)
        if not enrolled_embeddings:
            raise HTTPException(status_code=404, detail="No face profile registered")

        image = self.detector.decode_image(image_base64)
        previous = self.detector.decode_image(previous_image_base64) if previous_image_base64 else None

        verified, confidence, liveness_score, reason = self.engine.verify(
            image,
            enrolled_embeddings,
            expected_challenge,
            challenge_response,
            previous,
        )

        self.log_service.write(
            user_id=user_id,
            company_id=company_id,
            confidence=confidence,
            liveness_score=liveness_score,
            success=verified,
            device_id=device_id,
        )

        return {
            "verified": verified,
            "confidence": confidence,
            "liveness_score": liveness_score,
            "reason": reason,
        }

    def identify_in_company(
        self,
        company_id: int,
        image_base64: str,
        device_id: str | None,
        enforce_liveness: bool = True,
    ) -> dict:
        image = self.detector.decode_image(image_base64)
        liveness_score = 1.0
        if not enforce_liveness:
            try:
                query_emb, _ = self.engine.embedder.get_best_embedding(image)
            except ValueError as exc:
                reason = str(exc) if str(exc) else "Unable to compute embedding"
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": 0.0,
                    "reason": reason,
                }
        else:
            detections = self.detector.detect(image)
            if not detections:
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": 0.0,
                    "reason": "No face detected",
                }
            if len(detections) > 1:
                ranked = sorted(
                    detections,
                    key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                    reverse=True,
                )
                p = ranked[0]["bbox"]
                s = ranked[1]["bbox"]
                p_area = max(1, (p[2] - p[0]) * (p[3] - p[1]))
                s_area = max(1, (s[2] - s[0]) * (s[3] - s[1]))
                if (s_area / float(p_area)) >= 0.35:
                    return {
                        "verified": False,
                        "user_id": None,
                        "name": None,
                        "confidence": 0.0,
                        "liveness_score": 0.0,
                        "reason": "Multiple faces detected",
                    }
                detections = [ranked[0]]

            det = detections[0]
            face = self.detector.crop_face(image, det["bbox"])
            liveness_score = self.engine.liveness.score(face, det.get("landmarks", {}))
            if liveness_score < self.engine.settings.liveness_threshold:
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": liveness_score,
                    "reason": "Liveness failed",
                }

            try:
                query_emb = self.engine.embedder.get_embedding(face)
            except ValueError:
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": liveness_score,
                    "reason": "Unable to compute embedding",
                }

        best_similarity = 0.0
        best_user_id = None
        best_user_name = None
        company_index = self._get_company_index(company_id)
        for entry in company_index:
            similarity = max(self.engine.cosine_similarity(query_emb, emb) for emb in entry["embeddings"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_user_id = entry["user_id"]
                best_user_name = entry["name"]
                if best_similarity >= max(0.90, self.engine.settings.recognition_threshold + 0.25):
                    break

        verified = best_user_id is not None and best_similarity >= self.engine.settings.recognition_threshold
        reason = "Verified" if verified else "Face mismatch"

        self.log_service.write(
            user_id=best_user_id,
            company_id=company_id,
            confidence=float(best_similarity),
            liveness_score=float(liveness_score),
            success=bool(verified),
            device_id=device_id,
        )

        return {
            "verified": bool(verified),
            "user_id": best_user_id,
            "name": best_user_name,
            "confidence": float(best_similarity),
            "liveness_score": float(liveness_score),
            "reason": reason,
        }
