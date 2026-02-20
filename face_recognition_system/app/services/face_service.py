from functools import lru_cache
import time
import numpy as np
import cv2

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories.face_repository import FaceProfileRepository
from app.services.log_service import LogService
from app.services.user_service import UserService
from app.vision.detector import FaceDetector
from app.vision.passive_antispoof import PassiveAntiSpoofDetector
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


@lru_cache(maxsize=1)
def get_passive_antispoof_detector() -> PassiveAntiSpoofDetector:
    return PassiveAntiSpoofDetector()


class FaceService:
    def __init__(self, db: Session) -> None:
        self.user_service = UserService(db)
        self.face_repo = FaceProfileRepository(db)
        self.log_service = LogService(db)
        self.engine = get_recognition_engine()
        self.passive_antispoof = get_passive_antispoof_detector()
        self.detector = FaceDetector()

    @staticmethod
    def _invalidate_company_index(company_id: int) -> None:
        _company_index_cache.pop(company_id, None)

    @staticmethod
    def _resize_for_identify(image: np.ndarray, max_side: int = 640) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        return cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _embedding_failure_hint(face_bgr: np.ndarray) -> str:
        if face_bgr.size == 0:
            return "Face too small/blurred, move closer and hold still."
        h, w = face_bgr.shape[:2]
        min_side = min(h, w)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        if min_side < 100:
            return "Face too small, move closer to the camera."
        if blur_var < 20.0:
            return "Face too blurred, hold still and keep camera steady."
        if brightness < 45.0:
            return "Lighting too low, move to brighter light."
        if brightness > 220.0:
            return "Lighting too harsh, reduce glare and try again."
        return "Face too small/blurred, move closer and hold still."

    def _embed_face_with_retries(self, image_bgr: np.ndarray, bbox: list[int]) -> tuple[np.ndarray | None, str | None]:
        # Try a few crop paddings + scales to recover embeddings from borderline frames.
        candidates: list[np.ndarray] = []
        for pad in (0.10, 0.22, 0.34):
            crop = self.detector.crop_face(image_bgr, bbox, pad_ratio=pad)
            if crop.size > 0:
                candidates.append(crop)
                min_side = min(crop.shape[:2])
                if min_side < 220:
                    scale = 220.0 / float(max(1, min_side))
                    upscaled = cv2.resize(
                        crop,
                        (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    candidates.append(upscaled)

        for candidate in candidates[:6]:
            try:
                emb = self.engine.embedder.get_embedding(candidate)
                return emb, None
            except ValueError:
                continue

        base_face = self.detector.crop_face(image_bgr, bbox, pad_ratio=0.22)
        return None, self._embedding_failure_hint(base_face)

    def _get_company_index(self, company_id: int) -> dict:
        now = time.monotonic()
        cached = _company_index_cache.get(company_id)
        if cached and (now - float(cached.get("ts", 0.0))) < COMPANY_INDEX_TTL_SECONDS:
            return cached

        users = [u for u in self.user_service.list_by_company(company_id) if u.status == "active"]
        user_ids = [u.id for u in users]
        embeddings_by_user = self.face_repo.get_embeddings_by_user_ids(user_ids)

        entries = []
        flat_embeddings: list[np.ndarray] = []
        flat_user_ids: list[int] = []
        name_by_user: dict[int, str] = {}
        for user in users:
            embeddings = embeddings_by_user.get(user.id, [])
            if not embeddings:
                continue
            entries.append({"user_id": user.id, "name": user.name, "embeddings": embeddings})
            name_by_user[user.id] = user.name
            for emb in embeddings:
                flat_embeddings.append(np.asarray(emb, dtype=np.float32))
                flat_user_ids.append(int(user.id))

        if flat_embeddings:
            embedding_matrix = np.vstack(flat_embeddings)
            embedding_user_ids = np.asarray(flat_user_ids, dtype=np.int32)
        else:
            embedding_matrix = np.empty((0, 0), dtype=np.float32)
            embedding_user_ids = np.empty((0,), dtype=np.int32)

        payload = {
            "ts": now,
            "entries": entries,
            "embedding_matrix": embedding_matrix,
            "embedding_user_ids": embedding_user_ids,
            "name_by_user": name_by_user,
        }
        _company_index_cache[company_id] = payload
        return payload

    @staticmethod
    def _is_rigid_planar_replay(previous_face: np.ndarray, face: np.ndarray) -> bool:
        if previous_face.size == 0 or face.size == 0:
            return False
        prev_gray = cv2.cvtColor(cv2.resize(previous_face, (128, 128)), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(cv2.resize(face, (128, 128)), cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.01, minDistance=4)
        if points is None or len(points) < 12:
            return False

        tracked, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)
        if tracked is None or status is None:
            return False

        valid_mask = status.reshape(-1) == 1
        if int(np.sum(valid_mask)) < 12:
            return False
        src = points.reshape(-1, 2)[valid_mask].astype(np.float32)
        dst = tracked.reshape(-1, 2)[valid_mask].astype(np.float32)

        affine, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1500,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None:
            return False

        if inliers is not None:
            inlier_mask = inliers.reshape(-1) == 1
            if int(np.sum(inlier_mask)) >= 8:
                src = src[inlier_mask]
                dst = dst[inlier_mask]

        projected = cv2.transform(src.reshape(1, -1, 2), affine).reshape(-1, 2)
        residual = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
        motion_mag = float(np.mean(np.linalg.norm(dst - src, axis=1)))
        det = float(np.linalg.det(affine[:, :2]))

        # Flat printed/replayed faces moved in front of camera often fit an affine transform too perfectly.
        return bool(residual < 0.35 and motion_mag > 1.0 and abs(det - 1.0) < 0.08)

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
        previous_image_base64: str | None = None,
        require_live_motion: bool = False,
    ) -> dict:
        image = self.detector.decode_image(image_base64)
        image = self._resize_for_identify(image, max_side=640)
        liveness_score = 1.0
        if not enforce_liveness:
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
            det = max(
                detections,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            )
            query_emb, hint = self._embed_face_with_retries(image, det["bbox"])
            if query_emb is None:
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": 0.0,
                    "reason": hint or "Unable to compute embedding",
                }
        else:
            previous_image = None
            if previous_image_base64:
                try:
                    previous_image = self.detector.decode_image(previous_image_base64)
                    previous_image = self._resize_for_identify(previous_image, max_side=640)
                except ValueError:
                    previous_image = None

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
                        "reason": "Clear the Frame: multiple faces detected",
                    }
                detections = [ranked[0]]

            det = detections[0]
            face = self.detector.crop_face(image, det["bbox"])
            previous_face = None
            if previous_image is not None:
                # Reuse current-frame bbox for previous frame to avoid another heavy detector call.
                # Both frames come from the same camera stream and are close in time.
                previous_face = self.detector.crop_face(previous_image, det["bbox"])

            if previous_face is not None and previous_face.size > 0 and face.size > 0:
                if previous_face.shape[:2] != face.shape[:2]:
                    previous_face = self.detector.crop_face(previous_image, det["bbox"], pad_ratio=0.0)
                    if previous_face.shape[:2] != face.shape[:2]:
                        previous_face = None

            passive = self.passive_antispoof.score(face)
            if passive.is_environment_bad:
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": 0.0,
                    "reason": str(passive.environmental_error),
                }
            using_fallback_liveness = passive.reason.startswith("Passive anti-spoof model unavailable")
            if using_fallback_liveness:
                liveness_score = float(
                    self.engine.liveness.score(
                        face,
                        det.get("landmarks", {}),
                        previous_face_bgr=previous_face,
                    )
                )
                # Fallback heuristic is conservative; add a small calibration lift to reduce false rejects.
                liveness_score = min(1.0, liveness_score + 0.08)
            else:
                liveness_score = float(passive.confidence)

            # Keep motion as a secondary anti-replay signal only when passive model is active.
            if (
                not using_fallback_liveness
                and previous_face is not None
                and previous_face.size > 0
                and face.size > 0
            ):
                if self._is_rigid_planar_replay(previous_face, face):
                    liveness_score = max(0.0, liveness_score - 0.20)

            required_liveness = self.engine.settings.liveness_threshold
            if require_live_motion:
                required_liveness = min(required_liveness, self.engine.settings.attendance_liveness_threshold)
            elif using_fallback_liveness:
                required_liveness = min(required_liveness, 0.35)
            if liveness_score < required_liveness:
                reason = "Liveness failed"
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": liveness_score,
                    "reason": f"{reason} (score={liveness_score:.2f}, threshold={required_liveness:.2f})",
                }

            query_emb, hint = self._embed_face_with_retries(image, det["bbox"])
            if query_emb is None:
                return {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": liveness_score,
                    "reason": hint or "Unable to compute embedding",
                }

        best_similarity = 0.0
        best_user_id = None
        best_user_name = None
        company_index = self._get_company_index(company_id)
        matrix = company_index.get("embedding_matrix")
        user_ids = company_index.get("embedding_user_ids")
        name_by_user = company_index.get("name_by_user", {})
        if isinstance(matrix, np.ndarray) and isinstance(user_ids, np.ndarray) and matrix.size and user_ids.size:
            scores = matrix.dot(np.asarray(query_emb, dtype=np.float32))
            best_idx = int(np.argmax(scores))
            best_similarity = float(scores[best_idx])
            best_user_id = int(user_ids[best_idx])
            best_user_name = name_by_user.get(best_user_id)

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
