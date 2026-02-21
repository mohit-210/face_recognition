from functools import lru_cache
import logging
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

logger = logging.getLogger(__name__)

COMPANY_INDEX_TTL_SECONDS = 300.0
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
    def _clamp_bbox(bbox: list[int], img_w: int, img_h: int) -> list[int]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return [x1, y1, x2, y2]

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

    def _embed_face_with_retries(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
        max_candidates: int = 6,
    ) -> tuple[np.ndarray | None, str | None]:
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

        for candidate in candidates[: max(1, int(max_candidates))]:
            try:
                emb = self.engine.embedder.get_embedding(candidate)
                return emb, None
            except ValueError:
                continue

        base_face = self.detector.crop_face(image_bgr, bbox, pad_ratio=0.22)
        return None, self._embedding_failure_hint(base_face)

    @staticmethod
    def _recommended_embed_candidates(face_bgr: np.ndarray, fast_mode: bool) -> int:
        if fast_mode:
            return 2
        if face_bgr.size == 0:
            return 4
        h, w = face_bgr.shape[:2]
        min_side = min(h, w)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if min_side >= 140 and blur_var >= 45.0:
            return 3
        return 5

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
        previous_bbox: list[int] | None = None,
        fast_mode: bool = True,
        debug_timing: bool = False,
        strict_attendance: bool = False,
        allow_bbox_reuse: bool = True,
    ) -> dict:
        started = time.perf_counter()
        timings = {
            "decode_ms": 0.0,
            "prev_decode_ms": 0.0,
            "detect_ms": 0.0,
            "liveness_ms": 0.0,
            "embed_ms": 0.0,
            "index_ms": 0.0,
            "match_ms": 0.0,
        }

        def _finish(payload: dict) -> dict:
            timings["total_ms"] = (time.perf_counter() - started) * 1000.0
            if debug_timing:
                payload["debug_timings"] = {k: round(v, 2) for k, v in timings.items()}
            return payload

        model_used = "none"
        liveness_threshold_used: float | None = None
        device_tag = (device_id or "").strip().lower()
        mobile_mode = ("flutter" in device_tag) or ("mobile" in device_tag)

        def _fail(reason: str, liveness: float = 0.0, bbox: list[int] | None = None) -> dict:
            return _finish(
                {
                    "verified": False,
                    "user_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "liveness_score": float(liveness),
                    "liveness_threshold_used": liveness_threshold_used,
                    "model_used": model_used,
                    "reason": reason,
                    "_bbox": bbox,
                }
            )

        t0 = time.perf_counter()
        image = self.detector.decode_image(image_base64)
        image = self._resize_for_identify(image, max_side=640)
        timings["decode_ms"] += (time.perf_counter() - t0) * 1000.0

        previous_image = None
        if previous_image_base64:
            try:
                t0 = time.perf_counter()
                previous_image = self.detector.decode_image(previous_image_base64)
                previous_image = self._resize_for_identify(previous_image, max_side=640)
                timings["prev_decode_ms"] += (time.perf_counter() - t0) * 1000.0
            except ValueError:
                previous_image = None

        liveness_score = 1.0
        motion_score = 0.0
        det = None
        query_emb = None

        if not enforce_liveness:
            model_used = "liveness_disabled"
            t0 = time.perf_counter()
            detections = self.detector.detect(image)
            timings["detect_ms"] += (time.perf_counter() - t0) * 1000.0
            if not detections:
                return _fail("No face detected")
            det = max(
                detections,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            )
            face = self.detector.crop_face(image, det["bbox"])
            candidates = self._recommended_embed_candidates(face, fast_mode=fast_mode)
            t0 = time.perf_counter()
            query_emb, hint = self._embed_face_with_retries(image, det["bbox"], max_candidates=candidates)
            timings["embed_ms"] += (time.perf_counter() - t0) * 1000.0
            if query_emb is None:
                return _fail(hint or "Unable to compute embedding", bbox=det["bbox"])
        else:
            detections: list[dict] = []
            # Reuse recent bbox when available to avoid expensive detector calls on every frame.
            if allow_bbox_reuse and previous_bbox and len(previous_bbox) == 4:
                h, w = image.shape[:2]
                candidate_bbox = self._clamp_bbox(previous_bbox, w, h)
                reused_face = self.detector.crop_face(image, candidate_bbox, pad_ratio=0.0)
                if reused_face.size > 0:
                    detections = [{"bbox": candidate_bbox, "landmarks": {}, "score": 0.99}]

            if not detections:
                t0 = time.perf_counter()
                detections = self.detector.detect(image)
                timings["detect_ms"] += (time.perf_counter() - t0) * 1000.0

            if not detections:
                return _fail("No face detected")
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
                    return _fail("Clear the Frame: multiple faces detected")
                detections = [ranked[0]]

            det = detections[0]
            face = self.detector.crop_face(image, det["bbox"])
            previous_face = None
            if previous_image is not None:
                previous_face = self.detector.crop_face(previous_image, det["bbox"])

            if previous_face is not None and previous_face.size > 0 and face.size > 0:
                if previous_face.shape[:2] != face.shape[:2]:
                    previous_face = self.detector.crop_face(previous_image, det["bbox"], pad_ratio=0.0)
                    if previous_face.shape[:2] != face.shape[:2]:
                        previous_face = None

            t0 = time.perf_counter()
            passive = self.passive_antispoof.score(face)
            if passive.is_environment_bad:
                timings["liveness_ms"] += (time.perf_counter() - t0) * 1000.0
                return _fail(str(passive.environmental_error), bbox=det["bbox"])
            using_fallback_liveness = passive.reason.startswith("Passive anti-spoof model unavailable")
            model_used = (
                "heuristic_liveness_fallback"
                if using_fallback_liveness
                else f"{self.passive_antispoof.model_descriptor} + heuristic_liveness"
            )
            if using_fallback_liveness:
                heuristic_liveness = float(
                    self.engine.liveness.score(
                        face,
                        det.get("landmarks", {}),
                        previous_face_bgr=previous_face,
                    )
                )
                liveness_score = heuristic_liveness
                # Keep fallback calibration conservative; avoid inflating live confidence.
                liveness_score = min(1.0, liveness_score + 0.02)
            else:
                heuristic_liveness = float(
                    self.engine.liveness.score(
                        face,
                        det.get("landmarks", {}),
                        previous_face_bgr=previous_face,
                    )
                )
                if mobile_mode:
                    # Favor CNN output on mobile captures where heuristic can be noisy.
                    liveness_score = float((0.88 * float(passive.confidence)) + (0.12 * heuristic_liveness))
                else:
                    liveness_score = float((0.80 * float(passive.confidence)) + (0.20 * heuristic_liveness))

            motion_score = float(self.engine.liveness._motion_score(face, previous_face))

            if (
                not using_fallback_liveness
                and previous_face is not None
                and previous_face.size > 0
                and face.size > 0
                and self._is_rigid_planar_replay(previous_face, face)
            ):
                replay_penalty = 0.12 if mobile_mode else 0.20
                liveness_score = max(0.0, liveness_score - replay_penalty)
            timings["liveness_ms"] += (time.perf_counter() - t0) * 1000.0

            # Use attendance-specific threshold for attendance flows instead of enforcing
            # the stricter global threshold, which causes unnecessary false rejects.
            required_liveness = (
                self.engine.settings.attendance_liveness_threshold
                if require_live_motion
                else self.engine.settings.liveness_threshold
            )
            if using_fallback_liveness:
                required_liveness = max(required_liveness, self.engine.settings.fallback_liveness_threshold)
            liveness_threshold_used = float(required_liveness)
            if liveness_score < required_liveness:
                return _fail(
                    f"Liveness failed (score={liveness_score:.2f}, threshold={required_liveness:.2f})",
                    liveness=liveness_score,
                    bbox=det["bbox"],
                )
            if require_live_motion and motion_score < self.engine.settings.attendance_min_motion_score:
                return _fail(
                    (
                        "Insufficient live motion "
                        f"(score={motion_score:.2f}, min={self.engine.settings.attendance_min_motion_score:.2f})"
                    ),
                    liveness=liveness_score,
                    bbox=det["bbox"],
                )

            candidates = self._recommended_embed_candidates(face, fast_mode=fast_mode)
            t0 = time.perf_counter()
            query_emb, hint = self._embed_face_with_retries(image, det["bbox"], max_candidates=candidates)
            timings["embed_ms"] += (time.perf_counter() - t0) * 1000.0
            if query_emb is None:
                return _fail(hint or "Unable to compute embedding", liveness=liveness_score, bbox=det["bbox"])

        best_similarity = 0.0
        best_user_id = None
        best_user_name = None

        t0 = time.perf_counter()
        company_index = self._get_company_index(company_id)
        timings["index_ms"] += (time.perf_counter() - t0) * 1000.0

        matrix = company_index.get("embedding_matrix")
        user_ids = company_index.get("embedding_user_ids")
        name_by_user = company_index.get("name_by_user", {})
        t0 = time.perf_counter()
        if isinstance(matrix, np.ndarray) and isinstance(user_ids, np.ndarray) and matrix.size and user_ids.size:
            scores = matrix.dot(np.asarray(query_emb, dtype=np.float32))
            best_idx = int(np.argmax(scores))
            best_similarity = float(scores[best_idx])
            best_user_id = int(user_ids[best_idx])
            best_user_name = name_by_user.get(best_user_id)
        timings["match_ms"] += (time.perf_counter() - t0) * 1000.0

        required_recognition = self.engine.settings.recognition_threshold
        if strict_attendance:
            required_recognition = max(required_recognition, self.engine.settings.attendance_recognition_threshold)
        verified = best_user_id is not None and best_similarity >= required_recognition
        reason = "Verified" if verified else "Face mismatch"

        self.log_service.write(
            user_id=best_user_id,
            company_id=company_id,
            confidence=float(best_similarity),
            liveness_score=float(liveness_score),
            success=bool(verified),
            device_id=device_id,
        )

        payload = {
            "verified": bool(verified),
            "user_id": best_user_id,
            "name": best_user_name,
            "confidence": float(best_similarity),
            "liveness_score": float(liveness_score),
            "liveness_threshold_used": liveness_threshold_used,
            "model_used": model_used,
            "reason": reason,
            "_bbox": det["bbox"] if det else None,
        }
        finished = _finish(payload)
        if debug_timing:
            logger.info(
                "attendance-identify company=%s verified=%s total=%.2fms detect=%.2fms liveness=%.2fms embed=%.2fms",
                company_id,
                bool(verified),
                finished["debug_timings"].get("total_ms", 0.0),
                finished["debug_timings"].get("detect_ms", 0.0),
                finished["debug_timings"].get("liveness_ms", 0.0),
                finished["debug_timings"].get("embed_ms", 0.0),
            )
        return finished
