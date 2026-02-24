import numpy as np

from app.core.config import get_settings
from app.vision.detector import FaceDetector
from app.vision.embedder import FaceEmbedder
from app.vision.liveness import LivenessDetector


class RecognitionEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = FaceEmbedder()
        shared_det_model = getattr(self.embedder.model, "det_model", None)
        self.detector = FaceDetector(onnx_det_model=shared_det_model)
        self.liveness = LivenessDetector()

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / denom)

    def verify(
        self,
        image_bgr,
        enrolled_embeddings: list[np.ndarray],
        expected_challenge: str | None,
        challenge_response: str | None,
        previous_image_bgr=None,
    ) -> tuple[bool, float, float, str]:
        detections = self.detector.detect(image_bgr)
        if not detections:
            return False, 0.0, 0.0, "No face detected"

        prev_face = None
        if previous_image_bgr is not None:
            prev_dets = self.detector.detect(previous_image_bgr)
            if prev_dets:
                prev = max(prev_dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
                prev_face = self.detector.crop_face(previous_image_bgr, prev["bbox"])

        best_similarity = 0.0
        best_liveness = 0.0
        had_embedding_failure = False

        for det in detections:
            face = self.detector.crop_face(image_bgr, det["bbox"])
            liveness_score = self.liveness.score(
                face,
                det.get("landmarks", {}),
                expected_challenge=expected_challenge,
                challenge_response=challenge_response,
                previous_face_bgr=prev_face,
            )
            if liveness_score < self.settings.liveness_threshold:
                continue

            try:
                query_emb = self.embedder.get_embedding(face)
            except ValueError:
                had_embedding_failure = True
                continue

            similarity = max(self.cosine_similarity(query_emb, enrolled) for enrolled in enrolled_embeddings)

            if similarity > best_similarity:
                best_similarity = similarity
                best_liveness = liveness_score

        if best_similarity == 0.0:
            if had_embedding_failure:
                return False, 0.0, 0.0, "Unable to compute embedding"
            return False, 0.0, 0.0, "Liveness failed"

        verified = best_similarity >= self.settings.recognition_threshold
        if not verified:
            return False, best_similarity, best_liveness, "Face mismatch"
        return True, best_similarity, best_liveness, "Verified"
