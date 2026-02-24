import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class FaceEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.arcface_model_name.strip() or "buffalo_l"
        self.det_size = int(settings.arcface_det_size)
        self.ctx_id = int(getattr(settings, "arcface_ctx_id", -1))
        self.providers = [
            p.strip() for p in str(getattr(settings, "arcface_onnx_providers", "")).split(",") if p.strip()
        ]
        self._using_cpu_fallback = False
        self.model = self._build_model(self.providers, self.ctx_id)

    def _build_model(self, providers: list[str], ctx_id: int) -> FaceAnalysis:
        kwargs = {"name": self.model_name}
        if providers:
            kwargs["providers"] = providers
        model = FaceAnalysis(**kwargs)
        model.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        return model

    def _is_dml_runtime_error(self, exc: Exception) -> bool:
        text = str(exc)
        return (
            "ONNXRuntimeError" in text
            and "DmlExecutionProvider" in text
            and ("Reshape" in text or "RUNTIME_EXCEPTION" in text)
        )

    def _get_faces(self, rgb: np.ndarray):
        try:
            return self.model.get(rgb)
        except Exception as exc:
            if self._is_dml_runtime_error(exc) and not self._using_cpu_fallback:
                logger.warning(
                    "ArcFace DirectML runtime error detected; switching embedder to CPU fallback. error=%s",
                    exc,
                )
                self.model = self._build_model(["CPUExecutionProvider"], ctx_id=-1)
                self._using_cpu_fallback = True
                return self.model.get(rgb)
            raise

    def _switch_to_cpu_fallback(self) -> None:
        if self._using_cpu_fallback:
            return
        self.model = self._build_model(["CPUExecutionProvider"], ctx_id=-1)
        self._using_cpu_fallback = True

    @staticmethod
    def _landmarks_to_kps(landmarks: dict | None) -> np.ndarray | None:
        if not isinstance(landmarks, dict):
            return None
        keys = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
        pts: list[list[float]] = []
        for key in keys:
            value = landmarks.get(key)
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return None
            pts.append([float(value[0]), float(value[1])])
        return np.asarray(pts, dtype=np.float32)

    def preflight_runtime(self) -> None:
        """
        Warm only the recognition branch at startup.
        Avoid detector invocation here because DirectML can fail in RetinaFace path
        even when recognition itself is stable.
        """
        recognition = getattr(self.model, "models", {}).get("recognition")
        if recognition is None:
            return
        probe = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            _ = recognition.get_feat(probe)
        except Exception as exc:
            if self._is_dml_runtime_error(exc):
                logger.warning("ArcFace recognition preflight DirectML error, switching to CPU fallback: %s", exc)
                self._switch_to_cpu_fallback()
            else:
                logger.warning("ArcFace preflight inference failed: %s", exc)

    def get_embedding_from_detection(self, image_bgr: np.ndarray, det: dict) -> np.ndarray:
        """
        Fast path: reuse detector landmarks and run recognition directly.
        Avoids a second detector pass inside FaceAnalysis.get(...).
        """
        if image_bgr.size == 0:
            raise ValueError("Empty image")

        kps = self._landmarks_to_kps(det.get("landmarks") if isinstance(det, dict) else None)
        if kps is None:
            raise ValueError("Missing landmarks")

        recognition = getattr(self.model, "models", {}).get("recognition")
        if recognition is None:
            raise ValueError("Recognition model unavailable")

        try:
            aligned = face_align.norm_crop(image_bgr, landmark=kps, image_size=recognition.input_size[0])
            emb = recognition.get_feat(aligned).flatten().astype(np.float32)
        except Exception as exc:
            if self._is_dml_runtime_error(exc):
                logger.warning(
                    "ArcFace DirectML runtime error on recognition path; switching to CPU fallback. error=%s",
                    exc,
                )
                self._switch_to_cpu_fallback()
                recognition = getattr(self.model, "models", {}).get("recognition")
                if recognition is None:
                    raise ValueError("Recognition model unavailable after fallback") from exc
                aligned = face_align.norm_crop(image_bgr, landmark=kps, image_size=recognition.input_size[0])
                emb = recognition.get_feat(aligned).flatten().astype(np.float32)
            else:
                raise

        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Invalid embedding")
        return emb / norm

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr.size == 0:
            raise ValueError("Empty face crop")

        for candidate in self._candidate_images(face_bgr):
            rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
            faces = self._get_faces(rgb)
            if not faces:
                continue

            best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
            emb = best_face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            return emb / norm

        raise ValueError("Unable to compute embedding from provided face crop")

    def get_embedding_fast(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Single-pass embedding for low-latency paths.
        Skips padded/upscaled retries to reduce CPU load.
        """
        if face_bgr.size == 0:
            raise ValueError("Empty face crop")

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        faces = self._get_faces(rgb)
        if not faces:
            raise ValueError("Unable to compute embedding from provided face crop")

        best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = best_face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Invalid embedding")
        return emb / norm

    def get_best_embedding(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Returns normalized embedding for the highest-confidence face and number of faces found.
        Uses a single model pass, which is faster for live identify flows.
        """
        if image_bgr.size == 0:
            raise ValueError("Empty image")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = self._get_faces(rgb)
        if not faces:
            raise ValueError("No face detected")

        best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = best_face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Invalid embedding")
        return emb / norm, len(faces)

    def _candidate_images(self, face_bgr: np.ndarray) -> list[np.ndarray]:
        # Tight face crops can fail detector-based embedding. Try padded/upscaled variants.
        candidates: list[np.ndarray] = [face_bgr]
        h, w = face_bgr.shape[:2]

        pad_y = max(8, int(h * 0.2))
        pad_x = max(8, int(w * 0.2))
        padded = cv2.copyMakeBorder(face_bgr, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101)
        candidates.append(padded)

        min_target = 224
        for image in (face_bgr, padded):
            ih, iw = image.shape[:2]
            min_side = min(ih, iw)
            if min_side < min_target:
                scale = min_target / float(min_side)
                resized = cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_CUBIC)
                candidates.append(resized)

        return candidates

