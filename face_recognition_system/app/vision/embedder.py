import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self) -> None:
        self.model = FaceAnalysis(name="buffalo_l")
        # CPU-safe default. Switch to GPU by setting ctx_id >= 0 in production if available.
        self.model.prepare(ctx_id=-1, det_size=(640, 640))

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr.size == 0:
            raise ValueError("Empty face crop")

        for candidate in self._candidate_images(face_bgr):
            rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
            faces = self.model.get(rgb)
            if not faces:
                continue

            best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
            emb = best_face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            return emb / norm

        raise ValueError("Unable to compute embedding from provided face crop")

    def get_best_embedding(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Returns normalized embedding for the highest-confidence face and number of faces found.
        Uses a single model pass, which is faster for live identify flows.
        """
        if image_bgr.size == 0:
            raise ValueError("Empty image")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = self.model.get(rgb)
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
