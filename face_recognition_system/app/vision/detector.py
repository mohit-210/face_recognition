import base64

import cv2
import numpy as np
from retinaface import RetinaFace

from app.core.config import get_settings


class FaceDetector:
    def __init__(self) -> None:
        self.settings = get_settings()

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        result = RetinaFace.detect_faces(image_bgr)
        if not isinstance(result, dict):
            return []

        detections = []
        for face in result.values():
            x1, y1, x2, y2 = face["facial_area"]
            score = float(face.get("score", 0.0))
            if score < float(self.settings.retinaface_min_score):
                continue
            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "landmarks": face.get("landmarks", {}),
                    "score": score,
                }
            )
        detections.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        return detections

    @staticmethod
    def decode_image(image_base64: str) -> np.ndarray:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        data = base64.b64decode(image_base64)
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image payload")
        return image

    @staticmethod
    def crop_face(image: np.ndarray, bbox: list[int], pad_ratio: float = 0.22) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * max(0.0, pad_ratio))
        pad_y = int(bh * max(0.0, pad_ratio))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return image[y1:y2, x1:x2]
