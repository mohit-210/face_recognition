import base64
import logging
from functools import lru_cache

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from retinaface import RetinaFace

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _get_cached_onnx_detector_model(providers_key: str, ctx_id: int, model_name: str, det_size: int):
    providers = [p.strip() for p in providers_key.split(",") if p.strip()]
    kwargs = {"name": model_name}
    if providers:
        kwargs["providers"] = providers
    app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return getattr(app, "det_model", None)


class FaceDetector:
    def __init__(self, onnx_det_model=None) -> None:
        self.settings = get_settings()
        self.onnx_det_model = onnx_det_model
        self._onnx_disabled = False
        self._dml_detector_failed = False
        if self.onnx_det_model is None:
            providers_key = str(getattr(self.settings, "arcface_onnx_providers", "") or "")
            model_name = str(getattr(self.settings, "arcface_model_name", "buffalo_s") or "buffalo_s").strip()
            det_size = int(getattr(self.settings, "arcface_det_size", 320))
            ctx_id = int(getattr(self.settings, "arcface_ctx_id", -1))
            try:
                self.onnx_det_model = _get_cached_onnx_detector_model(providers_key, ctx_id, model_name, det_size)
            except Exception:
                self.onnx_det_model = None
                logger.exception("Failed to initialize ONNX detector model; falling back to retinaface CPU detector.")

    @staticmethod
    def _landmarks_from_kps(kps) -> dict:
        if kps is None:
            return {}
        arr = np.asarray(kps, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < 2:
            return {}
        return {
            "left_eye": [float(arr[0, 0]), float(arr[0, 1])],
            "right_eye": [float(arr[1, 0]), float(arr[1, 1])],
            "nose": [float(arr[2, 0]), float(arr[2, 1])],
            "mouth_left": [float(arr[3, 0]), float(arr[3, 1])],
            "mouth_right": [float(arr[4, 0]), float(arr[4, 1])],
        }

    def _detect_with_onnx(self, image_bgr: np.ndarray) -> list[dict]:
        if self._onnx_disabled or self.onnx_det_model is None:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bboxes, kpss = self.onnx_det_model.detect(rgb, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []
        detections = []
        for i in range(int(bboxes.shape[0])):
            bbox = bboxes[i, 0:4]
            score = float(bboxes[i, 4]) if bboxes.shape[1] >= 5 else 0.0
            if score < float(self.settings.retinaface_min_score):
                continue
            landmarks = self._landmarks_from_kps(kpss[i] if kpss is not None and len(kpss) > i else None)
            detections.append(
                {
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "landmarks": landmarks,
                    "score": score,
                }
            )
        detections.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        return detections

    def _detect_with_retinaface(self, image_bgr: np.ndarray) -> list[dict]:
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

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        try:
            onnx_detections = self._detect_with_onnx(image_bgr)
            if onnx_detections:
                return onnx_detections
        except Exception as exc:
            text = str(exc)
            is_dml_runtime = "DmlExecutionProvider" in text and "ONNXRuntimeError" in text
            if is_dml_runtime and not self._dml_detector_failed:
                self._dml_detector_failed = True
                logger.warning("ONNX detector DML runtime error; switching detector to CPU ONNX. error=%s", exc)
                try:
                    model_name = str(getattr(self.settings, "arcface_model_name", "buffalo_s") or "buffalo_s").strip()
                    det_size = int(getattr(self.settings, "arcface_det_size", 320))
                    self.onnx_det_model = _get_cached_onnx_detector_model(
                        "CPUExecutionProvider",
                        -1,
                        model_name,
                        det_size,
                    )
                    onnx_detections = self._detect_with_onnx(image_bgr)
                    if onnx_detections:
                        return onnx_detections
                except Exception:
                    logger.exception("Failed to switch ONNX detector to CPU provider.")
            else:
                logger.exception("ONNX detector failed; falling back to retinaface detector.")

        return self._detect_with_retinaface(image_bgr)

    @staticmethod
    def decode_image(image_base64: str) -> np.ndarray:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        data = base64.b64decode(image_base64)
        return FaceDetector.decode_image_bytes(data)

    @staticmethod
    def decode_image_bytes(data: bytes) -> np.ndarray:
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
