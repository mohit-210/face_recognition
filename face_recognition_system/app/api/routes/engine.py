import base64

import cv2
import numpy as np
from fastapi import APIRouter, Header, HTTPException
from functools import lru_cache
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.vision.embedder import FaceEmbedder

router = APIRouter(prefix="/engine", tags=["Face Engine"])


@lru_cache(maxsize=1)
def get_engine_embedder() -> FaceEmbedder:
    return FaceEmbedder()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


class EmbedRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)


class CompareRequest(BaseModel):
    embedding_a: list[float]
    embedding_b: list[float]


def _require_engine_key(x_face_engine_key: str | None) -> None:
    settings = get_settings()
    if settings.face_engine_key and x_face_engine_key != settings.face_engine_key:
        raise HTTPException(status_code=401, detail="Invalid face engine key")


def _decode_image(image_base64: str) -> np.ndarray:
    settings = get_settings()
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    if len(image_base64) > int(settings.engine_max_image_bytes * 1.4):
        raise HTTPException(status_code=413, detail="Image payload too large")
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc
    if len(raw) > settings.engine_max_image_bytes:
        raise HTTPException(status_code=413, detail="Image payload too large")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    return image


def _quality_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    blur_score = min(1.0, blur / 120.0)
    light_score = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0)
    return round(max(0.0, (blur_score * 0.65) + (light_score * 0.35)), 4)


@router.get("/health")
def engine_health(x_face_engine_key: str | None = Header(default=None)):
    _require_engine_key(x_face_engine_key)
    embedder = get_engine_embedder()
    return {
        "status": "ok",
        "model": embedder.model_name,
        "providers": embedder.providers or ["default"],
    }


@router.post("/embed")
def embed(payload: EmbedRequest, x_face_engine_key: str | None = Header(default=None)):
    _require_engine_key(x_face_engine_key)
    image = _decode_image(payload.image_base64)
    embedder = get_engine_embedder()
    try:
        embedding, face_count = embedder.get_best_embedding(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "face_detected": True,
        "face_count": int(face_count),
        "quality_score": _quality_score(image),
        "embedding": [float(v) for v in embedding.tolist()],
        "model": embedder.model_name,
    }


@router.post("/compare")
def compare(payload: CompareRequest, x_face_engine_key: str | None = Header(default=None)):
    _require_engine_key(x_face_engine_key)
    if not payload.embedding_a or not payload.embedding_b:
        raise HTTPException(status_code=422, detail="Both embeddings are required")
    vec_a = np.asarray(payload.embedding_a, dtype=np.float32)
    vec_b = np.asarray(payload.embedding_b, dtype=np.float32)
    score = cosine_similarity(vec_a, vec_b)
    return {"score": float(score)}
