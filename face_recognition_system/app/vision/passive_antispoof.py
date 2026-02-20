from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings

try:
    from tensorflow import keras
except Exception:  # pragma: no cover - runtime dependency may be absent in some dev setups
    keras = None


@dataclass
class PassiveLivenessResult:
    confidence: float
    environmental_error: str | None = None
    reason: str = "ok"

    @property
    def is_environment_bad(self) -> bool:
        return self.environmental_error is not None


def _build_minifasnet_lite(input_shape: tuple[int, int, int] = (128, 128, 3)):
    if keras is None:
        raise RuntimeError("TensorFlow/Keras is not available")

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(24, 1, padding="same", activation="relu")(x)
    x = keras.layers.DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(40, 1, padding="same", activation="relu")(x)
    x = keras.layers.DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(64, 1, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.15)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="live_prob")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mini_fasnet_lite")
    return model


class PassiveAntiSpoofDetector:
    """
    Passive liveness detector:
    1) environmental pre-check (blur + lighting)
    2) lightweight CNN inference on frequency/texture-aware channels
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = None
        self.model_ready = False
        self.calibration: dict = {}
        model_path = Path(self.settings.passive_antispoof_model_path)
        calibration_path = Path(self.settings.passive_antispoof_calibration_path)

        if calibration_path.exists():
            try:
                self.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            except Exception:
                self.calibration = {}

        if model_path.exists() and keras is not None:
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                self.model_ready = True
            except Exception:
                self.model = None

        # Safety gate: keep passive model disabled when calibration indicates poor quality.
        if self.model_ready and self.calibration:
            eer = float(self.calibration.get("eer", 1.0))
            val_auc_best = float(self.calibration.get("val_auc_best", 0.0))
            if eer > 0.35 or val_auc_best < 0.80:
                self.model_ready = False

    @staticmethod
    def preprocess(face_bgr: np.ndarray, out_size: int = 128) -> np.ndarray:
        resized = cv2.resize(face_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap = np.clip(np.abs(lap), 0.0, 1.0)

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        magnitude = magnitude / (float(np.max(magnitude)) + 1e-6)

        stacked = np.stack([gray, lap, magnitude.astype(np.float32)], axis=-1).astype(np.float32)
        return stacked

    def _environment_check(self, face_bgr: np.ndarray) -> tuple[str | None, float]:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        soft_penalty = 0.0

        blur_min = float(self.settings.passive_blur_min_variance)
        hard_floor = max(10.0, blur_min * 0.70)
        if blur_var < hard_floor:
            return f"Environmental Error: image too blurry ({blur_var:.1f})", 0.0
        if blur_var < blur_min:
            # Moderate blur should not hard-fail; keep scan usable with reduced confidence.
            soft_penalty += 0.12
        if brightness < float(self.settings.passive_lighting_min):
            return f"Environmental Error: lighting too low ({brightness:.1f})", 0.0
        if brightness > float(self.settings.passive_lighting_max):
            return f"Environmental Error: lighting too high ({brightness:.1f})", 0.0
        return None, soft_penalty

    def score(self, face_bgr: np.ndarray) -> PassiveLivenessResult:
        if face_bgr.size == 0:
            return PassiveLivenessResult(confidence=0.0, environmental_error="Environmental Error: empty face crop")

        env_error, soft_penalty = self._environment_check(face_bgr)
        if env_error:
            return PassiveLivenessResult(confidence=0.0, environmental_error=env_error)

        model = self.model
        if model is None or not self.model_ready:
            return PassiveLivenessResult(
                confidence=0.0,
                reason="Passive anti-spoof model unavailable. Train and place model at configured path.",
            )

        sample = self.preprocess(face_bgr)
        pred = model.predict(np.expand_dims(sample, axis=0), verbose=0)
        confidence = float(np.clip(float(pred.reshape(-1)[0]), 0.0, 1.0))

        calibrated_bias = float(self.calibration.get("score_bias", 0.0))
        confidence = float(np.clip(confidence + calibrated_bias - soft_penalty, 0.0, 1.0))

        return PassiveLivenessResult(confidence=confidence, reason="passive_inference")
