from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

STANDALONE_KERAS = None
STANDALONE_KERAS_IMPORT_ERROR = None
try:
    import keras as STANDALONE_KERAS
except Exception as exc:  # pragma: no cover - runtime dependency may be absent in some dev setups
    STANDALONE_KERAS_IMPORT_ERROR = exc

TF_KERAS = None
TF_KERAS_IMPORT_ERROR = None
try:
    from tensorflow import keras as TF_KERAS
except Exception as exc:  # pragma: no cover - runtime dependency may be absent in some dev setups
    TF_KERAS_IMPORT_ERROR = exc

LEGACY_TF_KERAS = None
LEGACY_TF_KERAS_IMPORT_ERROR = None
try:
    import tf_keras as LEGACY_TF_KERAS
except Exception as exc:  # pragma: no cover - runtime dependency may be absent in some dev setups
    LEGACY_TF_KERAS_IMPORT_ERROR = exc

# Prefer standalone keras for modern `.keras` artifacts.
keras = STANDALONE_KERAS if STANDALONE_KERAS is not None else (TF_KERAS if TF_KERAS is not None else LEGACY_TF_KERAS)
KERAS_BACKEND = (
    "keras"
    if STANDALONE_KERAS is not None
    else ("tensorflow.keras" if TF_KERAS is not None else ("tf_keras" if LEGACY_TF_KERAS is not None else "unavailable"))
)


logger = logging.getLogger(__name__)


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
        self.onnx_session = None
        self.onnx_input_name: str | None = None
        self.onnx_output_name: str | None = None
        self.model_ready = False
        self.calibration_ok = False
        self.calibration: dict = {}
        self.model_descriptor = "passive_antispoof_unavailable"
        self.model_path = Path(self.settings.passive_antispoof_model_path)
        self.onnx_path = Path(self.settings.passive_antispoof_onnx_path)
        self.calibration_path = Path(self.settings.passive_antispoof_calibration_path)
        self._load_attempted = False
        self._load_components()

    def _load_components(self) -> None:
        self._load_attempted = True
        if self.calibration_path.exists():
            try:
                self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            except Exception:
                self.calibration = {}

        backend_pref = (self.settings.passive_antispoof_backend or "auto").strip().lower()

        if self.onnx_path.exists() and (backend_pref in {"auto", "onnx"}):
            if ort is None:
                logger.warning("ONNX backend requested but onnxruntime is unavailable.")
            else:
                try:
                    session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
                    inps = session.get_inputs()
                    outs = session.get_outputs()
                    if not inps or not outs:
                        raise RuntimeError("ONNX model has no IO tensors")
                    self.onnx_session = session
                    self.onnx_input_name = inps[0].name
                    self.onnx_output_name = outs[0].name
                    self.model_ready = True
                    self.model_descriptor = f"mini_fasnet_onnx ({self.onnx_path.name}) via onnxruntime"
                except Exception:
                    self.onnx_session = None
                    self.onnx_input_name = None
                    self.onnx_output_name = None
                    self.model_ready = False
                    logger.exception("Failed to load passive anti-spoof ONNX model from %s", self.onnx_path)

        if (not self.model_ready) and self.model_path.exists() and (backend_pref in {"auto", "keras", "tf", "tensorflow"}):
            loaders: list[tuple[str, object]] = []
            if STANDALONE_KERAS is not None:
                loaders.append(("keras", STANDALONE_KERAS))
            if TF_KERAS is not None:
                loaders.append(("tensorflow.keras", TF_KERAS))
            if LEGACY_TF_KERAS is not None:
                loaders.append(("tf_keras", LEGACY_TF_KERAS))

            self.model = None
            self.model_ready = False
            if not loaders:
                logger.warning(
                    "Passive anti-spoof backend unavailable. keras error=%s ; tensorflow.keras error=%s ; tf_keras error=%s",
                    STANDALONE_KERAS_IMPORT_ERROR,
                    TF_KERAS_IMPORT_ERROR,
                    LEGACY_TF_KERAS_IMPORT_ERROR,
                )
            for backend_name, backend in loaders:
                try:
                    self.model = backend.models.load_model(self.model_path, compile=False)
                    self.onnx_session = None
                    self.model_ready = True
                    model_name = getattr(self.model, "name", "model")
                    self.model_descriptor = f"{model_name} ({self.model_path.name}) via {backend_name}"
                    break
                except Exception:
                    self.model = None
                    self.model_ready = False
                    logger.exception(
                        "Failed to load passive anti-spoof model from %s via %s",
                        self.model_path,
                        backend_name,
                    )
        elif not self.model_path.exists() and not self.onnx_path.exists():
            self.model_ready = False
            logger.warning(
                "Passive anti-spoof model files not found at %s (keras) or %s (onnx)",
                self.model_path,
                self.onnx_path,
            )

        if self.model_ready:
            eer = float(self.calibration.get("eer", 1.0))
            val_auc_best = float(self.calibration.get("val_auc_best", 0.0))
            self.calibration_ok = (
                eer <= float(self.settings.passive_calibration_max_eer)
                and val_auc_best >= float(self.settings.passive_calibration_min_auc)
            )
            if self.settings.passive_quality_gate_enabled and not self.calibration_ok:
                self.model_ready = False
                logger.warning(
                    "Passive anti-spoof disabled by quality gate: eer=%.4f auc=%.4f (max_eer=%.2f min_auc=%.2f)",
                    eer,
                    val_auc_best,
                    float(self.settings.passive_calibration_max_eer),
                    float(self.settings.passive_calibration_min_auc),
                )
            elif not self.calibration_ok:
                logger.warning(
                    "Passive anti-spoof running in permissive mode with weak calibration: eer=%.4f auc=%.4f",
                    eer,
                    val_auc_best,
                )
        else:
            self.calibration_ok = False

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

    @staticmethod
    def _enhance_luma(face_bgr: np.ndarray) -> np.ndarray:
        """
        Mild luminance normalization for difficult lighting without distorting spoof cues.
        """
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        merged = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

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

    def _apply_threshold_calibration(self, confidence: float) -> float:
        """
        Re-map raw model confidence around calibrated EER threshold.
        Keeps low scores low while expanding headroom for genuine live frames.
        """
        thr = float(self.calibration.get("threshold_eer", 0.0))
        if not (0.05 <= thr <= 0.95):
            return float(np.clip(confidence, 0.0, 1.0))

        conf = float(np.clip(confidence, 0.0, 1.0))
        if conf >= thr:
            # Map [thr..1.0] -> [0.52..1.0]
            scaled = 0.52 + ((conf - thr) * (0.48 / max(1e-6, (1.0 - thr))))
        else:
            # Map [0..thr] -> [0..0.48]
            scaled = conf * (0.48 / max(1e-6, thr))
        return float(np.clip(scaled, 0.0, 1.0))

    def score(self, face_bgr: np.ndarray) -> PassiveLivenessResult:
        if face_bgr.size == 0:
            return PassiveLivenessResult(confidence=0.0, environmental_error="Environmental Error: empty face crop")

        env_error, soft_penalty = self._environment_check(face_bgr)
        if env_error:
            return PassiveLivenessResult(confidence=0.0, environmental_error=env_error)

        # Lazy reload prevents permanent fallback if startup happened before model/runtime became available.
        if (self.model is None or not self.model_ready) and not self._load_attempted:
            self._load_components()
        if self.model is None or not self.model_ready:
            self._load_components()

        model = self.model
        onnx_session = self.onnx_session
        if ((model is None and onnx_session is None) or not self.model_ready):
            return PassiveLivenessResult(
                confidence=0.0,
                reason=(
                    "Passive anti-spoof model unavailable. "
                    f"Ensure model exists at {self.onnx_path} (onnx) or {self.model_path} (keras)."
                ),
            )

        # Base prediction first.
        preprocess_size = int(self.settings.passive_antispoof_onnx_input_size) if onnx_session is not None else 128
        sample = self.preprocess(face_bgr, out_size=preprocess_size)
        if onnx_session is not None and self.onnx_input_name and self.onnx_output_name:
            # MiniFASNet ONNX exports are channel-first (NCHW).
            onnx_input = np.expand_dims(sample, axis=0).astype(np.float32).transpose(0, 3, 1, 2)
            pred = onnx_session.run(
                [self.onnx_output_name],
                {self.onnx_input_name: onnx_input},
            )[0]
        else:
            pred = model.predict(np.expand_dims(sample, axis=0), verbose=0)
        base_conf = float(np.clip(float(np.asarray(pred).reshape(-1)[0]), 0.0, 1.0))

        # Boundary-focused TTA: only run extra views when score is ambiguous.
        confidence = base_conf
        if 0.30 <= base_conf <= 0.75:
            face_flip = cv2.flip(face_bgr, 1)
            face_luma = self._enhance_luma(face_bgr)
            tta_batch = np.stack(
                [
                    sample,
                    self.preprocess(face_flip, out_size=preprocess_size),
                    self.preprocess(face_luma, out_size=preprocess_size),
                ],
                axis=0,
            )
            if onnx_session is not None and self.onnx_input_name and self.onnx_output_name:
                # MiniFASNet ONNX exports are channel-first (NCHW).
                onnx_tta_input = tta_batch.astype(np.float32).transpose(0, 3, 1, 2)
                tta_pred = onnx_session.run(
                    [self.onnx_output_name],
                    {self.onnx_input_name: onnx_tta_input},
                )[0]
                tta_pred = np.asarray(tta_pred).reshape(-1)
            else:
                tta_pred = model.predict(tta_batch, verbose=0).reshape(-1)
            confidence = float(np.clip(float(np.mean(tta_pred)), 0.0, 1.0))

        confidence = self._apply_threshold_calibration(confidence)

        calibrated_bias = float(self.calibration.get("score_bias", 0.0))
        confidence = float(np.clip(confidence + calibrated_bias - soft_penalty, 0.0, 1.0))

        return PassiveLivenessResult(confidence=confidence, reason="passive_inference")
