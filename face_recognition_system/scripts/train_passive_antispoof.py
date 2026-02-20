from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_curve
from tensorflow import keras


def build_minifasnet_lite(input_shape: tuple[int, int, int] = (128, 128, 3)) -> keras.Model:
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
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="mini_fasnet_lite")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model


def preprocess(image_bgr: np.ndarray, out_size: int = 128) -> np.ndarray:
    resized = cv2.resize(image_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.clip(np.abs(lap), 0.0, 1.0)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    magnitude = magnitude / (float(np.max(magnitude)) + 1e-6)
    return np.stack([gray, lap, magnitude.astype(np.float32)], axis=-1).astype(np.float32)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    items: list[np.ndarray] = []
    labels: list[int] = []
    for class_name, label in (("live", 1), ("spoof", 0)):
        path = root / split / class_name
        if not path.exists():
            continue
        for file in path.rglob("*"):
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(file))
            if img is None:
                continue
            items.append(preprocess(img))
            labels.append(label)
    if not items:
        raise RuntimeError(f"No images found under: {root / split}")
    return np.asarray(items, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def find_eer_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return threshold, eer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train passive anti-spoof MiniFASNet-lite model.")
    parser.add_argument("--data-root", required=True, help="Dataset root with train/ and val/ splits.")
    parser.add_argument(
        "--out-model",
        default="models/passive_antispoof_mini_fasnet.keras",
        help="Output keras model path.",
    )
    parser.add_argument(
        "--out-calibration",
        default="models/passive_antispoof_calibration.json",
        help="Output calibration JSON path.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.data_root)
    x_train, y_train = load_split(root, "train")
    x_val, y_val = load_split(root, "val")

    model = build_minifasnet_lite()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=2, factor=0.5, min_lr=1e-5),
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    val_auc_best = float(max(model.history.history.get("val_auc", [0.0])))
    val_acc_best = float(max(model.history.history.get("val_acc", [0.0])))

    y_score = model.predict(x_val, verbose=0).reshape(-1)
    threshold, eer = find_eer_threshold(y_val, y_score)

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)

    out_calibration = Path(args.out_calibration)
    out_calibration.parent.mkdir(parents=True, exist_ok=True)
    out_calibration.write_text(
        json.dumps(
            {
                "threshold_eer": threshold,
                "eer": eer,
                "val_auc_best": val_auc_best,
                "val_acc_best": val_acc_best,
                "score_bias": 0.0,
                "notes": "Use threshold_eer as baseline. Tune attendance threshold from real deployment ROC.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {out_model}")
    print(f"Saved calibration: {out_calibration}")
    print(f"Validation threshold@EER: {threshold:.4f} | EER: {eer:.4f}")


if __name__ == "__main__":
    main()
