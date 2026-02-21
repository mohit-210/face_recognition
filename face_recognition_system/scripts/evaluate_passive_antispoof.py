from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras


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


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    items: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for class_name, label in (("live", 1), ("spoof", 0)):
        path = root / split / class_name
        if not path.exists():
            continue
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(file))
            if img is None:
                continue
            items.append(preprocess(img))
            labels.append(label)
            names.append(file.name)
    if not items:
        raise RuntimeError(f"No images found under: {root / split}")
    return np.asarray(items, dtype=np.float32), np.asarray(labels, dtype=np.float32), names


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    pred_live = (y_score >= threshold).astype(np.int32)
    true_live = (y_true == 1).astype(np.int32)
    true_spoof = (y_true == 0).astype(np.int32)

    tp = int(np.sum((pred_live == 1) & (true_live == 1)))
    tn = int(np.sum((pred_live == 0) & (true_spoof == 1)))
    fp = int(np.sum((pred_live == 1) & (true_spoof == 1)))
    fn = int(np.sum((pred_live == 0) & (true_live == 1)))

    total = max(1, len(y_true))
    live_total = max(1, int(np.sum(true_live)))
    spoof_total = max(1, int(np.sum(true_spoof)))

    acc = (tp + tn) / float(total)
    apcer = fp / float(spoof_total)  # spoof accepted as live
    bpcer = fn / float(live_total)  # live rejected as spoof
    acer = (apcer + bpcer) / 2.0
    tpr = tp / float(live_total)
    tnr = tn / float(spoof_total)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def find_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return threshold, eer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate passive anti-spoof model.")
    parser.add_argument("--data-root", required=True, help="Dataset root with train/ and val/ splits.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--model", default="models/passive_antispoof_mini_fasnet.keras")
    parser.add_argument("--calibration", default="models/passive_antispoof_calibration.json")
    parser.add_argument("--report-out", default="models/passive_antispoof_eval_report.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    model_path = Path(args.model)
    calibration_path = Path(args.calibration)

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    x, y, _ = load_split(root, args.split)
    model = keras.models.load_model(model_path, compile=False)
    scores = model.predict(x, verbose=0).reshape(-1)

    auc = float(roc_auc_score(y, scores))
    eer_thr, eer = find_eer(y, scores)

    cal = {}
    if calibration_path.exists():
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    cal_thr = float(cal.get("threshold_eer", eer_thr))

    m_eer = threshold_metrics(y, scores, eer_thr)
    m_cal = threshold_metrics(y, scores, cal_thr)
    m_050 = threshold_metrics(y, scores, 0.50)

    # Recommend threshold with target BPCER <= 0.15 and minimal ACER.
    candidates = []
    for thr in np.linspace(0.20, 0.80, num=121):
        m = threshold_metrics(y, scores, float(thr))
        if m["bpcer"] <= 0.15:
            candidates.append(m)
    best = min(candidates, key=lambda m: (m["acer"], m["apcer"])) if candidates else m_eer

    report = {
        "data_root": str(root),
        "split": args.split,
        "samples": int(len(y)),
        "live_samples": int(np.sum(y == 1)),
        "spoof_samples": int(np.sum(y == 0)),
        "auc": auc,
        "eer": eer,
        "thresholds": {
            "eer_threshold": eer_thr,
            "calibration_threshold": cal_thr,
            "recommended_threshold_balanced": best["threshold"],
        },
        "metrics": {
            "at_eer_threshold": m_eer,
            "at_calibration_threshold": m_cal,
            "at_0_50_threshold": m_050,
            "at_recommended_threshold": best,
        },
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
