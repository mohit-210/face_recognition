from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, roc_curve


def preprocess_hwc(image_bgr: np.ndarray, out_size: int, mode: str) -> np.ndarray:
    resized = cv2.resize(image_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    mode_norm = mode.strip().lower()
    if mode_norm == "keras_legacy":
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap = np.clip(np.abs(lap), 0.0, 1.0)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        magnitude = magnitude / (float(np.max(magnitude)) + 1e-6)
        return np.stack([gray, lap, magnitude.astype(np.float32)], axis=-1).astype(np.float32)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    if mode_norm == "imagenet":
        rgb = rgb / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
    else:
        norm = (rgb - 127.5) / 128.0
    return norm.astype(np.float32)


def infer_input_size(session: ort.InferenceSession, fallback: int) -> int:
    shape = session.get_inputs()[0].shape
    dims = [d for d in shape if isinstance(d, int) and d > 0]
    for candidate in (128, 224, 112, 96, 80):
        if candidate in dims:
            return int(candidate)
    return int(fallback)


def infer_input_layout(session: ort.InferenceSession) -> str:
    shape = session.get_inputs()[0].shape
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        c_last = shape[-1]
        c_mid = shape[1]
        if isinstance(c_last, int) and c_last == 3:
            return "nhwc"
        if isinstance(c_mid, int) and c_mid == 3:
            return "nchw"
    return "nchw"


def onnx_live_score(pred: np.ndarray, live_index: int, temperature: float, clip: bool) -> float:
    arr = np.asarray(pred, dtype=np.float32)
    if arr.size == 0:
        return 0.0

    if arr.size == 1:
        val = float(arr.reshape(-1)[0])
        conf = val if 0.0 <= val <= 1.0 else float(1.0 / (1.0 + np.exp(-val)))
        return float(np.clip(conf, 0.01, 0.99)) if clip else float(np.clip(conf, 0.0, 1.0))

    vec = arr if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)[0]
    if live_index < 0 or live_index >= vec.shape[0]:
        live_index = min(1, vec.shape[0] - 1)
    spoof_index = 1 - live_index if vec.shape[0] == 2 else None

    if np.all(vec >= 0.0) and np.all(vec <= 1.0):
        s = float(np.sum(vec))
        probs = vec if 0.95 <= s <= 1.05 else (vec / max(1e-6, s))
        if probs.shape[0] == 2 and spoof_index is not None:
            margin = float(probs[live_index] - probs[spoof_index])
            conf = 1.0 / (1.0 + np.exp(-(margin * temperature)))
        else:
            conf = float(probs[live_index])
    else:
        if vec.shape[0] == 2 and spoof_index is not None:
            margin = float(vec[live_index] - vec[spoof_index])
            conf = 1.0 / (1.0 + np.exp(-(margin / temperature)))
        else:
            shifted = vec - float(np.max(vec))
            exps = np.exp(shifted)
            probs = exps / max(1e-6, float(np.sum(exps)))
            conf = float(probs[live_index])

    if clip:
        return float(np.clip(conf, 0.01, 0.99))
    return float(np.clip(conf, 0.0, 1.0))


def load_split(root: Path, split: str) -> tuple[list[np.ndarray], np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
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
            images.append(img)
            labels.append(label)
    if not images:
        raise RuntimeError(f"No images found under: {root / split}")
    return images, np.asarray(labels, dtype=np.float32)


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    pred_live = (y_score >= threshold).astype(np.int32)
    true_live = (y_true == 1).astype(np.int32)
    true_spoof = (y_true == 0).astype(np.int32)
    tp = int(np.sum((pred_live == 1) & (true_live == 1)))
    tn = int(np.sum((pred_live == 0) & (true_spoof == 1)))
    fp = int(np.sum((pred_live == 1) & (true_spoof == 1)))
    fn = int(np.sum((pred_live == 0) & (true_live == 1)))
    live_total = max(1, int(np.sum(true_live)))
    spoof_total = max(1, int(np.sum(true_spoof)))
    total = max(1, len(y_true))
    apcer = fp / float(spoof_total)
    bpcer = fn / float(live_total)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / float(total)),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float((apcer + bpcer) / 2.0),
        "tpr": float(tp / float(live_total)),
        "tnr": float(tn / float(spoof_total)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def find_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float(thresholds[idx]), float((fpr[idx] + fnr[idx]) / 2.0)


def score_stats(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "p05": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX passive anti-spoof model and generate calibration.")
    parser.add_argument("--data-root", required=True, help="Dataset root with train/ and val/ splits.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--onnx", default="models/passive_antispoof.onnx")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--preprocess", choices=["minifasnet", "imagenet", "keras_legacy"], default="minifasnet")
    parser.add_argument("--live-index", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=6.0)
    parser.add_argument("--clip-runtime", action="store_true", help="Apply runtime clipping to [0.01,0.99].")
    parser.add_argument("--calibration-out", default="models/passive_antispoof_calibration.json")
    parser.add_argument("--report-out", default="models/passive_antispoof_eval_report_onnx.json")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(Path(args.onnx)), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_size = infer_input_size(session, args.input_size)
    input_layout = infer_input_layout(session)

    images, y = load_split(Path(args.data_root), args.split)
    scores: list[float] = []
    for img in images:
        hwc = preprocess_hwc(img, out_size=input_size, mode=args.preprocess)
        if input_layout == "nhwc":
            sample = np.expand_dims(hwc, axis=0).astype(np.float32)
        else:
            sample = np.expand_dims(np.transpose(hwc, (2, 0, 1)), axis=0).astype(np.float32)
        pred = session.run([output_name], {input_name: sample})[0]
        score = onnx_live_score(
            pred,
            live_index=args.live_index,
            temperature=float(args.temperature),
            clip=bool(args.clip_runtime),
        )
        scores.append(score)

    y_score = np.asarray(scores, dtype=np.float32)
    auc = float(roc_auc_score(y, y_score))
    eer_thr, eer = find_eer(y, y_score)
    m_eer = threshold_metrics(y, y_score, eer_thr)
    m_050 = threshold_metrics(y, y_score, 0.50)

    candidates = []
    for thr in np.linspace(0.01, 0.99, num=197):
        m = threshold_metrics(y, y_score, float(thr))
        if m["bpcer"] <= 0.15:
            candidates.append(m)
    best = min(candidates, key=lambda m: (m["acer"], m["apcer"])) if candidates else m_eer

    live_scores = y_score[y == 1]
    spoof_scores = y_score[y == 0]
    report = {
        "model": str(args.onnx),
        "providers_requested": providers,
        "providers_applied": session.get_providers(),
        "input_name": input_name,
        "output_name": output_name,
        "input_size": int(input_size),
        "input_layout": input_layout,
        "preprocess": args.preprocess,
        "live_index": int(args.live_index),
        "temperature": float(args.temperature),
        "clip_runtime": bool(args.clip_runtime),
        "data_root": str(args.data_root),
        "split": args.split,
        "samples": int(len(y)),
        "live_samples": int(np.sum(y == 1)),
        "spoof_samples": int(np.sum(y == 0)),
        "auc": auc,
        "eer": eer,
        "thresholds": {
            "eer_threshold": eer_thr,
            "recommended_threshold_balanced": best["threshold"],
        },
        "metrics": {
            "at_eer_threshold": m_eer,
            "at_0_50_threshold": m_050,
            "at_recommended_threshold": best,
        },
        "score_distribution": {
            "live": score_stats(live_scores) if live_scores.size else {},
            "spoof": score_stats(spoof_scores) if spoof_scores.size else {},
        },
    }

    calibration = {
        "threshold_eer": eer_thr,
        "eer": eer,
        "val_auc_best": auc,
        "val_acc_best": m_eer["accuracy"],
        "score_bias": 0.0,
        "onnx": {
            "preprocess": args.preprocess,
            "live_index": int(args.live_index),
            "temperature": float(args.temperature),
            "clip_runtime": bool(args.clip_runtime),
            "input_size": int(input_size),
        },
        "notes": "ONNX calibration generated from evaluate_passive_antispoof_onnx.py",
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cal_path = Path(args.calibration_out)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved calibration: {cal_path}")


if __name__ == "__main__":
    main()
