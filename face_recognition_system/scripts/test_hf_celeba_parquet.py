import io
import argparse

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from app.vision.passive_antispoof import PassiveAntiSpoofDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ViT anti-spoof on HF CelebA parquet shard.")
    parser.add_argument("--parquet", required=True, help="Path to parquet shard")
    parser.add_argument("--threshold", type=float, default=0.85, help="Live threshold")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Parquet batch size")
    args = parser.parse_args()

    parquet_file = pq.ParquetFile(args.parquet)
    anti = PassiveAntiSpoofDetector()

    TP = TN = FP = FN = 0
    seen = 0

    for batch in parquet_file.iter_batches(batch_size=args.batch_size):
        cols = batch.to_pydict()
        imgs = cols.get("cropped_image")
        labels = cols.get("labels")
        if imgs is None or labels is None:
            continue
        for img_obj, label in zip(imgs, labels):
            if seen >= args.max_samples:
                break
            if img_obj is None or label is None:
                continue
            img_bytes = img_obj.get("bytes") if isinstance(img_obj, dict) else None
            if not img_bytes:
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue
            arr = np.array(img)
            if arr.ndim != 3:
                continue
            bgr = arr[:, :, ::-1].copy()
            h, w = bgr.shape[:2]
            score = anti.score_from_image(bgr, [0, 0, w, h], face_bgr=bgr).confidence
            pred_live = 1 if score >= args.threshold else 0
            # labels: 0=live, 1=spoof
            if label == 0 and pred_live == 1:
                TP += 1
            elif label == 0 and pred_live == 0:
                FN += 1
            elif label == 1 and pred_live == 0:
                TN += 1
            else:
                FP += 1
            seen += 1
        if seen >= args.max_samples:
            break

    acc = (TP + TN) / max(1, TP + TN + FP + FN)
    far = FP / max(1, FP + TN)
    frr = FN / max(1, FN + TP)

    print(f"samples={seen}")
    print(f"TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"acc={acc:.4f} FAR={far:.4f} FRR={frr:.4f} (thr={args.threshold:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
