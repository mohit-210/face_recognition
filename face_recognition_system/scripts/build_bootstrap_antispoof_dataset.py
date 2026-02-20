from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_image(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def _make_spoof_variants(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    out: list[np.ndarray] = []

    # Replay-like: moire stripes + recompression.
    stripe = np.zeros((h, w), dtype=np.float32)
    freq = random.uniform(8.0, 18.0)
    for y in range(h):
        stripe[y, :] = 0.5 + 0.5 * np.sin((2.0 * np.pi * y) / freq)
    stripe_rgb = np.dstack([stripe, stripe, stripe])
    replay = np.clip(img.astype(np.float32) * (0.76 + 0.24 * stripe_rgb), 0, 255).astype(np.uint8)
    replay = cv2.GaussianBlur(replay, (3, 3), 0.9)
    out.append(replay)

    # Print-like: reduced color depth + blur + mild perspective.
    quant = (img // 16) * 16
    quant = cv2.GaussianBlur(quant, (5, 5), 1.2)
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dx = int(0.04 * w)
    dy = int(0.04 * h)
    dst = np.float32([[dx, dy], [w - 1 - dx, 0], [0, h - 1 - dy], [w - 1, h - 1]])
    m = cv2.getPerspectiveTransform(src, dst)
    print_like = cv2.warpPerspective(quant, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    out.append(print_like)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bootstrap anti-spoof dataset from internet portraits.")
    parser.add_argument(
        "--out-root",
        default="datasets/passive_antispoof_bootstrap",
        help="Output dataset root.",
    )
    parser.add_argument("--samples", type=int, default=180, help="Number of source live portraits to fetch.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    args = parser.parse_args()

    root = Path(args.out_root)
    for split in ("train", "val"):
        for cls in ("live", "spoof"):
            _ensure_dir(root / split / cls)

    # RandomUser has 100 ids per gender.
    pool = [("men", i) for i in range(100)] + [("women", i) for i in range(100)]
    random.shuffle(pool)
    chosen = pool[: max(20, min(len(pool), args.samples))]

    live_images: list[np.ndarray] = []
    for gender, idx in chosen:
        url = f"https://randomuser.me/api/portraits/{gender}/{idx}.jpg"
        try:
            data = _download(url)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            live_images.append(img)
        except Exception:
            continue

    if len(live_images) < 40:
        raise RuntimeError(f"Too few images downloaded: {len(live_images)}")

    random.shuffle(live_images)
    val_count = max(8, int(len(live_images) * float(args.val_ratio)))
    val_set = live_images[:val_count]
    train_set = live_images[val_count:]

    def write_split(images: list[np.ndarray], split: str) -> None:
        for i, img in enumerate(images):
            _save_image(root / split / "live" / f"live_{i:05d}.jpg", img)
            for j, spoof in enumerate(_make_spoof_variants(img)):
                _save_image(root / split / "spoof" / f"spoof_{i:05d}_{j}.jpg", spoof)

    write_split(train_set, "train")
    write_split(val_set, "val")

    print(f"Dataset ready at: {root}")
    print(f"train/live={len(list((root / 'train' / 'live').glob('*.jpg')))}")
    print(f"train/spoof={len(list((root / 'train' / 'spoof').glob('*.jpg')))}")
    print(f"val/live={len(list((root / 'val' / 'live').glob('*.jpg')))}")
    print(f"val/spoof={len(list((root / 'val' / 'spoof').glob('*.jpg')))}")


if __name__ == "__main__":
    main()
