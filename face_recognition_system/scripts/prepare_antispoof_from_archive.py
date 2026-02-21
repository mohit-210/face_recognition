from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _extract_video_frames(video_path: Path, max_frames: int = 12) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 0:
        indices = np.linspace(0, max(total - 1, 0), num=max_frames, dtype=int).tolist()
    else:
        indices = []

    if indices:
        target_iter = iter(indices)
        next_idx = next(target_iter, None)
        frame_idx = 0
        while next_idx is not None:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= next_idx:
                frames.append(frame)
                next_idx = next(target_iter, None)
            frame_idx += 1
    else:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    cap.release()
    return frames


def _make_spoof_variants(img: np.ndarray, rng: random.Random) -> list[np.ndarray]:
    h, w = img.shape[:2]
    out: list[np.ndarray] = []

    # Screen replay style: sinusoidal bands + blur + jpeg artifacts.
    stripe = np.zeros((h, w), dtype=np.float32)
    freq = rng.uniform(7.0, 18.0)
    phase = rng.uniform(0.0, np.pi)
    for y in range(h):
        stripe[y, :] = 0.5 + 0.5 * np.sin((2.0 * np.pi * y) / freq + phase)
    replay = np.clip(img.astype(np.float32) * (0.72 + 0.28 * np.dstack([stripe, stripe, stripe])), 0, 255).astype(np.uint8)
    replay = cv2.GaussianBlur(replay, (3, 3), 1.0)
    out.append(replay)

    # Print attack style: quantization + mild perspective + glare line.
    quant = ((img // 16) * 16).astype(np.uint8)
    quant = cv2.GaussianBlur(quant, (5, 5), 1.1)
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dx = int(0.04 * w)
    dy = int(0.04 * h)
    dst = np.float32([[dx, dy], [w - 1 - dx, 0], [0, h - 1 - dy], [w - 1, h - 1]])
    m = cv2.getPerspectiveTransform(src, dst)
    print_like = cv2.warpPerspective(quant, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    glare = print_like.astype(np.float32)
    line_y = rng.randint(int(0.2 * h), int(0.8 * h))
    cv2.line(glare, (0, line_y), (w - 1, min(h - 1, line_y + 8)), (220, 220, 220), 4)
    print_like = np.clip(glare, 0, 255).astype(np.uint8)
    out.append(print_like)

    return out


def _save_jpg(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build passive anti-spoof dataset from archive.zip (real samples).")
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--out-root", default="datasets/passive_antispoof_from_archive")
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--max-live-per-subject", type=int, default=24)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    zip_path = Path(args.zip_path)
    out_root = Path(args.out_root)

    if out_root.exists():
        # Keep behavior explicit and reproducible.
        raise RuntimeError(f"Output path already exists: {out_root}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = "real_30.csv"
        if csv_name not in {i.filename for i in zf.infolist()}:
            raise RuntimeError("real_30.csv not found in archive")

        rows = []
        with zf.open(csv_name, "r") as f:
            text = f.read().decode("utf-8", errors="ignore").splitlines()
            reader = csv.DictReader(text)
            rows = list(reader)
        if not rows:
            raise RuntimeError("No rows found in real_30.csv")

        subject_ids: list[str] = []
        by_subject: dict[str, list[np.ndarray]] = {}

        temp_dir = out_root / "_tmp_extract"
        _ensure(temp_dir)

        for row in rows:
            selfie_rel = f"samples/{row['selfie_link']}"
            video_rel = f"samples/{row['video_link']}"
            subject = row["selfie_link"].split("/", 1)[0]

            live_frames: list[np.ndarray] = []

            if selfie_rel in {i.filename for i in zf.infolist()}:
                data = zf.read(selfie_rel)
                img = _decode_image(data)
                if img is not None:
                    live_frames.append(img)

            if video_rel in {i.filename for i in zf.infolist()}:
                video_out = temp_dir / Path(video_rel).name
                video_out.write_bytes(zf.read(video_rel))
                live_frames.extend(_extract_video_frames(video_out, max_frames=args.frames_per_video))

            if not live_frames:
                continue

            rng.shuffle(live_frames)
            clipped = live_frames[: max(4, int(args.max_live_per_subject))]
            if subject not in by_subject:
                subject_ids.append(subject)
                by_subject[subject] = []
            by_subject[subject].extend(clipped)

    if not by_subject:
        raise RuntimeError("No usable samples extracted from archive")

    rng.shuffle(subject_ids)
    val_count = max(1, int(len(subject_ids) * float(args.val_ratio)))
    val_subjects = set(subject_ids[:val_count])

    for split in ("train", "val"):
        for cls in ("live", "spoof"):
            _ensure(out_root / split / cls)

    live_written = 0
    spoof_written = 0
    for subject in subject_ids:
        split = "val" if subject in val_subjects else "train"
        images = by_subject[subject]
        for idx, img in enumerate(images):
            resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            live_name = f"{subject}_live_{idx:04d}.jpg"
            _save_jpg(out_root / split / "live" / live_name, resized)
            live_written += 1

            for v_idx, spoof in enumerate(_make_spoof_variants(resized, rng)):
                spoof_name = f"{subject}_spoof_{idx:04d}_{v_idx}.jpg"
                _save_jpg(out_root / split / "spoof" / spoof_name, spoof)
                spoof_written += 1

    print(f"Prepared dataset: {out_root}")
    print(f"Subjects: total={len(subject_ids)} train={len(subject_ids)-len(val_subjects)} val={len(val_subjects)}")
    print(f"Images: live={live_written} spoof={spoof_written}")


if __name__ == "__main__":
    main()
