from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    out: list[Path] = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val anti-spoof dataset from partially extracted CelebA-Spoof.")
    parser.add_argument("--src-root", required=True, help="Path like datasets/CelebA-Spoof-partial/CelebA_Spoof/Data/test")
    parser.add_argument("--out-root", default="datasets/passive_antispoof_celeba_partial_v1")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-live-per-subject", type=int, default=120)
    parser.add_argument("--max-spoof-per-subject", type=int, default=120)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    if not src_root.exists():
        raise RuntimeError(f"Source root not found: {src_root}")
    if out_root.exists():
        raise RuntimeError(f"Output already exists: {out_root}")

    rng = random.Random(args.seed)
    subject_dirs = [p for p in src_root.iterdir() if p.is_dir()]
    usable: list[tuple[str, list[Path], list[Path]]] = []
    for subj in subject_dirs:
        live = _list_images(subj / "live")
        spoof = _list_images(subj / "spoof")
        if not live or not spoof:
            continue
        rng.shuffle(live)
        rng.shuffle(spoof)
        usable.append((subj.name, live[: args.max_live_per_subject], spoof[: args.max_spoof_per_subject]))

    if len(usable) < 4:
        raise RuntimeError(f"Not enough usable subjects found: {len(usable)}")

    rng.shuffle(usable)
    val_count = max(1, int(len(usable) * float(args.val_ratio)))
    val_subjects = {name for name, _, _ in usable[:val_count]}

    for split in ("train", "val"):
        for cls in ("live", "spoof"):
            _ensure(out_root / split / cls)

    copied = {"train_live": 0, "train_spoof": 0, "val_live": 0, "val_spoof": 0}
    for subj_name, live_list, spoof_list in usable:
        split = "val" if subj_name in val_subjects else "train"
        for i, src in enumerate(live_list):
            dst = out_root / split / "live" / f"{subj_name}_live_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            copied[f"{split}_live"] += 1
        for i, src in enumerate(spoof_list):
            dst = out_root / split / "spoof" / f"{subj_name}_spoof_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            copied[f"{split}_spoof"] += 1

    print(f"Prepared from subjects: total={len(usable)} val={len(val_subjects)} train={len(usable)-len(val_subjects)}")
    print(
        "Counts: "
        f"train/live={copied['train_live']} train/spoof={copied['train_spoof']} "
        f"val/live={copied['val_live']} val/spoof={copied['val_spoof']}"
    )
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
