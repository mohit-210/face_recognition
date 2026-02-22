import sys
from pathlib import Path

import cv2

try:
    from app.vision.detector import FaceDetector
except Exception:  # pragma: no cover - optional dependency for test runner
    FaceDetector = None
from app.vision.passive_antispoof import PassiveAntiSpoofDetector


def _largest_bbox(dets: list[dict]) -> list[int] | None:
    if not dets:
        return None
    det = max(
        dets,
        key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
    )
    return det.get("bbox")


def _detect_faces(image: "cv2.Mat") -> list[dict]:
    if FaceDetector is not None:
        return FaceDetector().detect(image)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    dets = []
    for (x, y, w, h) in faces:
        dets.append({"bbox": [int(x), int(y), int(x + w), int(y + h)]})
    return dets


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_antispoof.py <image_or_dir>")
        return 2

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path not found: {target}")
        return 2

    if target.is_dir():
        images = sorted([p for p in target.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        images = [target]

    if not images:
        print("No images found.")
        return 2

    antispoof = PassiveAntiSpoofDetector()

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"{img_path.name}: unreadable")
            continue
        dets = _detect_faces(image)
        bbox = _largest_bbox(dets)
        if not bbox:
            print(f"{img_path.name}: no face detected")
            continue
        result = antispoof.score_from_image(image, bbox)
        print(f"{img_path.name}: live={result.confidence:.3f} reason={result.reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
