from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_access_token, verify_password
from app.database.session import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.services.company_service import CompanyService
from app.services.face_service import FaceService
from app.services.log_service import LogService
from app.services.user_service import UserService
from app.vision.detector import FaceDetector

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
settings = get_settings()
detector = FaceDetector()
guidance_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

router = APIRouter(prefix="/admin", tags=["Admin Panel"])


class FaceLockManager:
    """
    Smooths bounding boxes and enforces temporal stability before lock.
    """

    def __init__(self) -> None:
        self.prev_bbox: Optional[np.ndarray] = None
        self.smooth_alpha = 0.75
        self.stability_count = 0
        self.STABILITY_THRESHOLD = 4

    def update_and_smooth(self, current_bbox: list[int]) -> list[int]:
        curr = np.array(current_bbox, dtype=np.float32)
        if self.prev_bbox is None:
            self.prev_bbox = curr
            return current_bbox

        smoothed = (self.smooth_alpha * curr) + ((1.0 - self.smooth_alpha) * self.prev_bbox)
        self.prev_bbox = smoothed
        return smoothed.astype(int).tolist()

    def reset(self) -> None:
        self.prev_bbox = None
        self.stability_count = 0


_lock_managers: dict[int, FaceLockManager] = {}


def _get_lock_manager(user_id: int) -> FaceLockManager:
    manager = _lock_managers.get(user_id)
    if manager is None:
        manager = FaceLockManager()
        _lock_managers[user_id] = manager
    return manager


def _redirect(path: str, msg: str = "", error: str = "") -> RedirectResponse:
    query = []
    if msg:
        query.append(f"msg={msg}")
    if error:
        query.append(f"error={error}")
    suffix = "?" + "&".join(query) if query else ""
    return RedirectResponse(url=f"{path}{suffix}", status_code=303)


def _current_user_or_none(request: Request, db: Session) -> User | None:
    token = request.cookies.get("admin_token")
    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            return None
        employee_code = payload.get("sub")
        company_id = payload.get("company_id")
        if not employee_code or not company_id:
            return None
    except JWTError:
        return None

    user = UserRepository(db).get_by_employee_code(company_id=company_id, employee_code=employee_code)
    if not user or user.status != "active":
        return None
    return user


def _require_admin(request: Request, db: Session) -> User:
    user = _current_user_or_none(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Admin session required")
    return user


def _image_to_base64(upload: UploadFile) -> str:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image uploaded")
    return base64.b64encode(data).decode("utf-8")


def _decode_upload_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    upload.file.seek(0)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image uploaded")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    return image


def _resize_for_detection(image_bgr: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_side:
        return image_bgr, 1.0
    scale = max_side / float(max(h, w))
    resized = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, 1.0 / scale


def _scale_bbox(bbox: list[int], scale: float) -> list[int]:
    return [int(round(v * scale)) for v in bbox]


def _clamp_bbox(bbox: list[int], img_w: int, img_h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return [int(x1), int(y1), int(x2), int(y2)]


def _fast_detect_faces(image_bgr: np.ndarray) -> list[dict]:
    if guidance_cascade.empty():
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = guidance_cascade.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(40, 40),
    )
    h, w = image_bgr.shape[:2]
    detections: list[dict] = []
    for (x, y, fw, fh) in faces:
        area_ratio = (fw * fh) / float(max(1, w * h))
        # Pseudo-confidence for UX guidance flow.
        score = float(min(0.98, 0.78 + (area_ratio * 1.3)))
        detections.append({"bbox": [int(x), int(y), int(x + fw), int(y + fh)], "score": score})
    detections.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]), reverse=True)
    return detections


def _frame_guidance(
    image_bgr: np.ndarray,
    lock_mgr: FaceLockManager,
    update_state: bool = True,
    target_x: float = 0.5,
    target_y: float = 0.5,
) -> dict:
    h, w = image_bgr.shape[:2]
    analysis_image, scale = _resize_for_detection(image_bgr, max_side=512)
    detections = detector.detect(analysis_image)
    if not detections:
        detections = _fast_detect_faces(analysis_image)

    if not detections:
        if update_state:
            lock_mgr.reset()
        return {
            "ready": False,
            "status": "SEARCHING",
            "guidance": "No face detected",
            "quality": 0.0,
            "lock_score": 0.0,
            "lock_progress": 0,
            "bbox": None,
            "frame_width": w,
            "frame_height": h,
            "metrics": {"face_area_ratio": 0.0, "off_x": 0.0, "off_y": 0.0, "brightness": 0.0, "sharpness": 0.0},
        }

    if len(detections) > 1:
        # Keep the largest face when extra detections are tiny/noisy.
        ranked = sorted(
            detections,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        p = ranked[0]["bbox"]
        s = ranked[1]["bbox"]
        p_area = max(1, (p[2] - p[0]) * (p[3] - p[1]))
        s_area = max(1, (s[2] - s[0]) * (s[3] - s[1]))
        if (s_area / float(p_area)) >= 0.35:
            if update_state:
                lock_mgr.reset()
            return {
                "ready": False,
                "status": "MULTIPLE",
                "guidance": "Too many faces",
                "quality": 0.0,
                "lock_score": 0.0,
                "lock_progress": 0,
                "bbox": None,
                "frame_width": w,
                "frame_height": h,
                "metrics": {"face_area_ratio": 0.0, "off_x": 0.0, "off_y": 0.0, "brightness": 0.0, "sharpness": 0.0},
            }
        detections = [ranked[0]]

    det = detections[0]
    raw_bbox = _scale_bbox(det["bbox"], scale)
    bbox = lock_mgr.update_and_smooth(raw_bbox) if update_state else raw_bbox
    bbox = _clamp_bbox(bbox, w, h)
    score = float(det.get("score", 0.0))

    target_x = max(0.05, min(0.95, float(target_x)))
    target_y = max(0.05, min(0.95, float(target_y)))
    target_px_x = w * target_x
    target_px_y = h * target_y

    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    off_x = (cx - target_px_x) / w
    off_y = (cy - target_px_y) / h

    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    face_area_ratio = (face_w * face_h) / float(w * h)

    gray_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_frame))
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    else:
        sharpness = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

    is_centered = abs(off_x) < 0.16 and abs(off_y) < 0.16
    is_sized = 0.10 < face_area_ratio < 0.55
    is_sharp = sharpness > 30.0

    next_stability = lock_mgr.stability_count
    if is_centered and is_sized and is_sharp and score >= 0.70:
        next_stability += 1
    else:
        next_stability = max(0, next_stability - 1)
    next_stability = min(next_stability, lock_mgr.STABILITY_THRESHOLD)
    if update_state:
        lock_mgr.stability_count = next_stability

    is_locked = next_stability >= lock_mgr.STABILITY_THRESHOLD
    lock_progress = min(100, int((next_stability / lock_mgr.STABILITY_THRESHOLD) * 100))
    lock_score = lock_progress / 100.0

    guidance = "HOLD STILL..." if is_locked else "Aligning..."
    if score < 0.70:
        guidance = "Face detection weak"
    elif not is_centered:
        guidance = "Center your face"
    elif face_area_ratio < 0.10:
        guidance = "Move closer"
    elif face_area_ratio > 0.55:
        guidance = "Move back"
    elif not is_sharp:
        guidance = "Hold still (Focusing)"

    return {
        "ready": is_locked,
        "status": "LOCKED" if is_locked else "TRACKING",
        "guidance": guidance,
        "quality": score,
        "bbox": bbox,
        "frame_width": w,
        "frame_height": h,
        "target_x": target_x,
        "target_y": target_y,
        "lock_progress": lock_progress,
        "lock_score": lock_score,
        "metrics": {
            "area": round(face_area_ratio, 2),
            "sharpness": round(sharpness, 1),
            "offset_x": round(off_x, 2),
            "face_area_ratio": round(face_area_ratio, 4),
            "off_x": round(off_x, 4),
            "off_y": round(off_y, 4),
            "brightness": round(brightness, 2),
        },
    }


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    if _current_user_or_none(request, db):
        return RedirectResponse(url="/admin", status_code=303)
    companies = CompanyService(db).list_all()
    return templates.TemplateResponse(
        request,
        "admin/login.html",
        {
            "companies": companies,
            "msg": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@router.post("/login")
def login_submit(
    company_id: int = Form(...),
    employee_code: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = UserRepository(db).get_by_employee_code(company_id=company_id, employee_code=employee_code)
    if not user or not verify_password(password, user.password_hash) or user.status != "active":
        return _redirect("/admin/login", error="Invalid credentials or inactive user")

    token = create_access_token(subject=user.employee_code, company_id=user.company_id)
    response = _redirect("/admin", msg="Login successful")
    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * settings.access_token_expire_minutes,
    )
    return response


@router.post("/bootstrap/company")
def bootstrap_company(name: str = Form(...), db: Session = Depends(get_db)):
    try:
        CompanyService(db).create(name=name.strip())
        return _redirect("/admin/login", msg="Company created. Add first user below.")
    except HTTPException as exc:
        return _redirect("/admin/login", error=str(exc.detail))


@router.post("/bootstrap/user")
def bootstrap_first_user(
    company_id: int = Form(...),
    name: str = Form(...),
    employee_code: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    company_service = CompanyService(db)
    user_repo = UserRepository(db)

    try:
        company_service.ensure_exists(company_id)
    except HTTPException as exc:
        return _redirect("/admin/login", error=str(exc.detail))

    if user_repo.list_by_company(company_id):
        return _redirect(
            "/admin/login",
            error="Bootstrap user creation is only allowed when company has no users.",
        )

    try:
        UserService(db).create(
            company_id=company_id,
            name=name.strip(),
            employee_code=employee_code.strip(),
            password=password,
        )
        return _redirect("/admin/login", msg="First user created. You can log in now.")
    except HTTPException as exc:
        return _redirect("/admin/login", error=str(exc.detail))


@router.post("/logout")
def logout():
    response = _redirect("/admin/login", msg="Logged out")
    response.delete_cookie("admin_token")
    return response


@router.get("", response_class=HTMLResponse)
def admin_home(
    request: Request,
    user_id: int | None = None,
    date: str | None = None,
    db: Session = Depends(get_db),
):
    current = _current_user_or_none(request, db)
    if not current:
        return RedirectResponse(url="/admin/login", status_code=303)

    company_service = CompanyService(db)
    user_service = UserService(db)
    log_service = LogService(db)

    companies = company_service.list_all()
    users = user_service.list_by_company(current.company_id)

    date_filter = None
    if date:
        try:
            date_filter = datetime.fromisoformat(date)
        except ValueError:
            date_filter = None

    logs = log_service.list(company_id=current.company_id, user_id=user_id, date=date_filter)

    return templates.TemplateResponse(
        request,
        "admin/dashboard.html",
        {
            "current_user": current,
            "companies": companies,
            "users": users,
            "logs": logs,
            "msg": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
            "selected_user_id": user_id,
            "selected_date": date or "",
        },
    )


@router.post("/company/create")
def create_company(request: Request, name: str = Form(...), db: Session = Depends(get_db)):
    _require_admin(request, db)
    try:
        CompanyService(db).create(name=name.strip())
        return _redirect("/admin", msg="Company created")
    except HTTPException as exc:
        return _redirect("/admin", error=str(exc.detail))


@router.post("/users/create")
def create_user(
    request: Request,
    name: str = Form(...),
    employee_code: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    current = _require_admin(request, db)
    try:
        UserService(db).create(
            company_id=current.company_id,
            name=name.strip(),
            employee_code=employee_code.strip(),
            password=password,
        )
        return _redirect("/admin", msg="User created")
    except HTTPException as exc:
        return _redirect("/admin", error=str(exc.detail))


@router.post("/users/{user_id}/update")
def update_user(
    user_id: int,
    request: Request,
    name: str = Form(...),
    status: str = Form(...),
    password: str = Form(""),
    db: Session = Depends(get_db),
):
    current = _require_admin(request, db)
    service = UserService(db)
    user = service.get(user_id)
    if user.company_id != current.company_id:
        return _redirect("/admin", error="Cross-company update denied")

    try:
        service.update(
            user_id=user_id,
            name=name.strip(),
            status=status,
            password=password.strip() or None,
        )
        return _redirect("/admin", msg="User updated")
    except HTTPException as exc:
        return _redirect("/admin", error=str(exc.detail))


@router.post("/users/{user_id}/delete")
def delete_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    current = _require_admin(request, db)
    service = UserService(db)
    user = service.get(user_id)
    if user.company_id != current.company_id:
        return _redirect("/admin", error="Cross-company delete denied")

    service.delete(user_id)
    return _redirect("/admin", msg="User deleted")


@router.post("/users/{user_id}/toggle")
def toggle_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    current = _require_admin(request, db)
    service = UserService(db)
    user = service.get(user_id)
    if user.company_id != current.company_id:
        return _redirect("/admin", error="Cross-company toggle denied")

    new_status = "inactive" if user.status == "active" else "active"
    service.update(user_id=user_id, name=user.name, status=new_status, password=None)
    return _redirect("/admin", msg=f"User {new_status}")


@router.post("/face/register")
def register_face(
    request: Request,
    user_id: int = Form(...),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    current = _require_admin(request, db)
    user = UserService(db).get(user_id)
    if user.company_id != current.company_id:
        return _redirect("/admin", error="Cross-company face registration denied")

    try:
        image_payloads = [_image_to_base64(file) for file in images]
        saved = FaceService(db).register_embeddings(user_id=user_id, images_base64=image_payloads)
        return _redirect("/admin", msg=f"Face embeddings saved: {saved}")
    except HTTPException as exc:
        return _redirect("/admin", error=str(exc.detail))


@router.post("/face/verify")
def verify_face(
    request: Request,
    user_id: int = Form(...),
    image: UploadFile = File(...),
    previous_image: UploadFile | None = File(default=None),
    expected_challenge: str = Form(default=""),
    challenge_response: str = Form(default=""),
    device_id: str = Form(default="admin-web"),
    db: Session = Depends(get_db),
):
    current = _require_admin(request, db)
    user = UserService(db).get(user_id)
    if user.company_id != current.company_id:
        return _redirect("/admin", error="Cross-company verification denied")

    try:
        lock_mgr = _get_lock_manager(current.id)
        frame = _decode_upload_image(image)
        guide = _frame_guidance(frame, lock_mgr, update_state=False)
        if not guide.get("ready", False):
            return _redirect("/admin", error=f"Alignment lost: {guide.get('guidance', 'Face not stable')}")

        image_b64 = _image_to_base64(image)
        previous_b64 = _image_to_base64(previous_image) if previous_image else None
        result = FaceService(db).verify(
            company_id=current.company_id,
            user_id=user_id,
            image_base64=image_b64,
            expected_challenge=expected_challenge.strip() or None,
            challenge_response=challenge_response.strip() or None,
            previous_image_base64=previous_b64,
            device_id=device_id.strip() or None,
        )
        verdict = "VERIFIED" if result["verified"] else "REJECTED"
        message = (
            f"{verdict} | conf={result['confidence']:.4f} | "
            f"liveness={result['liveness_score']:.4f} | reason={result['reason']}"
        )
        return _redirect("/admin", msg=message)
    except HTTPException as exc:
        return _redirect("/admin", error=str(exc.detail))


@router.post("/face/identify")
def identify_face(
    request: Request,
    image: UploadFile = File(...),
    device_id: str = Form(default="admin-web"),
    fast_mode: bool = Form(default=True),
    db: Session = Depends(get_db),
):
    current = _require_admin(request, db)
    image_b64 = _image_to_base64(image)
    return FaceService(db).identify_in_company(
        company_id=current.company_id,
        image_base64=image_b64,
        device_id=device_id.strip() or None,
        enforce_liveness=not bool(fast_mode),
    )


@router.post("/face/frame-guide")
def frame_guide(
    request: Request,
    image: UploadFile = File(...),
    target_x: float = Form(default=0.5),
    target_y: float = Form(default=0.5),
    db: Session = Depends(get_db),
):
    current = _require_admin(request, db)
    frame = _decode_upload_image(image)
    lock_mgr = _get_lock_manager(current.id)
    return _frame_guidance(
        frame,
        lock_mgr,
        update_state=True,
        target_x=target_x,
        target_y=target_y,
    )


@router.post("/logs/filter")
def filter_logs(user_id: int | None = Form(default=None), date: str = Form(default="")):
    query = []
    if user_id:
        query.append(f"user_id={user_id}")
    if date.strip():
        query.append(f"date={date.strip()}")
    suffix = "?" + "&".join(query) if query else ""
    return RedirectResponse(url=f"/admin{suffix}", status_code=303)
