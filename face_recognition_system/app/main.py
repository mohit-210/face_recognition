from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import cv2

from app.api.routes import auth, company, users, face, logs, attendance, engine
from app.core.config import get_settings
from app.services.face_service import get_passive_antispoof_detector, get_recognition_engine
from app.web.admin import router as admin_router, BASE_DIR as ADMIN_BASE_DIR

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(company.router, prefix=settings.api_v1_prefix)
app.include_router(users.router, prefix=settings.api_v1_prefix)
app.include_router(face.router, prefix=settings.api_v1_prefix)
app.include_router(logs.router, prefix=settings.api_v1_prefix)
app.include_router(attendance.router, prefix=settings.api_v1_prefix)
app.include_router(engine.router)
app.include_router(admin_router)

app.mount("/admin/static", StaticFiles(directory=str(ADMIN_BASE_DIR / "static")), name="admin_static")


@app.on_event("startup")
def warm_vision_models() -> None:
    # Reduce CPU burst from OpenCV internal thread oversubscription on Windows.
    try:
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    # Preload deep models once at boot to avoid first-request latency spikes.
    recognition = get_recognition_engine()
    recognition.embedder.preflight_runtime()
    get_passive_antispoof_detector()


@app.get("/health")
def health_check():
    return {"status": "ok"}
