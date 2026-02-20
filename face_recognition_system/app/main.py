from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, company, users, face, logs, attendance
from app.core.config import get_settings
from app.web.admin import router as admin_router, BASE_DIR as ADMIN_BASE_DIR

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(company.router, prefix=settings.api_v1_prefix)
app.include_router(users.router, prefix=settings.api_v1_prefix)
app.include_router(face.router, prefix=settings.api_v1_prefix)
app.include_router(logs.router, prefix=settings.api_v1_prefix)
app.include_router(attendance.router, prefix=settings.api_v1_prefix)
app.include_router(admin_router)

app.mount("/admin/static", StaticFiles(directory=str(ADMIN_BASE_DIR / "static")), name="admin_static")


@app.get("/health")
def health_check():
    return {"status": "ok"}
