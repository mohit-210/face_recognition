import cv2
from fastapi import FastAPI

from app.api.routes import engine
from app.core.config import get_settings
from app.api.routes.engine import get_engine_embedder

settings = get_settings()
app = FastAPI(title=f"{settings.app_name} Engine")

app.include_router(engine.router)


@app.on_event("startup")
def warm_engine_model() -> None:
    try:
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    get_engine_embedder().preflight_runtime()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "face-engine"}
