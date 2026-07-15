# Laravel Face Engine

Small FastAPI service used by `laravel_face_recognition` when `FACE_MATCHER=python`.
Laravel handles users, companies, attendance, logs, and matching. This service only returns ArcFace embeddings for uploaded face images.

## Endpoints
- `GET /health`
- `GET /engine/health`
- `POST /engine/embed`
- `POST /engine/compare`

## Local Run
```bash
cd face_recognition_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.engine_main:app --host 127.0.0.1 --port 9001
```

Laravel should use:
```env
FACE_MATCHER=python
FACE_ENGINE_URL=http://127.0.0.1:9001
FACE_ENGINE_KEY=change-face-engine-key
```

## Docker Run
```bash
cd face_recognition_system
docker compose up --build
```

The container exposes port `9001`.

## Environment Variables
```env
APP_NAME=Laravel Face Engine
FACE_ENGINE_KEY=change-face-engine-key
ARCFACE_MODEL_NAME=buffalo_s
ARCFACE_ONNX_PROVIDERS=CPUExecutionProvider
ARCFACE_CTX_ID=-1
ARCFACE_DET_SIZE=320
ENGINE_MAX_IMAGE_BYTES=5242880
```

InsightFace downloads its ArcFace model files into its cache on first run. Those generated cache files are intentionally not committed.
