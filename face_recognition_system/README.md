# Face Recognition System

Production-oriented FastAPI service for face registration and verification with company tenancy, JWT auth, anti-spoofing, and PostgreSQL persistence.

## Features
- RetinaFace detection + ArcFace embeddings
- Liveness checks: texture, replay-pattern, challenge-response, motion variation
- Multi-company user management
- Embedding-only storage for matching
- JWT auth and company-scope enforcement
- Verification logging
- Alembic migrations

## Project Structure
```text
face_recognition_system/
  app/
    main.py
    core/
    models/
    schemas/
    api/routes/
    services/
    repositories/
    vision/
    database/migrations/
  requirements.txt
  docker-compose.yml
```

## Environment Variables
Create `face_recognition_system/.env`:
```env
APP_NAME=Face Recognition System
API_V1_PREFIX=/api/v1
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_db
DB_USER=face_user
DB_PASSWORD=face_pass
JWT_SECRET_KEY=change-this
JWT_REFRESH_SECRET_KEY=change-this-refresh
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_MINUTES=10080
RECOGNITION_THRESHOLD=0.45
LIVENESS_THRESHOLD=0.65
```

## Local Run
### Lightweight Laravel Face Engine
Use this when the Laravel project is the main app and Python only provides ArcFace embeddings.

1. Install only engine dependencies:
```bash
cd face_recognition_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the engine:
```bash
uvicorn app.engine_main:app --host 127.0.0.1 --port 9001
```

3. Test:
```bash
curl -H "X-Face-Engine-Key: change-face-engine-key" http://127.0.0.1:9001/engine/health
```

The old all-in-one Python app dependencies are kept in `requirements.full.txt`.

### Full Legacy Python API
1. Install dependencies:
```bash
cd face_recognition_system
pip install -r requirements.full.txt
```
2. Start PostgreSQL.
3. Run migrations:
```bash
cd face_recognition_system
alembic -c alembic.ini upgrade head
```
4. Start API:
```bash
uvicorn app.main:app --reload --port 8000
```
5. Open docs: `http://localhost:8000/docs`

### Windows Intel GPU (DirectML) Run
Use this if you want Intel Iris Xe acceleration for ONNX liveness on Windows (outside Docker).

1. Create and activate venv (once):
```powershell
cd face_recognition_system
py -3.11 -m venv .venv
```

2. Install DirectML runtime in venv:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_directml.ps1
```

3. Run API with DirectML provider priority:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows_directml.ps1
```

Notes:
- This path is separate from Docker and avoids NVIDIA-only runtime assumptions.
- If `DmlExecutionProvider` is unavailable on your machine, runtime falls back to CPU.

## Docker Run
```bash
cd face_recognition_system
docker compose up --build
```

## API Endpoints
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/refresh`
- `POST /api/v1/company`
- `GET /api/v1/company`
- `POST /api/v1/users`
- `PUT /api/v1/users/{id}`
- `DELETE /api/v1/users/{id}`
- `GET /api/v1/users?company_id=`
- `POST /api/v1/face/register`
- `POST /api/v1/face/verify`
- `GET /api/v1/logs?user_id=&date=`

## Security Notes
- Store only embeddings, not raw images for matching.
- Use HTTPS and rotate JWT secrets in production.
- Tune recognition/liveness thresholds on real deployment data.

## Web Admin Panel
- URL: http://localhost:8000/admin/login
- Login uses existing user credentials: company_id, employee_code, password.
- Features available from UI:
  - create/list companies
  - create/update/delete/enable/disable users (within your company)
  - register face embeddings from multiple uploaded images
  - verify face + liveness from uploaded images
  - filter and inspect verification logs

## Attendance Performance Guardrails
- Use single-pass attendance mark first; burst fallback only on uncertain results.
- API responses can include `debug_timings` when `debug_timing=true` in request payload.
- Production runtime should not use `--reload`; keep one worker for shared model cache.
- Keep model warmup enabled at startup and monitor latency SLOs:
  - attendance mark P50 < 2.5s
  - attendance mark P95 < 5s
- Recommended weekly checks:
  - failure reason distribution (`No face`, `Liveness failed`, `Face mismatch`)
  - median `debug_timings.total_ms` and `debug_timings.detect_ms`
  - cache hit ratio for company embedding index

## Attendance Anti-Spoof Policy (Balanced)
- `mark_attendance=true` uses stricter policy than normal identify:
  - live-motion required
  - fallback liveness threshold increased
  - attendance recognition threshold increased
  - fast mode disabled
  - minimum 2 verified samples and 67% burst consensus
- non-attendance identify/preview stays less strict to avoid unnecessary false rejects.

## Passive CNN Calibration
- Calibration file: `models/passive_antispoof_calibration.json`
- Retrain + calibrate (inside Docker):
```bash
docker compose -f face_recognition_system/docker-compose.yml exec -T api python scripts/train_passive_antispoof.py --data-root datasets/passive_antispoof_bootstrap --out-model models/passive_antispoof_mini_fasnet.keras --out-calibration models/passive_antispoof_calibration.json --epochs 8 --batch-size 32
```
- Runtime controls:
  - `PASSIVE_QUALITY_GATE_ENABLED=false` (default): start CNN in permissive mode, blended with heuristic liveness.
  - `PASSIVE_QUALITY_GATE_ENABLED=true`: disable CNN if calibration does not meet:
    - `PASSIVE_CALIBRATION_MAX_EER` (default `0.75`)
    - `PASSIVE_CALIBRATION_MIN_AUC` (default `0.45`)
