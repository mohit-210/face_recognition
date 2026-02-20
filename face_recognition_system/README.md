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
1. Install dependencies:
```bash
cd face_recognition_system
pip install -r requirements.txt
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
