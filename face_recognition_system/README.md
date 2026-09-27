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

## AWS Lambda (container image)

This project is ready to deploy through AWS SAM as a Lambda container behind an API Gateway HTTP API. The image build downloads the `buffalo_s` model into `/opt/insightface`; Lambda will therefore not try to download a model during a cold start. CPU inference is used, because Lambda does not provide a GPU runtime.

Prerequisites: Docker, the AWS CLI configured for the target AWS account, and the AWS SAM CLI. From this directory:

```bash
sam build
sam deploy --guided --parameter-overrides FaceEngineKey='use-a-long-random-secret-here'
```

SAM creates an ECR repository, pushes the container image, creates the Lambda function and exposes `FaceEngineApiUrl` in the stack outputs. Set Laravel to:

```env
FACE_MATCHER=python
FACE_ENGINE_URL=https://your-api-id.execute-api.your-region.amazonaws.com
FACE_ENGINE_KEY=the-same-long-random-secret
```

Use the `X-Face-Engine-Key` header for each direct request. Start with 3008 MB memory and the configured 29-second timeout; adjust those values after measuring real image sizes and latency. API Gateway and Lambda have payload limits, so keep encoded image requests below the configured 5 MB raw-image limit. For public production traffic, restrict access with an API Gateway authorizer, AWS WAF, or a private API integration; the header key should not be the only internet-facing protection.
