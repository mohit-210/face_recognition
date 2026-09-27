"""AWS Lambda entry point for API Gateway HTTP API requests."""

from mangum import Mangum

from app.engine_main import app

# Keep the FastAPI application and the InsightFace model warm between Lambda
# invocations whenever AWS reuses the execution environment.
handler = Mangum(app, lifespan="auto")
