# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN PIP_REQUIRE_HASHES=0 pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
CMD ["uvicorn", "app.engine_main:app", "--host", "0.0.0.0", "--port", "9001"]
