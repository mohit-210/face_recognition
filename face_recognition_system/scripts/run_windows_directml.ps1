param(
  [string]$VenvPath = ".venv",
  [string]$HostAddr = "0.0.0.0",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Python venv not found at '$python'."
}

# DirectML first, CPU fallback.
$env:PASSIVE_ANTISPOOF_BACKEND = "onnx"
$env:PASSIVE_ANTISPOOF_ONNX_PROVIDERS = "DmlExecutionProvider,CPUExecutionProvider"
$env:PASSIVE_ANTISPOOF_ONNX_PATH = "./models/passive_antispoof.onnx"
$env:PASSIVE_ANTISPOOF_CALIBRATION_PATH = "./models/passive_antispoof_calibration.json"
$env:ARCFACE_ONNX_PROVIDERS = "DmlExecutionProvider,CPUExecutionProvider"
$env:ARCFACE_CTX_ID = "0"
# Keep ArcFace model consistent with current Docker CPU runtime (:8000)
# so embeddings/matching behavior remains compatible.
$env:ARCFACE_MODEL_NAME = "buffalo_s"
$env:ARCFACE_DET_SIZE = "320"
# Keep CPU spikes controlled while preserving throughput.
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:TF_NUM_INTRAOP_THREADS = "2"
$env:TF_NUM_INTEROP_THREADS = "2"

# Local DB defaults for native Windows run (Postgres exposed on localhost:5432).
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_NAME = "face_db"
$env:DB_USER = "face_user"
$env:DB_PASSWORD = "face_pass"

Write-Host "Using ONNX providers: $env:PASSIVE_ANTISPOOF_ONNX_PROVIDERS"

& $python -m alembic -c alembic.ini upgrade head
& $python -m uvicorn app.main:app --host $HostAddr --port $Port --workers 1
