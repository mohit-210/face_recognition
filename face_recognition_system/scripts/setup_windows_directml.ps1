param(
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Python venv not found at '$python'. Create venv first: py -3.11 -m venv .venv"
}

Write-Host "Using venv Python: $python"

& $python -m pip install --upgrade pip

# Base dependencies first.
& $python -m pip install -r requirements.txt

# Switch ONNX runtime package to DirectML on Windows.
& $python -m pip uninstall -y onnxruntime | Out-Null
& $python -m pip install onnxruntime-directml==1.20.1

# Quick provider check.
& $python -c "import onnxruntime as ort; print('ONNX providers:', ort.get_available_providers())"

Write-Host ""
Write-Host "DirectML setup complete."
Write-Host "Next: run scripts\run_windows_directml.ps1"
