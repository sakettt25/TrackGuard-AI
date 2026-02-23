# Complete Installation Script for TrackGuard AI in .venv
# This script installs all dependencies properly in the virtual environment

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TrackGuard AI - Complete Setup (.venv)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if .venv exists
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate venv and check
$venvPython = ".\.venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    Write-Host "✗ Virtual environment Python not found" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $venvPython" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --quiet
Write-Host "✓ Pip upgraded" -ForegroundColor Green
Write-Host ""

# Install CPU-only PyTorch (avoids CUDA DLL errors)
Write-Host "Installing PyTorch (CPU-only version)..." -ForegroundColor Yellow
Write-Host "This will take a few minutes..." -ForegroundColor Gray
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ PyTorch installation failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Install other dependencies
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install opencv-python numpy ultralytics deep-sort-realtime scipy requests
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Dependency installation failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$directories = @("data", "weights", "output")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ Exists: $dir" -ForegroundColor Green
    }
}
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
$testScript = @"
import sys
try:
    import cv2
    import numpy
    import torch
    import ultralytics
    from deep_sort_realtime.deepsort_tracker import DeepSort
    print('✓ All packages imported successfully')
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')
    sys.exit(0)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"@

$testResult = & $venvPython -c $testScript
if ($LASTEXITCODE -eq 0) {
    Write-Host $testResult -ForegroundColor Green
} else {
    Write-Host "✗ Verification failed" -ForegroundColor Red
    Write-Host $testResult -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run the project: .\.venv\Scripts\python.exe main_production.py" -ForegroundColor White
Write-Host "   Or use: .\run.ps1" -ForegroundColor White
Write-Host ""
