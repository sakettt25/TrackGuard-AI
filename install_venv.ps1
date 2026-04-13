# Complete installation script for TrackGuard AI in .venv
# This script installs all dependencies using a Python version compatible with torch/ultralytics.

$ErrorActionPreference = 'Stop'

function Get-CompatiblePythonCommand {
    # Prefer Python 3.11, then 3.10 on Windows via the py launcher.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.11 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @('py', '-3.11')
            }
        } catch {}

        try {
            & py -3.10 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @('py', '-3.10')
            }
        } catch {}
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @('python')
    }

    throw 'No Python interpreter found. Install Python 3.10 or 3.11 and rerun this script.'
}

function Invoke-PythonCommand {
    param(
        [string[]]$BaseCommand,
        [string[]]$Args
    )

    $exe = $BaseCommand[0]
    $baseArgs = @()
    if ($BaseCommand.Length -gt 1) {
        $baseArgs = $BaseCommand[1..($BaseCommand.Length - 1)]
    }

    return & $exe @baseArgs @Args
}

function Test-CompatibleVersion {
    param([string]$VersionString)

    # Supports Python 3.10, 3.11, or 3.12.
    return $VersionString -match '^Python 3\.(10|11|12)\.'
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TrackGuard AI - Complete Setup (.venv)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = Get-CompatiblePythonCommand
$pythonVersion = (Invoke-PythonCommand -BaseCommand $pythonCmd -Args @('--version') 2>&1).Trim()

if (!(Test-CompatibleVersion -VersionString $pythonVersion)) {
    Write-Host "✗ Incompatible Python version: $pythonVersion" -ForegroundColor Red
    Write-Host "Please use Python 3.10, 3.11, or 3.12 for this project." -ForegroundColor Yellow
    exit 1
}

Write-Host "Using base interpreter: $pythonVersion" -ForegroundColor Green

# Create or validate .venv
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    Invoke-PythonCommand -BaseCommand $pythonCmd -Args @('-m', 'venv', '.venv') | Out-Null
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

$venvVersion = (& $venvPython --version 2>&1).Trim()
if (!(Test-CompatibleVersion -VersionString $venvVersion)) {
    Write-Host "✗ Current .venv uses an incompatible version: $venvVersion" -ForegroundColor Red
    Write-Host "Delete .venv and rerun this script so it can be recreated with Python 3.10/3.11." -ForegroundColor Yellow
    Write-Host "PowerShell command: Remove-Item -Recurse -Force .venv" -ForegroundColor Gray
    exit 1
}

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
& $venvPython -m pip install -r requirements.txt
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
