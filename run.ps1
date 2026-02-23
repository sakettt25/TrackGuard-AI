# TrackGuard AI Run Script
# This script runs the TrackGuard AI project using .venv

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TrackGuard AI - Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .venv exists
$venvPython = ".\.venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    Write-Host "✗ Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run: .\install_venv.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Check if setup has been run
if (!(Test-Path "deep_sort1/deep_sort.py")) {
    Write-Host "✗ Project not set up properly" -ForegroundColor Red
    Write-Host "Please run: .\install_venv.ps1 first" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using Python: $venvPython" -ForegroundColor Green
Write-Host "Starting TrackGuard AI..." -ForegroundColor Yellow
Write-Host ""

try {
    & $venvPython main_production.py
} catch {
    Write-Host ""
    Write-Host "✗ Error running TrackGuard AI" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TrackGuard AI - Stopped" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
