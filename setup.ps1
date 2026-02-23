# TrackGuard AI Setup Script
# This script sets up the TrackGuard AI project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TrackGuard AI - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "✓ Python is installed: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
$directories = @("data", "weights", "output", "deep_sort1/deep/checkpoint")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "✓ Created directory: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ Directory already exists: $dir" -ForegroundColor Green
    }
}

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
try {
    & python -m pip install --upgrade pip
    & python -m pip install -r requirements.txt
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Check if sample video exists
Write-Host ""
Write-Host "Checking for video file..." -ForegroundColor Yellow
if (!(Test-Path "data/test2.mp4")) {
    Write-Host "⚠ No video file found at data/test2.mp4" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "1. Place your video file at: data/test2.mp4" -ForegroundColor White
    Write-Host "2. Use webcam (edit CONFIG in main_production.py)" -ForegroundColor White
    Write-Host "3. The script will create a sample video when run" -ForegroundColor White
} else {
    Write-Host "✓ Video file found" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. (Optional) Place your video at: data/test2.mp4" -ForegroundColor White
Write-Host "2. (Optional) Place fire detection model at: weights/best.pt" -ForegroundColor White
Write-Host "3. Run the project with: .\run.ps1" -ForegroundColor White
Write-Host ""
