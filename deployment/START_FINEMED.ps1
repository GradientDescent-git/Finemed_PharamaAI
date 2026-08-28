# ============================================================
# FinemedAI - Client Startup Script
# ============================================================

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------
# Resolve project root
# ------------------------------------------------------------

$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                 FINEMED AI STARTUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# Check Docker
# ------------------------------------------------------------

Write-Host "[1/4] Checking Docker..." -ForegroundColor Yellow

try {

    docker info *> $null

    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not available."
    }

}
catch {

    Write-Host ""
    Write-Host "ERROR: Docker Desktop is not running." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start Docker Desktop and wait until it is fully running." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"

    exit 1
}

Write-Host "Docker is available." -ForegroundColor Green

# ------------------------------------------------------------
# Move to project root
# ------------------------------------------------------------

Set-Location $ProjectRoot

# ------------------------------------------------------------
# Start FinemedAI
# ------------------------------------------------------------

Write-Host ""
Write-Host "[2/4] Starting FinemedAI..." -ForegroundColor Yellow

docker compose `
    -f docker-compose.prod.yml `
    up `
    -d `
    --build

if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: FinemedAI failed to start." -ForegroundColor Red
    Write-Host ""

    docker compose `
        -f docker-compose.prod.yml `
        logs `
        --tail 50

    Read-Host "Press Enter to exit"

    exit 1
}

Write-Host "FinemedAI container started." -ForegroundColor Green

# ------------------------------------------------------------
# Wait for application health
# ------------------------------------------------------------

Write-Host ""
Write-Host "[3/4] Waiting for FinemedAI to become ready..." -ForegroundColor Yellow

$MaxAttempts = 30
$Attempt = 0
$Ready = $false

while (
    $Attempt -lt $MaxAttempts `
    -and `
    -not $Ready
) {

    Start-Sleep -Seconds 2

    $Attempt++

    try {

        $Response = Invoke-WebRequest `
            -UseBasicParsing `
            -Uri "http://localhost:8000/health" `
            -TimeoutSec 5

        if ($Response.StatusCode -eq 200) {

            $Ready = $true

        }

    }
    catch {

        Write-Host "Waiting for FinemedAI... ($Attempt/$MaxAttempts)"

    }

}

# ------------------------------------------------------------
# Handle startup failure
# ------------------------------------------------------------

if (-not $Ready) {

    Write-Host ""
    Write-Host "ERROR: FinemedAI did not become ready in time." -ForegroundColor Red
    Write-Host ""

    Write-Host "Container status:" -ForegroundColor Yellow

    docker compose `
        -f docker-compose.prod.yml `
        ps

    Write-Host ""

    Write-Host "Recent application logs:" -ForegroundColor Yellow

    docker compose `
        -f docker-compose.prod.yml `
        logs `
        --tail 50

    Write-Host ""

    Read-Host "Press Enter to exit"

    exit 1

}

# ------------------------------------------------------------
# Success
# ------------------------------------------------------------

Write-Host ""
Write-Host "[4/4] FinemedAI is ready." -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "       FINEMED AI IS RUNNING SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Opening FinemedAI in your browser..." -ForegroundColor Cyan
Write-Host ""

Start-Process "http://localhost:8000"

Write-Host "FinemedAI is now running."
Write-Host ""
Write-Host "Application URL:" -ForegroundColor Cyan
Write-Host "http://localhost:8000"
Write-Host ""

Read-Host "Press Enter to close this window"