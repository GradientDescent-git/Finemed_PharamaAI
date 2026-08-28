# ============================================================
# FINEMED PHARMAAI - STOP SCRIPT
# ============================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.prod.yml"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                 FINEMED AI SHUTDOWN" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# Verify Docker
# ------------------------------------------------------------

Write-Host "[1/3] Checking Docker..." -ForegroundColor Yellow

try {
    docker version *> $null
}
catch {
    Write-Host ""
    Write-Host "ERROR: Docker Desktop is not running." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Docker is available." -ForegroundColor Green

# ------------------------------------------------------------
# Verify compose file
# ------------------------------------------------------------

if (-not (Test-Path $ComposeFile)) {
    Write-Host ""
    Write-Host "ERROR: Production compose file was not found." -ForegroundColor Red
    Write-Host "Expected: $ComposeFile" -ForegroundColor Yellow
    exit 1
}

# ------------------------------------------------------------
# Stop FinemedAI
# ------------------------------------------------------------

Write-Host ""
Write-Host "[2/3] Stopping FinemedAI..." -ForegroundColor Yellow

Push-Location $ProjectRoot

try {
    docker compose -f $ComposeFile down

    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose returned exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "[3/3] Verifying shutdown..." -ForegroundColor Yellow

$RunningContainer = docker ps `
    --filter "name=finemed-pharmaai" `
    --format "{{.Names}}"

if ($RunningContainer) {

    Write-Host ""
    Write-Host "WARNING: FinemedAI may still be running." -ForegroundColor Yellow
    Write-Host "Container detected: $RunningContainer" -ForegroundColor Yellow

}
else {

    Write-Host "FinemedAI stopped successfully." -ForegroundColor Green

}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "               FINEMED AI IS STOPPED" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to close this window"