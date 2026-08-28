# ============================================================
# FINEMED PHARMAAI - STATUS SCRIPT
# ============================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.prod.yml"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                 FINEMED AI STATUS" -ForegroundColor Cyan
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
    Write-Host "Docker Desktop is not available." -ForegroundColor Red
    Write-Host ""

    Read-Host "Press Enter to close this window"
    exit 1
}

Write-Host "Docker is available." -ForegroundColor Green

# ------------------------------------------------------------
# Container status
# ------------------------------------------------------------

Write-Host ""
Write-Host "[2/3] Checking FinemedAI container..." -ForegroundColor Yellow
Write-Host ""

Push-Location $ProjectRoot

try {

    docker compose -f $ComposeFile ps

}
finally {

    Pop-Location

}

# ------------------------------------------------------------
# Application health
# ------------------------------------------------------------

Write-Host ""
Write-Host "[3/3] Checking application health..." -ForegroundColor Yellow

try {

    $Response = Invoke-WebRequest `
        -UseBasicParsing `
        -Uri "http://localhost:8000/health" `
        -TimeoutSec 10

    if ($Response.StatusCode -eq 200) {

        $Health = $Response.Content | ConvertFrom-Json

        Write-Host ""
        Write-Host "Application Status: HEALTHY" -ForegroundColor Green
        Write-Host ""
        Write-Host "Health Response:" -ForegroundColor Cyan

        $Health | ConvertTo-Json -Depth 5

    }
    else {

        Write-Host ""
        Write-Host "Application returned HTTP $($Response.StatusCode)" -ForegroundColor Yellow

    }

}
catch {

    Write-Host ""
    Write-Host "Application Status: UNAVAILABLE" -ForegroundColor Red
    Write-Host ""
    Write-Host "FinemedAI is not responding at http://localhost:8000" -ForegroundColor Yellow

}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                 STATUS CHECK COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to close this window"