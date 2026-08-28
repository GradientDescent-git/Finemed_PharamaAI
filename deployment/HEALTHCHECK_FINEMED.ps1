# ============================================================
# FINEMED PHARMAAI - HEALTH CHECK SCRIPT
# ============================================================

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.prod.yml"

$HealthUrl = "http://localhost:8000/health"

$FailedChecks = 0

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "              FINEMED AI SYSTEM HEALTH CHECK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# CHECK 1 - Docker
# ------------------------------------------------------------

Write-Host "[CHECK 1] Docker Engine" -ForegroundColor Yellow

try {

    docker version *> $null

    if ($LASTEXITCODE -eq 0) {

        Write-Host "PASS - Docker Engine is available." -ForegroundColor Green

    }
    else {

        Write-Host "FAIL - Docker Engine is unavailable." -ForegroundColor Red
        $FailedChecks++

    }

}
catch {

    Write-Host "FAIL - Docker Desktop is not running." -ForegroundColor Red
    $FailedChecks++

}

# ------------------------------------------------------------
# CHECK 2 - Container
# ------------------------------------------------------------

Write-Host ""
Write-Host "[CHECK 2] FinemedAI Container" -ForegroundColor Yellow

$ContainerStatus = docker ps `
    --filter "name=finemed-pharmaai" `
    --format "{{.Status}}"

if ($ContainerStatus) {

    Write-Host "PASS - Container is running." -ForegroundColor Green
    Write-Host "Status: $ContainerStatus"

}
else {

    Write-Host "FAIL - FinemedAI container is not running." -ForegroundColor Red
    $FailedChecks++

}

# ------------------------------------------------------------
# CHECK 3 - Docker Health
# ------------------------------------------------------------

Write-Host ""
Write-Host "[CHECK 3] Container Health" -ForegroundColor Yellow

$HealthStatus = docker inspect `
    --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}" `
    finemed-pharmaai 2>$null

if ($HealthStatus -eq "healthy") {

    Write-Host "PASS - Container health status is HEALTHY." -ForegroundColor Green

}
elseif ($HealthStatus -eq "starting") {

    Write-Host "WARNING - Container health check is still starting." -ForegroundColor Yellow
    $FailedChecks++

}
else {

    Write-Host "FAIL - Container health status: $HealthStatus" -ForegroundColor Red
    $FailedChecks++

}

# ------------------------------------------------------------
# CHECK 4 - API Health
# ------------------------------------------------------------

Write-Host ""
Write-Host "[CHECK 4] FinemedAI API" -ForegroundColor Yellow

try {

    $Response = Invoke-WebRequest `
        -UseBasicParsing `
        -Uri $HealthUrl `
        -TimeoutSec 10

    if ($Response.StatusCode -eq 200) {

        $Health = $Response.Content | ConvertFrom-Json

        if ($Health.status -eq "ok") {

            Write-Host "PASS - API health endpoint returned OK." -ForegroundColor Green

        }
        else {

            Write-Host "WARNING - API returned an unexpected health response." -ForegroundColor Yellow
            $FailedChecks++

        }

        Write-Host ""
        Write-Host "Forecast Store Loaded: $($Health.forecast_store_loaded)"
        Write-Host "Medicines Available: $($Health.medicines_available)"
        Write-Host "Conversation Service: $($Health.conversation_service_available)"
        Write-Host "Chat Available: $($Health.chat_available)"

    }
    else {

        Write-Host "FAIL - API returned HTTP $($Response.StatusCode)." -ForegroundColor Red
        $FailedChecks++

    }

}
catch {

    Write-Host "FAIL - Unable to reach $HealthUrl" -ForegroundColor Red
    $FailedChecks++

}

# ------------------------------------------------------------
# Final Result
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan

if ($FailedChecks -eq 0) {

    Write-Host "       RESULT: FINEMED AI IS HEALTHY" -ForegroundColor Green

}
else {

    Write-Host "       RESULT: $FailedChecks HEALTH CHECK(S) FAILED" -ForegroundColor Red

    Write-Host ""
    Write-Host "Recent FinemedAI logs:" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $ProjectRoot

    try {

        docker compose -f $ComposeFile logs --tail 30

    }
    finally {

        Pop-Location

    }

}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to close this window"