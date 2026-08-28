$ErrorActionPreference = "Stop"

Clear-Host

Write-Host ""
Write-Host "============================================================"
Write-Host "              FINEMED AI INSTALLATION CHECK"
Write-Host "============================================================"
Write-Host ""

$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[1/7] Checking project directory..."

if (-not (Test-Path $ProjectRoot)) {
    Write-Host "ERROR - FinemedAI project directory was not found."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "PASS - Project directory found."
Write-Host "Location: $ProjectRoot"
Write-Host ""

Write-Host "[2/7] Checking Docker..."

try {
    docker version | Out-Null
}
catch {
    Write-Host ""
    Write-Host "ERROR - Docker Desktop is not available."
    Write-Host ""
    Write-Host "Please:"
    Write-Host "1. Install Docker Desktop."
    Write-Host "2. Start Docker Desktop."
    Write-Host "3. Run this installer again."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "PASS - Docker Engine is available."
Write-Host ""

Write-Host "[3/7] Checking production configuration..."

$EnvFile = Join-Path $ProjectRoot ".env.production"

if (-not (Test-Path $EnvFile)) {

    Write-Host "WARNING - .env.production was not found."

    $ExampleEnv = Join-Path $ProjectRoot ".env.production.example"

    if (Test-Path $ExampleEnv) {

        Copy-Item $ExampleEnv $EnvFile

        Write-Host "Created .env.production from example template."
        Write-Host ""
        Write-Host "IMPORTANT:"
        Write-Host "Change ADMIN_TOKEN and CLIENT_API_KEY before production use."
    }
    else {

        Write-Host "ERROR - Production environment template is missing."
        Read-Host "Press Enter to exit"
        exit 1
    }
}
else {

    Write-Host "PASS - Production configuration found."
}

Write-Host ""

Write-Host "[4/7] Checking production data..."

$DataDirectory = Join-Path $ProjectRoot "data"

if (-not (Test-Path $DataDirectory)) {

    Write-Host "ERROR - Data directory was not found."
    Write-Host "FinemedAI requires the production data directory."

    Read-Host "Press Enter to exit"
    exit 1
}

$ForecastFile = Join-Path `
    $ProjectRoot `
    "data\05_gold\demand_forecasting\production_forecasts\latest.parquet"

if (-not (Test-Path $ForecastFile)) {

    Write-Host "WARNING - Production forecast file was not found."
    Write-Host "Expected location:"
    Write-Host $ForecastFile
}
else {

    Write-Host "PASS - Production forecast data found."
}

Write-Host ""

Write-Host "[5/7] Checking deployment files..."

$ComposeFile = Join-Path $ProjectRoot "docker-compose.prod.yml"
$StartFile = Join-Path $ProjectRoot "deployment\START_FINEMED.ps1"
$StopFile = Join-Path $ProjectRoot "deployment\STOP_FINEMED.ps1"
$StatusFile = Join-Path $ProjectRoot "deployment\STATUS_FINEMED.ps1"
$HealthFile = Join-Path $ProjectRoot "deployment\HEALTHCHECK_FINEMED.ps1"
$BackupFile = Join-Path $ProjectRoot "deployment\BACKUP_FINEMED.ps1"

$RequiredFiles = @(
    $ComposeFile,
    $StartFile,
    $StopFile,
    $StatusFile,
    $HealthFile,
    $BackupFile
)

foreach ($File in $RequiredFiles) {

    if (-not (Test-Path $File)) {

        Write-Host "ERROR - Required deployment file is missing:"
        Write-Host $File

        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "PASS - All required deployment files found."
Write-Host ""

Write-Host "[6/7] Validating Docker Compose configuration..."

Push-Location $ProjectRoot

try {

    docker compose -f docker-compose.prod.yml config | Out-Null

    Write-Host "PASS - Docker Compose configuration is valid."
}
catch {

    Write-Host "ERROR - Docker Compose configuration validation failed."

    Pop-Location

    Read-Host "Press Enter to exit"
    exit 1
}

Pop-Location

Write-Host ""

Write-Host "[7/7] Installation check complete."

Write-Host ""
Write-Host "============================================================"
Write-Host "        FINEMED AI INSTALLATION READY"
Write-Host "============================================================"
Write-Host ""

Write-Host "Next step:"
Write-Host ""

Write-Host "Run:"
Write-Host ""

Write-Host "powershell -ExecutionPolicy Bypass -File .\deployment\START_FINEMED.ps1"
Write-Host ""

Write-Host "Useful commands:"
Write-Host ""

Write-Host "START:"
Write-Host ".\deployment\START_FINEMED.ps1"
Write-Host ""

Write-Host "STATUS:"
Write-Host ".\deployment\STATUS_FINEMED.ps1"
Write-Host ""

Write-Host "HEALTH CHECK:"
Write-Host ".\deployment\HEALTHCHECK_FINEMED.ps1"
Write-Host ""

Write-Host "BACKUP:"
Write-Host ".\deployment\BACKUP_FINEMED.ps1"
Write-Host ""

Write-Host "STOP:"
Write-Host ".\deployment\STOP_FINEMED.ps1"
Write-Host ""

Read-Host "Press Enter to exit"