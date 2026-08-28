# ============================================================
# FINEMED PHARMAAI - BACKUP SCRIPT
# ============================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$DataDirectory = Join-Path $ProjectRoot "data"
$EnvironmentFile = Join-Path $ProjectRoot ".env.production"

$BackupDirectory = Join-Path $PSScriptRoot "backups"

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$BackupName = "finemed_backup_$Timestamp.zip"

$BackupFile = Join-Path $BackupDirectory $BackupName

$TemporaryDirectory = Join-Path `
    $env:TEMP `
    "FinemedBackup_$Timestamp"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                 FINEMED AI BACKUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# Verify project data
# ------------------------------------------------------------

Write-Host "[1/5] Checking FinemedAI data..." -ForegroundColor Yellow

if (-not (Test-Path $DataDirectory)) {

    Write-Host ""
    Write-Host "ERROR: Data directory not found." -ForegroundColor Red
    Write-Host "Expected: $DataDirectory" -ForegroundColor Yellow

    Read-Host "Press Enter to close this window"
    exit 1

}

Write-Host "Data directory found." -ForegroundColor Green

# ------------------------------------------------------------
# Create backup directory
# ------------------------------------------------------------

Write-Host ""
Write-Host "[2/5] Preparing backup directory..." -ForegroundColor Yellow

if (-not (Test-Path $BackupDirectory)) {

    New-Item `
        -ItemType Directory `
        -Path $BackupDirectory `
        -Force | Out-Null

}

Write-Host "Backup directory ready." -ForegroundColor Green

# ------------------------------------------------------------
# Prepare temporary backup
# ------------------------------------------------------------

Write-Host ""
Write-Host "[3/5] Collecting production data..." -ForegroundColor Yellow

New-Item `
    -ItemType Directory `
    -Path $TemporaryDirectory `
    -Force | Out-Null

$BackupDataDirectory = Join-Path `
    $TemporaryDirectory `
    "data"

Copy-Item `
    -Path $DataDirectory `
    -Destination $BackupDataDirectory `
    -Recurse `
    -Force

if (Test-Path $EnvironmentFile) {

    Copy-Item `
        -Path $EnvironmentFile `
        -Destination $TemporaryDirectory `
        -Force

}

Copy-Item `
    -Path (Join-Path $ProjectRoot "docker-compose.prod.yml") `
    -Destination $TemporaryDirectory `
    -Force

Write-Host "Production data collected." -ForegroundColor Green

# ------------------------------------------------------------
# Create ZIP archive
# ------------------------------------------------------------

Write-Host ""
Write-Host "[4/5] Creating backup archive..." -ForegroundColor Yellow

Compress-Archive `
    -Path "$TemporaryDirectory\*" `
    -DestinationPath $BackupFile `
    -CompressionLevel Optimal `
    -Force

# ------------------------------------------------------------
# Verify backup
# ------------------------------------------------------------

Write-Host ""
Write-Host "[5/5] Verifying backup..." -ForegroundColor Yellow

if (-not (Test-Path $BackupFile)) {

    throw "Backup archive was not created."

}

$BackupSize = (
    Get-Item $BackupFile
).Length

if ($BackupSize -le 0) {

    throw "Backup archive is empty."

}

$BackupSizeMB = [math]::Round(
    $BackupSize / 1MB,
    2
)

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

Remove-Item `
    -Path $TemporaryDirectory `
    -Recurse `
    -Force

# ------------------------------------------------------------
# Retention
# ------------------------------------------------------------

$RetentionCount = 10

$ExistingBackups = Get-ChildItem `
    -Path $BackupDirectory `
    -Filter "finemed_backup_*.zip" |
    Sort-Object LastWriteTime -Descending

if ($ExistingBackups.Count -gt $RetentionCount) {

    $OldBackups = $ExistingBackups |
        Select-Object -Skip $RetentionCount

    foreach ($OldBackup in $OldBackups) {

        Remove-Item `
            -Path $OldBackup.FullName `
            -Force

    }

}

# ------------------------------------------------------------
# Success
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "           FINEMED AI BACKUP SUCCESSFUL" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Backup File:" -ForegroundColor Cyan
Write-Host $BackupFile

Write-Host ""
Write-Host "Backup Size: $BackupSizeMB MB"

Write-Host ""

Read-Host "Press Enter to close this window"