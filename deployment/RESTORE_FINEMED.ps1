# ============================================================
# FINEMED PHARMAAI - SAFE RESTORE SCRIPT
# ============================================================

param (
    [string]$BackupZipPath = "",
    [string]$TargetDirectory = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackupDirectory = Join-Path $PSScriptRoot "backups"

if ([string]::IsNullOrWhiteSpace($TargetDirectory)) {
    $TargetDirectory = $ProjectRoot
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                FINEMED AI SAFE RESTORE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Resolve backup archive
if ([string]::IsNullOrWhiteSpace($BackupZipPath)) {
    $LatestBackup = Get-ChildItem -Path $BackupDirectory -Filter "finemed_backup_*.zip" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($null -eq $LatestBackup) {
        Write-Host "ERROR: No backup archives found in $BackupDirectory" -ForegroundColor Red
        exit 1
    }
    $BackupZipPath = $LatestBackup.FullName
}

if (-not (Test-Path $BackupZipPath)) {
    Write-Host "ERROR: Specified backup file does not exist: $BackupZipPath" -ForegroundColor Red
    exit 1
}

Write-Host "[1/4] Selected Backup: $BackupZipPath" -ForegroundColor Yellow
Write-Host "[2/4] Target Restore Location: $TargetDirectory" -ForegroundColor Yellow

# 2. Extract backup to target directory safely
Write-Host "[3/4] Extracting backup archive..." -ForegroundColor Yellow

Expand-Archive -Path $BackupZipPath -DestinationPath $TargetDirectory -Force

# 3. Verify restored artifacts
Write-Host "[4/4] Verifying restored artifacts..." -ForegroundColor Yellow

$RestoredDataDir = Join-Path $TargetDirectory "data"
if (-not (Test-Path $RestoredDataDir)) {
    throw "Restore verification failed: 'data' directory not found in restored output."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          FINEMED AI RESTORE VERIFIED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Restored From: $BackupZipPath"
Write-Host "Restored To:   $TargetDirectory"
Write-Host ""
