# ============================================================
# FINEMED AI - PRODUCTION ACCEPTANCE TEST
# ============================================================
#
# Tests the deployed FinemedAI production API contract.
#
# Coverage:
#   1. Docker/container health
#   2. API health and readiness
#   3. Version endpoint
#   4. Authentication rejection
#   5. Authenticated forecast endpoints
#   6. Forecast summary retrieval
#   7. Conversational AI
#   8. Admin endpoint authentication
#   9. Pipeline/operations visibility
#
# ============================================================

$ErrorActionPreference = "Stop"

# ============================================================
# CONFIGURATION
# ============================================================

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "docker-compose.prod.yml"
$EnvFile = Join-Path $ProjectRoot ".env.production"

$BaseUrl = "http://localhost:8000"

$Passed = 0
$Failed = 0
$Warnings = 0

$Results = @()


# ============================================================
# DISPLAY HELPERS
# ============================================================

function Write-Section {
    param (
        [string]$Title
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " $Title"
    Write-Host "============================================================"
}


function Add-TestResult {
    param (
        [string]$Name,
        [bool]$Success,
        [string]$Message
    )

    $script:Results += [PSCustomObject]@{
        Test    = $Name
        Status  = if ($Success) { "PASS" } else { "FAIL" }
        Message = $Message
    }

    if ($Success) {
        $script:Passed++
        Write-Host "PASS - $Name" -ForegroundColor Green

        if ($Message) {
            Write-Host "       $Message"
        }
    }
    else {
        $script:Failed++
        Write-Host "FAIL - $Name" -ForegroundColor Red

        if ($Message) {
            Write-Host "       $Message" -ForegroundColor Red
        }
    }
}


function Add-WarningResult {
    param (
        [string]$Message
    )

    $script:Warnings++

    Write-Host "WARNING - $Message" -ForegroundColor Yellow
}


# ============================================================
# ENVIRONMENT LOADING
# ============================================================

function Get-EnvValue {
    param (
        [string]$Name,
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    $line = Get-Content $Path |
        Where-Object {
            $_ -match "^\s*$Name\s*="
        } |
        Select-Object -First 1

    if (-not $line) {
        return $null
    }

    return ($line -split "=", 2)[1].Trim()
}


# ============================================================
# HTTP REQUEST HELPER
# ============================================================

function Invoke-FinemedRequest {
    param (
        [string]$Method,
        [string]$Uri,
        [hashtable]$Headers = @{},
        [object]$Body = $null
    )

    try {

        $params = @{
            Method          = $Method
            Uri             = $Uri
            Headers         = $Headers
            UseBasicParsing = $true
            TimeoutSec      = 30
        }

        if ($null -ne $Body) {

            if ($Body -is [string]) {

                $params["Body"] = $Body
            }
            else {

                $params["Body"] = (
                    $Body |
                    ConvertTo-Json -Depth 10
                )

                $params["ContentType"] = "application/json"
            }
        }

        $response = Invoke-WebRequest @params

        return [PSCustomObject]@{
            Success    = $true
            StatusCode = $response.StatusCode
            Content    = $response.Content
            Error      = $null
        }
    }
    catch {

        $statusCode = $null
        $content = $null

        if ($_.Exception.Response) {

            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }
            catch {
                $statusCode = $null
            }

            try {

                $reader = New-Object System.IO.StreamReader(
                    $_.Exception.Response.GetResponseStream()
                )

                $content = $reader.ReadToEnd()
            }
            catch {
                $content = $_.Exception.Message
            }
        }

        return [PSCustomObject]@{
            Success    = $false
            StatusCode = $statusCode
            Content    = $content
            Error      = $_.Exception.Message
        }
    }
}


# ============================================================
# START
# ============================================================

Clear-Host

Write-Host ""
Write-Host "============================================================"
Write-Host "       FINEMED AI PRODUCTION ACCEPTANCE TEST"
Write-Host "============================================================"

Write-Host ""
Write-Host "Project:"
Write-Host $ProjectRoot

Write-Host ""
Write-Host "Target API:"
Write-Host $BaseUrl


# ============================================================
# CHECK 1 - PROJECT CONFIGURATION
# ============================================================

Write-Section "CHECK 1 - PRODUCTION CONFIGURATION"

if (Test-Path $ComposeFile) {

    Add-TestResult `
        -Name "Production Docker Compose File" `
        -Success $true `
        -Message "docker-compose.prod.yml found."
}
else {

    Add-TestResult `
        -Name "Production Docker Compose File" `
        -Success $false `
        -Message "docker-compose.prod.yml not found."
}


if (Test-Path $EnvFile) {

    Add-TestResult `
        -Name "Production Environment File" `
        -Success $true `
        -Message ".env.production found."
}
else {

    Add-TestResult `
        -Name "Production Environment File" `
        -Success $false `
        -Message ".env.production not found."
}


$ClientApiKey = Get-EnvValue `
    -Name "CLIENT_API_KEY" `
    -Path $EnvFile

$AdminToken = Get-EnvValue `
    -Name "ADMIN_TOKEN" `
    -Path $EnvFile


if ([string]::IsNullOrWhiteSpace($ClientApiKey)) {

    Add-TestResult `
        -Name "Client API Key Configuration" `
        -Success $false `
        -Message "CLIENT_API_KEY is missing or empty."
}
else {

    Add-TestResult `
        -Name "Client API Key Configuration" `
        -Success $true `
        -Message "CLIENT_API_KEY is configured."
}


if ([string]::IsNullOrWhiteSpace($AdminToken)) {

    Add-TestResult `
        -Name "Admin Token Configuration" `
        -Success $false `
        -Message "ADMIN_TOKEN is missing or empty."
}
else {

    Add-TestResult `
        -Name "Admin Token Configuration" `
        -Success $true `
        -Message "ADMIN_TOKEN is configured."
}


# ============================================================
# CHECK 2 - DOCKER
# ============================================================

Write-Section "CHECK 2 - DOCKER CONTAINER"

try {

    docker info *> $null

    if ($LASTEXITCODE -eq 0) {

        Add-TestResult `
            -Name "Docker Engine" `
            -Success $true `
            -Message "Docker Engine is available."
    }
    else {

        Add-TestResult `
            -Name "Docker Engine" `
            -Success $false `
            -Message "Docker Engine is unavailable."
    }
}
catch {

    Add-TestResult `
        -Name "Docker Engine" `
        -Success $false `
        -Message $_.Exception.Message
}


$ContainerStatus = docker compose `
    -f $ComposeFile `
    ps `
    --format json `
    2>$null


if ($ContainerStatus) {

    try {

        $Containers = $ContainerStatus |
            ConvertFrom-Json

        $Container = $Containers |
            Where-Object {
                $_.Name -eq "finemed-pharmaai"
            } |
            Select-Object -First 1


        if ($Container) {

            if ($Container.State -eq "running") {

                Add-TestResult `
                    -Name "FinemedAI Container Running" `
                    -Success $true `
                    -Message "Container state: $($Container.State)"
            }
            else {

                Add-TestResult `
                    -Name "FinemedAI Container Running" `
                    -Success $false `
                    -Message "Container state: $($Container.State)"
            }


            if ($Container.Health -eq "healthy") {

                Add-TestResult `
                    -Name "Container Health" `
                    -Success $true `
                    -Message "Container health: healthy"
            }
            else {

                Add-WarningResult `
                    -Message "Container health is '$($Container.Health)'."
            }
        }
        else {

            Add-TestResult `
                -Name "FinemedAI Container Running" `
                -Success $false `
                -Message "finemed-pharmaai container was not found."
        }
    }
    catch {

        Add-WarningResult `
            -Message "Could not parse Docker container status."
    }
}
else {

    Add-TestResult `
        -Name "FinemedAI Container Running" `
        -Success $false `
        -Message "No container status returned."
}


# ============================================================
# CHECK 3 - HEALTH
# ============================================================

Write-Section "CHECK 3 - API HEALTH"

$HealthResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/health"


if (
    $HealthResponse.Success -and
    $HealthResponse.StatusCode -eq 200
) {

    try {

        $HealthData = $HealthResponse.Content |
            ConvertFrom-Json


        $HealthPassed = (
            $HealthData.status -eq "ok" -and
            $HealthData.forecast_store_loaded -eq $true -and
            $HealthData.medicines_available -gt 0 -and
            $HealthData.conversation_service_available -eq $true -and
            $HealthData.chat_available -eq $true
        )


        Add-TestResult `
            -Name "API Health Endpoint" `
            -Success $HealthPassed `
            -Message (
                "Medicines: " +
                $HealthData.medicines_available +
                " | Forecast Store: " +
                $HealthData.forecast_store_loaded +
                " | Chat: " +
                $HealthData.chat_available
            )
    }
    catch {

        Add-TestResult `
            -Name "API Health Endpoint" `
            -Success $false `
            -Message "Health response could not be parsed."
    }
}
else {

    Add-TestResult `
        -Name "API Health Endpoint" `
        -Success $false `
        -Message $HealthResponse.Error
}


# ============================================================
# CHECK 4 - READINESS
# ============================================================

Write-Section "CHECK 4 - API READINESS"

$ReadyResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/ready"


if (
    $ReadyResponse.Success -and
    $ReadyResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "API Readiness Endpoint" `
        -Success $true `
        -Message "Application reports ready."
}
else {

    Add-TestResult `
        -Name "API Readiness Endpoint" `
        -Success $false `
        -Message $ReadyResponse.Error
}


# ============================================================
# CHECK 5 - VERSION
# ============================================================

Write-Section "CHECK 5 - VERSION"

$VersionResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/version"


if (
    $VersionResponse.Success -and
    $VersionResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "Version Endpoint" `
        -Success $true `
        -Message "Version information returned."
}
else {

    Add-TestResult `
        -Name "Version Endpoint" `
        -Success $false `
        -Message $VersionResponse.Error
}


# ============================================================
# CHECK 6 - AUTHENTICATION
# ============================================================

Write-Section "CHECK 6 - SECURITY AND AUTHENTICATION"

$UnauthorizedForecast = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/forecast/top?n=1"


if (
    $UnauthorizedForecast.StatusCode -eq 401 -or
    $UnauthorizedForecast.StatusCode -eq 403
) {

    Add-TestResult `
        -Name "Forecast Endpoint Rejects Missing API Key" `
        -Success $true `
        -Message "Unauthenticated request rejected with HTTP $($UnauthorizedForecast.StatusCode)."
}
else {

    Add-TestResult `
        -Name "Forecast Endpoint Rejects Missing API Key" `
        -Success $false `
        -Message "Expected 401 or 403 but received HTTP $($UnauthorizedForecast.StatusCode)."
}


$InvalidApiKeyResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/forecast/top?n=1" `
    -Headers @{
        "X-API-Key" = "INVALID_FINEMED_TEST_KEY"
    }


if (
    $InvalidApiKeyResponse.StatusCode -eq 401 -or
    $InvalidApiKeyResponse.StatusCode -eq 403
) {

    Add-TestResult `
        -Name "Forecast Endpoint Rejects Invalid API Key" `
        -Success $true `
        -Message "Invalid API key rejected with HTTP $($InvalidApiKeyResponse.StatusCode)."
}
else {

    Add-TestResult `
        -Name "Forecast Endpoint Rejects Invalid API Key" `
        -Success $false `
        -Message "Expected 401 or 403 but received HTTP $($InvalidApiKeyResponse.StatusCode)."
}


$InvalidAdminResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/admin/pipeline-status" `
    -Headers @{
        "X-Admin-Token" = "INVALID_FINEMED_ADMIN_TOKEN"
    }


if (
    $InvalidAdminResponse.StatusCode -eq 401 -or
    $InvalidAdminResponse.StatusCode -eq 403
) {

    Add-TestResult `
        -Name "Admin Endpoint Rejects Invalid Token" `
        -Success $true `
        -Message "Invalid admin token rejected with HTTP $($InvalidAdminResponse.StatusCode)."
}
else {

    Add-TestResult `
        -Name "Admin Endpoint Rejects Invalid Token" `
        -Success $false `
        -Message "Expected 401 or 403 but received HTTP $($InvalidAdminResponse.StatusCode)."
}


# ============================================================
# CHECK 7 - FORECAST API
# ============================================================

Write-Section "CHECK 7 - FORECAST FUNCTIONALITY"

$ApiHeaders = @{
    "X-API-Key" = $ClientApiKey
}


$TopForecastResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/forecast/top?n=5" `
    -Headers $ApiHeaders


$SampleMedicineId = $null


if (
    $TopForecastResponse.Success -and
    $TopForecastResponse.StatusCode -eq 200
) {

    try {

        $TopForecastData = $TopForecastResponse.Content |
            ConvertFrom-Json


        $TopForecastItems = @()

        if ($TopForecastData -is [System.Array]) {

            $TopForecastItems = $TopForecastData
        }
        elseif ($TopForecastData.data) {

            $TopForecastItems = @($TopForecastData.data)
        }
        elseif ($TopForecastData.items) {

            $TopForecastItems = @($TopForecastData.items)
        }
        elseif ($TopForecastData.medicines) {

            $TopForecastItems = @($TopForecastData.medicines)
        }


        if ($TopForecastItems.Count -gt 0) {

            $SampleMedicine = $TopForecastItems[0]

            foreach (
                $PossibleProperty in @(
                    "medicine_id",
                    "id",
                    "medicine"
                )
            ) {

                if ($SampleMedicine.PSObject.Properties.Name -contains $PossibleProperty) {

                    $SampleMedicineId = [string]$SampleMedicine.$PossibleProperty

                    break
                }
            }


            Add-TestResult `
                -Name "Top Demand Forecast" `
                -Success $true `
                -Message "Returned $($TopForecastItems.Count) forecast records."
        }
        else {

            Add-TestResult `
                -Name "Top Demand Forecast" `
                -Success $false `
                -Message "Endpoint returned HTTP 200 but no forecast records were detected."
        }
    }
    catch {

        Add-TestResult `
            -Name "Top Demand Forecast" `
            -Success $false `
            -Message "Forecast response could not be parsed."
    }
}
else {

    Add-TestResult `
        -Name "Top Demand Forecast" `
        -Success $false `
        -Message $TopForecastResponse.Error
}


$UncertainResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/forecast/uncertain?n=5" `
    -Headers $ApiHeaders


if (
    $UncertainResponse.Success -and
    $UncertainResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "Forecast Uncertainty Endpoint" `
        -Success $true `
        -Message "Uncertainty data returned."
}
else {

    Add-TestResult `
        -Name "Forecast Uncertainty Endpoint" `
        -Success $false `
        -Message $UncertainResponse.Error
}


$TrendResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/forecast/trend?direction=increasing&n=5" `
    -Headers $ApiHeaders


if (
    $TrendResponse.Success -and
    $TrendResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "Forecast Trend Endpoint" `
        -Success $true `
        -Message "Trend query completed successfully."
}
else {

    Add-WarningResult `
        -Message "Trend query did not return HTTP 200. Direction labels may differ from 'increasing'."
}


# ============================================================
# CHECK 8 - MEDICINE FORECAST DETAIL
# ============================================================

Write-Section "CHECK 8 - MEDICINE FORECAST DETAIL"

if ([string]::IsNullOrWhiteSpace($SampleMedicineId)) {

    Add-WarningResult `
        -Message "Could not automatically extract a medicine ID from /forecast/top."
}
else {

    $EncodedMedicineId = [System.Uri]::EscapeDataString(
        $SampleMedicineId
    )


    $SummaryResponse = Invoke-FinemedRequest `
        -Method "GET" `
        -Uri "$BaseUrl/forecast/$EncodedMedicineId/summary" `
        -Headers $ApiHeaders


    if (
        $SummaryResponse.Success -and
        $SummaryResponse.StatusCode -eq 200
    ) {

        Add-TestResult `
            -Name "Medicine Forecast Summary" `
            -Success $true `
            -Message "Summary returned for medicine: $SampleMedicineId"
    }
    else {

        Add-TestResult `
            -Name "Medicine Forecast Summary" `
            -Success $false `
            -Message $SummaryResponse.Error
    }


    $ForecastDetailResponse = Invoke-FinemedRequest `
        -Method "GET" `
        -Uri "$BaseUrl/forecast/$EncodedMedicineId" `
        -Headers $ApiHeaders


    if (
        $ForecastDetailResponse.Success -and
        $ForecastDetailResponse.StatusCode -eq 200
    ) {

        Add-TestResult `
            -Name "Medicine Forecast Detail" `
            -Success $true `
            -Message "Forecast returned for medicine: $SampleMedicineId"
    }
    else {

        Add-TestResult `
            -Name "Medicine Forecast Detail" `
            -Success $false `
            -Message $ForecastDetailResponse.Error
    }
}


# ============================================================
# CHECK 9 - CHAT
# ============================================================

Write-Section "CHECK 9 - CONVERSATIONAL INTELLIGENCE"

$ChatResponse = Invoke-FinemedRequest `
    -Method "POST" `
    -Uri "$BaseUrl/chat" `
    -Headers $ApiHeaders `
    -Body @{
        question = "What medicines have the highest forecast demand?"
    }


if (
    $ChatResponse.Success -and
    $ChatResponse.StatusCode -eq 200
) {

    try {

        $ChatData = $ChatResponse.Content |
            ConvertFrom-Json


        $ChatPassed = (
            -not [string]::IsNullOrWhiteSpace(
                $ChatData.answer
            )
        )


        Add-TestResult `
            -Name "Forecast Conversation Service" `
            -Success $ChatPassed `
            -Message "Chat response generated successfully."
    }
    catch {

        Add-TestResult `
            -Name "Forecast Conversation Service" `
            -Success $false `
            -Message "Chat response could not be parsed."
    }
}
else {

    Add-TestResult `
        -Name "Forecast Conversation Service" `
        -Success $false `
        -Message $ChatResponse.Error
}


# ============================================================
# CHECK 10 - ADMIN API
# ============================================================

Write-Section "CHECK 10 - ADMIN AND OPERATIONS"

$AdminHeaders = @{
    "X-Admin-Token" = $AdminToken
}


$PipelineStatusResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/admin/pipeline-status" `
    -Headers $AdminHeaders


if (
    $PipelineStatusResponse.Success -and
    $PipelineStatusResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "Admin Pipeline Status" `
        -Success $true `
        -Message "Pipeline status is accessible to authorized administrator."
}
else {

    Add-TestResult `
        -Name "Admin Pipeline Status" `
        -Success $false `
        -Message $PipelineStatusResponse.Error
}


$PipelineLatestResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/pipeline/latest" `
    -Headers $AdminHeaders


if (
    $PipelineLatestResponse.Success -and
    $PipelineLatestResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "Latest Pipeline Metadata" `
        -Success $true `
        -Message "Latest pipeline metadata returned."
}
else {

    Add-WarningResult `
        -Message "Latest pipeline metadata endpoint did not return HTTP 200."
}


$OperationsResponse = Invoke-FinemedRequest `
    -Method "GET" `
    -Uri "$BaseUrl/operations/summary" `
    -Headers $AdminHeaders


if (
    $OperationsResponse.Success -and
    $OperationsResponse.StatusCode -eq 200
) {

    Add-TestResult `
        -Name "Operations Summary" `
        -Success $true `
        -Message "Operations summary returned."
}
else {

    Add-TestResult `
        -Name "Operations Summary" `
        -Success $false `
        -Message $OperationsResponse.Error
}


# ============================================================
# FINAL REPORT
# ============================================================

Write-Section "ACCEPTANCE TEST RESULTS"

Write-Host ""

$Results |
    Format-Table `
        -AutoSize


Write-Host ""

Write-Host "Passed:   $Passed" -ForegroundColor Green
Write-Host "Failed:   $Failed" -ForegroundColor Red
Write-Host "Warnings: $Warnings" -ForegroundColor Yellow


Write-Host ""
Write-Host "============================================================"

if ($Failed -eq 0) {

    Write-Host " RESULT: FINEMED AI PASSED PRODUCTION ACCEPTANCE TEST" `
        -ForegroundColor Green

    Write-Host "============================================================"

    Write-Host ""
    Write-Host "The deployed application passed all mandatory acceptance checks."

    if ($Warnings -gt 0) {

        Write-Host ""
        Write-Host "Review the warnings above before final client handover." `
            -ForegroundColor Yellow
    }

    exit 0
}
else {

    Write-Host " RESULT: FINEMED AI HAS ACCEPTANCE TEST FAILURES" `
        -ForegroundColor Red

    Write-Host "============================================================"

    Write-Host ""
    Write-Host "Do not perform final client handover until the failures are reviewed."

    exit 1
}