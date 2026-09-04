[CmdletBinding()]
param(
    [ValidateSet('run', 'status')]
    [string]$Action = 'status',
    [Parameter(Mandatory = $true)]
    [string]$FeatureSelection,
    [Parameter(Mandatory = $true)]
    [string]$BuildManifest,
    [Parameter(Mandatory = $true)]
    [string]$CodeCommit,
    [string]$Seeds = '17,29,43',
    [int]$TuningTrials = 20,
    [int]$TuningTimeoutSeconds = 900,
    [int]$EstimatorCount = 300,
    [int]$InitialTrainPeriods = 500,
    [int]$ValidationPeriods = 60,
    [int]$StepPeriods = 60,
    [int]$HoldoutPeriods = 120,
    [bool]$RunDiagnostics = $true,
    [int]$MaxMaterializedOofRows = 20000000,
    [int]$MaxModelFits = 5000,
    [string]$CampaignRoot = 'artifacts/campaigns/goal3-candidates',
    [string]$ReportRoot = 'artifacts/reports/advanced-candidates',
    [string]$ExperimentRegistry = 'artifacts/experiments/advanced.jsonl',
    [string]$CheckpointRoot = 'artifacts/checkpoints/advanced-candidates',
    [string]$LogRoot = 'artifacts/logs/goal3-candidates'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function Assert-SourceProvenance([string]$Commit) {
    if (-not $Commit.Trim() -or $Commit -eq 'UNSET') {
        throw 'candidate model code commit must be explicit'
    }
    $safeDirectory = "safe.directory=$($ProjectRoot.Replace('\', '/'))"
    & git -c $safeDirectory -C $ProjectRoot cat-file -e "$Commit`^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "candidate model code commit is unavailable: $Commit"
    }
    & git -c $safeDirectory -C $ProjectRoot diff --quiet $Commit -- src pyproject.toml uv.lock
    if ($LASTEXITCODE -ne 0) {
        throw 'current model source or dependency lock differs from candidate code commit'
    }
    $untrackedSource = @(
        & git -c $safeDirectory -C $ProjectRoot ls-files --others --exclude-standard -- src
    )
    if ($LASTEXITCODE -ne 0 -or $untrackedSource.Count -gt 0) {
        throw 'untracked model source prevents candidate provenance authentication'
    }
}

function Write-RunnerEvent([string]$Root, [string]$Event, [string]$Detail) {
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    $record = [ordered]@{
        at = [DateTimeOffset]::UtcNow.ToString('o')
        event = $Event
        detail = $Detail
    }
    Add-Content -LiteralPath (Join-Path $Root 'runner.jsonl') -Value ($record | ConvertTo-Json -Compress) -Encoding UTF8
}

$selectionPath = Resolve-ProjectPath $FeatureSelection
$buildPath = Resolve-ProjectPath $BuildManifest
$campaignPath = Resolve-ProjectPath $CampaignRoot
$reportPath = Resolve-ProjectPath $ReportRoot
$registryPath = Resolve-ProjectPath $ExperimentRegistry
$checkpointPath = Resolve-ProjectPath $CheckpointRoot
$logPath = Resolve-ProjectPath $LogRoot
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "candidate runner Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $selectionPath -PathType Leaf)) {
    throw "frozen feature selection is missing: $selectionPath"
}
if (-not (Test-Path -LiteralPath $buildPath -PathType Leaf)) {
    throw "Production Build manifest is missing: $buildPath"
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Remove-Item Env:JQUANTS_API_KEY -ErrorAction SilentlyContinue

if ($Action -eq 'status') {
    & $python -m stock_ai research candidate-status `
        --feature-selection $selectionPath `
        --campaign-root $campaignPath
    exit $LASTEXITCODE
}

Assert-SourceProvenance $CodeCommit
New-Item -ItemType Directory -Path $campaignPath -Force | Out-Null
$selectionId = [System.IO.Path]::GetFileNameWithoutExtension($selectionPath)
$lockPath = Join-Path $campaignPath "$selectionId.runner.lock"
$lock = $null
try {
    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
} catch [System.IO.IOException] {
    Write-RunnerEvent $logPath 'LOCKED' 'another independent runner owns the candidate workflow lock'
    exit 0
}

try {
    $arguments = @(
        '-m', 'stock_ai', 'research', 'candidate-campaigns',
        '--feature-selection', $selectionPath,
        '--build-manifest', $buildPath,
        '--code-commit', $CodeCommit,
        '--seeds', $Seeds,
        '--tuning-trials', [string]$TuningTrials,
        '--tuning-timeout-seconds', [string]$TuningTimeoutSeconds,
        '--estimator-count', [string]$EstimatorCount,
        '--initial-train-periods', [string]$InitialTrainPeriods,
        '--validation-periods', [string]$ValidationPeriods,
        '--step-periods', [string]$StepPeriods,
        '--holdout-periods', [string]$HoldoutPeriods,
        $(if ($RunDiagnostics) { '--run-diagnostics' } else { '--no-run-diagnostics' }),
        '--max-materialized-oof-rows', [string]$MaxMaterializedOofRows,
        '--max-model-fits', [string]$MaxModelFits,
        '--campaign-root', $campaignPath,
        '--report-root', $reportPath,
        '--experiment-registry', $registryPath,
        '--checkpoint-root', $checkpointPath,
        '--log-root', $logPath
    )
    Write-RunnerEvent $logPath 'RUN_OR_RESUME' $selectionId
    & $python @arguments
    $exitCode = $LASTEXITCODE
    Write-RunnerEvent $logPath $(if ($exitCode -eq 0) { 'EXIT_SUCCESS' } else { 'EXIT_FAILED' }) "exit_code=$exitCode"
    exit $exitCode
} catch {
    Write-RunnerEvent $logPath 'RUNNER_FAILED' $_.Exception.GetType().FullName
    exit 1
} finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
}
