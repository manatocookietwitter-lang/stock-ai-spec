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

function Test-WorkerIdentity($Batch) {
    if ($null -eq $Batch.child_pid -or $null -eq $Batch.started_at) {
        return $false
    }
    $process = Get-Process -Id ([int]$Batch.child_pid) -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.ProcessName -notlike 'python*') {
        return $false
    }
    try {
        $recordedStart = [DateTimeOffset]::Parse([string]$Batch.started_at).UtcDateTime
        $observedStart = $process.StartTime.ToUniversalTime()
    } catch {
        return $false
    }
    return [Math]::Abs(($observedStart - $recordedStart).TotalSeconds) -le 120
}

function Invoke-PythonCode([string]$Python, [string]$Code, [string[]]$Arguments) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $bootstrap = "import base64;exec(base64.b64decode('$encoded'))"
    & $Python -c $bootstrap @Arguments
    return $LASTEXITCODE
}

function Set-StaleBatchesInterrupted([string]$ManifestPath, [string[]]$BatchIds) {
    if ($BatchIds.Count -eq 0) {
        return
    }
    $transition = @'
from datetime import UTC, datetime
from pathlib import Path
import sys
from stock_ai.ml.campaign import CampaignBatchStatus, load_campaign_manifest, write_campaign_manifest

path = Path(sys.argv[1]).resolve()
targets = frozenset(sys.argv[2:])
manifest = load_campaign_manifest(path)
for batch in manifest.batches:
    if batch.batch_id in targets and batch.status is CampaignBatchStatus.RUNNING:
        batch.status = CampaignBatchStatus.INTERRUPTED
        batch.child_pid = None
        batch.completed_at = datetime.now(UTC)
        batch.last_error = "candidate runner observed missing or mismatched worker identity"
write_campaign_manifest(manifest, path)
'@
    $exitCode = Invoke-PythonCode $python $transition (@($ManifestPath) + $BatchIds)
    if ($exitCode -ne 0) {
        throw 'failed to persist stale candidate RUNNING to INTERRUPTED transition'
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

function Get-CandidateStatus() {
    $statusJson = @(& $python -m stock_ai research candidate-status `
        --feature-selection $selectionPath `
        --campaign-root $campaignPath)
    if ($LASTEXITCODE -ne 0) {
        throw 'candidate status authentication failed'
    }
    $status = ($statusJson -join [Environment]::NewLine) | ConvertFrom-Json
    foreach ($horizon in $status.horizons) {
        if ([string]$horizon.status -ne 'STARTED') {
            continue
        }
        $manifestPath = [string]$horizon.campaign.manifest
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($batchStatus in $horizon.campaign.batches) {
            $batch = @($manifest.batches | Where-Object {
                [string]$_.batch_id -eq [string]$batchStatus.batch_id
            }) | Select-Object -First 1
            if ($null -eq $batch) {
                throw 'candidate status batch is absent from its authenticated manifest'
            }
            $workerAlive = ([string]$batch.status -eq 'RUNNING') -and (Test-WorkerIdentity $batch)
            $batchStatus.worker_alive = $workerAlive
            $batchStatus.effective_status = $(
                if ([string]$batch.status -eq 'RUNNING' -and -not $workerAlive) {
                    'INTERRUPTED'
                } else {
                    [string]$batch.status
                }
            )
        }
        $current = @($horizon.campaign.batches | Where-Object {
            [string]$_.effective_status -eq 'RUNNING'
        }) | Select-Object -First 1
        if ($null -eq $current) {
            $current = @($horizon.campaign.batches | Where-Object {
                [string]$_.effective_status -ne 'SUCCEEDED'
            }) | Select-Object -First 1
        }
        $horizon.campaign.current_batch = $current
    }
    return $status
}

if ($Action -eq 'status') {
    Get-CandidateStatus | ConvertTo-Json -Depth 12
    exit 0
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
    $status = Get-CandidateStatus
    $active = @(
        $status.horizons |
            Where-Object { [string]$_.status -eq 'STARTED' } |
            ForEach-Object { $_.campaign.batches } |
            Where-Object { [string]$_.effective_status -eq 'RUNNING' }
    )
    if ($active.Count -gt 0) {
        Write-RunnerEvent $logPath 'ALREADY_RUNNING' (($active.batch_id) -join ',')
        exit 0
    }
    foreach ($horizon in $status.horizons) {
        if ([string]$horizon.status -ne 'STARTED') {
            continue
        }
        $stale = @(
            $horizon.campaign.batches |
                Where-Object {
                    [string]$_.stored_status -eq 'RUNNING' -and
                    [string]$_.effective_status -eq 'INTERRUPTED'
                } |
                ForEach-Object { [string]$_.batch_id }
        )
        Set-StaleBatchesInterrupted ([string]$horizon.campaign.manifest) $stale
    }
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
