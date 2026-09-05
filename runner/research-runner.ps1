[CmdletBinding()]
param(
    [ValidateSet('run', 'status')]
    [string]$Action = 'status',
    [string]$Manifest = 'artifacts/campaigns/goal3-base-v3.json',
    [string]$LogRoot = '',
    [switch]$Json,
    [switch]$VerifyArtifacts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'worker-state.ps1')

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function Invoke-PythonCode([string]$Python, [string]$Code, [string[]]$Arguments) {
    # Windows PowerShell 5.1 rewrites embedded double quotes in native command
    # arguments.  Base64 keeps the audited static snippet byte-exact without
    # persisting a temporary source file.
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $bootstrap = "import base64;exec(base64.b64decode('$encoded'))"
    & $Python -c $bootstrap @Arguments
    return $LASTEXITCODE
}

function Invoke-PythonValidation([string]$ManifestPath, [bool]$FullArtifactCheck) {
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "research runner Python is missing: $python"
    }
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    Remove-Item Env:JQUANTS_API_KEY -ErrorAction SilentlyContinue
    $validation = @'
from pathlib import Path
import sys
from stock_ai.ml.campaign import load_campaign_build_id, load_campaign_manifest, reconcile_campaign

path = Path(sys.argv[1]).resolve()
manifest = load_campaign_manifest(path)
observed_build_id = load_campaign_build_id(Path(manifest.build_manifest_path))
if observed_build_id != manifest.build_id:
    raise RuntimeError("campaign and Production Build identities differ")
if sys.argv[2] == "verify":
    completed = frozenset(batch.batch_id for batch in manifest.batches if batch.status == "SUCCEEDED")
    reconcile_campaign(manifest, batch_ids=completed)
'@
    $mode = if ($FullArtifactCheck) { 'verify' } else { 'metadata' }
    $exitCode = Invoke-PythonCode $python $validation @($ManifestPath, $mode)
    if ($exitCode -ne 0) {
        throw 'campaign manifest, build, or completed artifact authentication failed'
    }
}

function Assert-SourceProvenance($Payload) {
    $commit = [string]$Payload.code_commit
    $safeDirectory = "safe.directory=$($ProjectRoot.Replace('\', '/'))"
    & git -c $safeDirectory -C $ProjectRoot cat-file -e "$commit`^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "campaign code commit is unavailable: $commit"
    }
    & git -c $safeDirectory -C $ProjectRoot diff --quiet $commit -- src pyproject.toml uv.lock
    if ($LASTEXITCODE -ne 0) {
        throw 'current model source or dependency lock differs from campaign code commit'
    }
    $untrackedSource = @(
        & git -c $safeDirectory -C $ProjectRoot ls-files --others --exclude-standard -- src
    )
    if ($LASTEXITCODE -ne 0) {
        throw 'unable to authenticate untracked model source state'
    }
    if ($untrackedSource.Count -gt 0) {
        throw 'untracked model source exists outside the campaign code commit'
    }
}

function Get-RunnerStatus([string]$ManifestPath, [bool]$FullArtifactCheck) {
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "campaign manifest is missing: $ManifestPath"
    }
    $payload = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-SourceProvenance $payload
    Invoke-PythonValidation $ManifestPath $FullArtifactCheck
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    # Read checkpoint progress without calling campaign._process_is_running.
    # Python's POSIX-style os.kill(pid, 0) probe is not a safe Windows process
    # query and can raise WinError 87. Process identity is classified below by
    # Get-WorkerProcessState instead.
    $granularReader = @'
import json
from pathlib import Path
import sys
from stock_ai.ml.campaign import campaign_batch_checkpoint_path, load_campaign_manifest
from stock_ai.ml.checkpoint import read_checkpoint_status

manifest = load_campaign_manifest(Path(sys.argv[1]).resolve())
batches = []
for batch in manifest.batches:
    checkpoint_path = campaign_batch_checkpoint_path(manifest, batch)
    summary = None
    if checkpoint_path is not None and checkpoint_path.is_dir():
        checkpoint = read_checkpoint_status(checkpoint_path)
        units = checkpoint["units"]
        latest = max(units.values(), key=lambda item: str(item.get("updated_at", "")), default=None)
        active = tuple(value for value in units.values() if value.get("status") == "RUNNING")
        current = max(active or tuple(units.values()), key=lambda item: str(item.get("updated_at", "")), default=None)
        summary = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "updated_at": checkpoint["updated_at"],
            "unit_counts": checkpoint["unit_counts"],
            "active_units": active,
            "current_unit": current,
            "latest_unit": latest,
        }
    batches.append({"batch_id": batch.batch_id, "checkpoint": summary})
print(json.dumps({"batches": batches}, ensure_ascii=False, sort_keys=True))
'@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($granularReader))
    $bootstrap = "import base64;exec(base64.b64decode('$encoded'))"
    $granularJson = @(& $python -c $bootstrap $ManifestPath)
    if ($LASTEXITCODE -ne 0) {
        throw 'read-only granular campaign status authentication failed'
    }
    $granular = ($granularJson -join [Environment]::NewLine) | ConvertFrom-Json
    $batches = @()
    foreach ($batch in $payload.batches) {
        $workerState = $null
        if ([string]$batch.status -eq 'RUNNING') {
            $workerState = Get-WorkerProcessState $batch
        }
        $effective = Get-EffectiveWorkerStatus ([string]$batch.status) $workerState
        $workerAlive = ConvertTo-WorkerAlive $workerState
        $granularBatch = @($granular.batches | Where-Object {
            [string]$_.batch_id -eq [string]$batch.batch_id
        }) | Select-Object -First 1
        $currentEvidence = $null
        if ($null -ne $granularBatch -and $null -ne $granularBatch.checkpoint) {
            $currentEvidence = $granularBatch.checkpoint.current_unit.evidence
        }
        $batches += [pscustomobject]@{
            batch_id = [string]$batch.batch_id
            horizon = [int]$batch.horizon
            model_family = [string]$batch.model_family
            seed = $(
                if ($batch.PSObject.Properties.Name -contains 'seed' -and $null -ne $batch.seed) {
                    [int]$batch.seed
                } else {
                    $null
                }
            )
            stored_status = [string]$batch.status
            effective_status = $effective
            attempts = [int]$batch.attempts
            worker_state = $workerState
            worker_alive = $workerAlive
            report_id = $batch.report_id
            last_error = $batch.last_error
            current_task = $(if ($null -eq $currentEvidence) { $null } else { $currentEvidence.task })
            current_fold = $(if ($null -eq $currentEvidence) { $null } else { $currentEvidence.fold })
            checkpoint = $(if ($null -eq $granularBatch) { $null } else { $granularBatch.checkpoint })
        }
    }
    return [pscustomobject]@{
        schema_version = 'research-runner-status-v2'
        checked_at = [DateTimeOffset]::UtcNow.ToString('o')
        campaign_id = [string]$payload.campaign_id
        code_commit = [string]$payload.code_commit
        manifest = $ManifestPath
        validation = $(if ($FullArtifactCheck) { 'FULL_AUTHENTICATED' } else { 'MANIFEST_BUILD_SOURCE_AUTHENTICATED' })
        locked_holdout_accessed = $false
        batches = $batches
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

function Get-PauseAfterBatch([string]$ManifestPath) {
    $policyPath = "$ManifestPath.pause-after.json"
    if (-not (Test-Path -LiteralPath $policyPath -PathType Leaf)) {
        return $null
    }
    $policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        [string]$policy.schema_version -ne 'research-runner-pause-after-v1' -or
        -not ([string]$policy.batch_id).Trim()
    ) {
        throw 'runner pause-after policy is invalid'
    }
    return $policy
}

function Set-StaleBatchesInterrupted([string]$ManifestPath, [string[]]$BatchIds) {
    if ($BatchIds.Count -eq 0) {
        return
    }
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    Remove-Item Env:JQUANTS_API_KEY -ErrorAction SilentlyContinue
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
        batch.last_error = "independent runner observed missing or mismatched worker identity"
write_campaign_manifest(manifest, path)
'@
    $exitCode = Invoke-PythonCode $python $transition (@($ManifestPath) + $BatchIds)
    if ($exitCode -ne 0) {
        throw 'failed to persist stale RUNNING to INTERRUPTED transition'
    }
}

function Invoke-Campaign([string]$ManifestPath, [string]$RunnerLogRoot) {
    $status = Get-RunnerStatus $ManifestPath $false
    $active = @($status.batches | Where-Object { $_.effective_status -eq 'RUNNING' })
    if ($active.Count -gt 0) {
        Write-RunnerEvent $RunnerLogRoot 'ALREADY_RUNNING' (($active.batch_id) -join ',')
        return 0
    }
    $pausePolicy = Get-PauseAfterBatch $ManifestPath
    if ($null -ne $pausePolicy) {
        if (
            $pausePolicy.PSObject.Properties.Name -contains 'campaign_id' -and
            [string]$pausePolicy.campaign_id -ne [string]$status.campaign_id
        ) {
            throw 'runner pause-after campaign identity mismatch'
        }
        $boundaryState = Get-PauseBoundaryState $status.batches ([string]$pausePolicy.batch_id)
        if ($boundaryState -eq 'PAUSE') {
            Write-RunnerEvent $RunnerLogRoot 'PAUSED_AT_POLICY_BOUNDARY' ([string]$pausePolicy.batch_id)
            return 0
        }
    }
    $unknown = @($status.batches | Where-Object { $_.effective_status -eq 'UNKNOWN' })
    if ($unknown.Count -gt 0) {
        Write-RunnerEvent $RunnerLogRoot 'WORKER_STATE_UNKNOWN' (($unknown.batch_id) -join ',')
        return 0
    }
    $stale = @($status.batches | Where-Object {
        $_.stored_status -eq 'RUNNING' -and $_.effective_status -eq 'INTERRUPTED'
    } | ForEach-Object { $_.batch_id })
    Set-StaleBatchesInterrupted $ManifestPath $stale

    $payload = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $remaining = @($payload.batches | Where-Object { $_.status -ne 'SUCCEEDED' })
    if ($remaining.Count -eq 0) {
        Write-RunnerEvent $RunnerLogRoot 'COMPLETE' ([string]$payload.campaign_id)
        return 0
    }

    $horizons = @($payload.batches | ForEach-Object { [int]$_.horizon } | Select-Object -Unique) -join ','
    $families = @($payload.batches | ForEach-Object { [string]$_.model_family } | Select-Object -Unique) -join ','
    $config = $payload.common_config
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    Remove-Item Env:JQUANTS_API_KEY -ErrorAction SilentlyContinue
    $childLogRoot = Join-Path $RunnerLogRoot 'batches'
    New-Item -ItemType Directory -Path $childLogRoot -Force | Out-Null
    $arguments = @(
        '-m', 'stock_ai', 'research', 'campaign',
        '--build-manifest', [string]$payload.build_manifest_path,
        '--code-commit', [string]$payload.code_commit,
        '--horizons', $horizons,
        '--model-families', $families,
        '--feature-names', (@($config.feature_names) -join ','),
        '--seeds', (@($config.seeds) -join ','),
        '--target-family', [string]$config.target_family,
        '--tuning-trials', [string]$config.tuning_trials,
        '--tuning-timeout-seconds', [string]$config.tuning_timeout_seconds,
        '--estimator-count', [string]$config.estimator_count,
        '--initial-train-periods', [string]$config.initial_train_periods,
        '--validation-periods', [string]$config.validation_periods,
        '--step-periods', [string]$config.step_periods,
        '--holdout-periods', [string]$config.holdout_periods,
        $(if ([bool]$config.run_ablations) { '--run-ablations' } else { '--no-run-ablations' }),
        $(if ([bool]$config.run_diagnostics) { '--run-diagnostics' } else { '--no-run-diagnostics' }),
        '--max-materialized-oof-rows', [string]$config.max_materialized_oof_rows,
        '--max-model-fits', [string]$config.max_model_fits,
        '--campaign-manifest', $ManifestPath,
        '--report-root', [string]$payload.report_root,
        '--experiment-registry', [string]$payload.experiment_registry,
        '--log-root', $childLogRoot
    )
    if (
        $payload.PSObject.Properties.Name -contains 'checkpoint_root' -and
        $null -ne $payload.checkpoint_root
    ) {
        $arguments += @('--checkpoint-root', [string]$payload.checkpoint_root)
    }
    Write-RunnerEvent $RunnerLogRoot 'RESUME' ([string]$payload.campaign_id)
    & $python @arguments
    $exitCode = $LASTEXITCODE
    Write-RunnerEvent $RunnerLogRoot $(if ($exitCode -eq 0) { 'EXIT_SUCCESS' } else { 'EXIT_FAILED' }) "exit_code=$exitCode"
    return $exitCode
}

$manifestPath = Resolve-ProjectPath $Manifest
if ($Action -eq 'status') {
    $status = Get-RunnerStatus $manifestPath ([bool]$VerifyArtifacts)
    if ($Json) {
        $status | ConvertTo-Json -Depth 6
    } else {
        "campaign=$($status.campaign_id) validation=$($status.validation) holdout_accessed=false"
        $status.batches | Format-Table horizon, model_family, seed, current_task, current_fold, stored_status, effective_status, attempts, worker_state -AutoSize
    }
    exit 0
}

$runnerLogRoot = if ($LogRoot) {
    Resolve-ProjectPath $LogRoot
} else {
    Join-Path $ProjectRoot 'artifacts\logs\research-runner'
}
$lockPath = "$manifestPath.runner.lock"
$lock = $null
try {
    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
} catch [System.IO.IOException] {
    Write-RunnerEvent $runnerLogRoot 'LOCKED' 'another independent runner owns the campaign lock'
    exit 0
}
try {
    try {
        exit (Invoke-Campaign $manifestPath $runnerLogRoot)
    } catch {
        $exceptionType = $_.Exception.GetType().FullName
        Write-RunnerEvent $runnerLogRoot 'RUNNER_FAILED' $exceptionType
        exit 1
    }
} finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
}
