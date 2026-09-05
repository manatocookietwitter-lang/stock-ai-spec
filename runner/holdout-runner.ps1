[CmdletBinding()]
param(
    [ValidateSet('run', 'status')]
    [string]$Action = 'status',
    [Parameter(Mandatory = $true)]
    [string]$Selection,
    [Parameter(Mandatory = $true)]
    [string]$BuildManifest,
    [Parameter(Mandatory = $true)]
    [string]$CodeCommit,
    [string]$EvaluationRoot = 'artifacts/holdout/goal3',
    [string]$ExperimentRegistry = 'artifacts/experiments/advanced.jsonl',
    [string]$LogRoot = 'artifacts/logs/holdout-runner'
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
        throw 'holdout evaluator code commit must be explicit'
    }
    $safeDirectory = "safe.directory=$($ProjectRoot.Replace('\', '/'))"
    & git -c $safeDirectory -C $ProjectRoot cat-file -e "$Commit`^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "holdout evaluator commit is unavailable: $Commit"
    }
    & git -c $safeDirectory -C $ProjectRoot diff --quiet $Commit -- src pyproject.toml uv.lock
    if ($LASTEXITCODE -ne 0) {
        throw 'current evaluator source or dependency lock differs from the frozen commit'
    }
    $untrackedSource = @(
        & git -c $safeDirectory -C $ProjectRoot ls-files --others --exclude-standard -- src
    )
    if ($LASTEXITCODE -ne 0 -or $untrackedSource.Count -gt 0) {
        throw 'untracked evaluator source prevents holdout provenance authentication'
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

$selectionPath = Resolve-ProjectPath $Selection
$buildPath = Resolve-ProjectPath $BuildManifest
$evaluationPath = Resolve-ProjectPath $EvaluationRoot
$registryPath = Resolve-ProjectPath $ExperimentRegistry
$runnerLogPath = Resolve-ProjectPath $LogRoot
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "holdout runner Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $selectionPath -PathType Leaf)) {
    throw "frozen development selection is missing: $selectionPath"
}
if (-not (Test-Path -LiteralPath $buildPath -PathType Leaf)) {
    throw "Production Build manifest is missing: $buildPath"
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Remove-Item Env:JQUANTS_API_KEY -ErrorAction SilentlyContinue
$selectionId = [System.IO.Path]::GetFileNameWithoutExtension($selectionPath)
$evaluationDirectory = Join-Path $evaluationPath $selectionId

if ($Action -eq 'status') {
    $ledgerPath = Join-Path $evaluationDirectory 'ledger.json'
    if (-not (Test-Path -LiteralPath $ledgerPath -PathType Leaf)) {
        [pscustomobject]@{
            schema_version = 'locked-holdout-runner-status-v1'
            selection_id = $selectionId
            status = 'NOT_STARTED'
            holdout_accessed = $false
            evaluation_directory = $evaluationDirectory
        } | ConvertTo-Json -Compress
        exit 0
    }
    & $python -m stock_ai research holdout-status --evaluation-directory $evaluationDirectory
    exit $LASTEXITCODE
}

try {
    Assert-SourceProvenance $CodeCommit
    Write-RunnerEvent $runnerLogPath 'RUN_OR_RESUME' $selectionId
    & $python -m stock_ai research holdout-evaluate `
        --selection $selectionPath `
        --build-manifest $buildPath `
        --code-commit $CodeCommit `
        --evaluation-root $evaluationPath `
        --experiment-registry $registryPath
    $exitCode = $LASTEXITCODE
    Write-RunnerEvent $runnerLogPath $(if ($exitCode -eq 0) { 'EXIT_SUCCESS' } else { 'EXIT_FAILED' }) "exit_code=$exitCode"
    exit $exitCode
} catch {
    Write-RunnerEvent $runnerLogPath 'RUNNER_FAILED' $_.Exception.GetType().FullName
    exit 1
}
