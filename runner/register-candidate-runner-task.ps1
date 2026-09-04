[CmdletBinding()]
param(
    [ValidateSet('install', 'remove', 'show')]
    [string]$Action = 'show',
    [string]$TaskName = 'StockAI-Goal3-Candidates',
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
$Runner = (Resolve-Path (Join-Path $PSScriptRoot 'candidate-runner.ps1')).Path

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function Quote-Argument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

$selectionPath = Resolve-ProjectPath $FeatureSelection
$buildPath = Resolve-ProjectPath $BuildManifest
$campaignPath = Resolve-ProjectPath $CampaignRoot
$reportPath = Resolve-ProjectPath $ReportRoot
$registryPath = Resolve-ProjectPath $ExperimentRegistry
$checkpointPath = Resolve-ProjectPath $CheckpointRoot
$logPath = Resolve-ProjectPath $LogRoot
$selectionId = [System.IO.Path]::GetFileNameWithoutExtension($selectionPath)
$windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = @(
    '-NoLogo',
    '-NoProfile',
    '-NonInteractive',
    '-WindowStyle Hidden',
    '-ExecutionPolicy Bypass',
    '-File', (Quote-Argument $Runner),
    '-Action run',
    '-FeatureSelection', (Quote-Argument $selectionPath),
    '-BuildManifest', (Quote-Argument $buildPath),
    '-CodeCommit', (Quote-Argument $CodeCommit),
    '-Seeds', (Quote-Argument $Seeds),
    '-TuningTrials', [string]$TuningTrials,
    '-TuningTimeoutSeconds', [string]$TuningTimeoutSeconds,
    '-EstimatorCount', [string]$EstimatorCount,
    '-InitialTrainPeriods', [string]$InitialTrainPeriods,
    '-ValidationPeriods', [string]$ValidationPeriods,
    '-StepPeriods', [string]$StepPeriods,
    '-HoldoutPeriods', [string]$HoldoutPeriods,
    "-RunDiagnostics:$($RunDiagnostics.ToString().ToLowerInvariant())",
    '-MaxMaterializedOofRows', [string]$MaxMaterializedOofRows,
    '-MaxModelFits', [string]$MaxModelFits,
    '-CampaignRoot', (Quote-Argument $campaignPath),
    '-ReportRoot', (Quote-Argument $reportPath),
    '-ExperimentRegistry', (Quote-Argument $registryPath),
    '-CheckpointRoot', (Quote-Argument $checkpointPath),
    '-LogRoot', (Quote-Argument $logPath)
) -join ' '

if ($Action -eq 'show') {
    [pscustomobject]@{
        task_name = $TaskName
        executable = $windowsPowerShell
        arguments = $arguments
        selection = $selectionPath
        code_commit = $CodeCommit
        seeds = $Seeds
        tuning_trials = $TuningTrials
        tuning_timeout_seconds = $TuningTimeoutSeconds
        estimator_count = $EstimatorCount
        holdout_periods = $HoldoutPeriods
        schedule = 'at logon and every 5 minutes; overlapping instances ignored'
        holdout_accessed = $false
    } | Format-List
    exit 0
}

if ($Action -eq 'remove') {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    "removed=$TaskName"
    exit 0
}

$actionDefinition = New-ScheduledTaskAction -Execute $windowsPowerShell -Argument $arguments -WorkingDirectory $ProjectRoot
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$repeating = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -Hidden `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 7)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $actionDefinition `
    -Trigger @($atLogon, $repeating) `
    -Settings $settings `
    -Description 'Resume authenticated Goal 3 final candidates; never opens the locked holdout or places orders.' `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
"installed=$TaskName selection=$selectionId"
