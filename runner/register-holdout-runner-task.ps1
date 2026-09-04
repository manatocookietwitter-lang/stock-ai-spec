[CmdletBinding()]
param(
    [ValidateSet('install', 'remove', 'show')]
    [string]$Action = 'show',
    [string]$TaskName = 'StockAI-Goal3-Locked-Holdout',
    [Parameter(Mandatory = $true)]
    [string]$Selection,
    [Parameter(Mandatory = $true)]
    [string]$BuildManifest,
    [Parameter(Mandatory = $true)]
    [string]$CodeCommit,
    [string]$EvaluationRoot = 'artifacts/holdout/goal3',
    [string]$ExperimentRegistry = 'artifacts/experiments/advanced.jsonl'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runner = (Resolve-Path (Join-Path $PSScriptRoot 'holdout-runner.ps1')).Path

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

function Quote-Argument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

$selectionPath = Resolve-ProjectPath $Selection
$buildPath = Resolve-ProjectPath $BuildManifest
$evaluationPath = Resolve-ProjectPath $EvaluationRoot
$registryPath = Resolve-ProjectPath $ExperimentRegistry
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
    '-Selection', (Quote-Argument $selectionPath),
    '-BuildManifest', (Quote-Argument $buildPath),
    '-CodeCommit', (Quote-Argument $CodeCommit),
    '-EvaluationRoot', (Quote-Argument $evaluationPath),
    '-ExperimentRegistry', (Quote-Argument $registryPath)
) -join ' '

if ($Action -eq 'show') {
    [pscustomobject]@{
        task_name = $TaskName
        executable = $windowsPowerShell
        arguments = $arguments
        selection = $selectionPath
        schedule = 'at logon and every 5 minutes; completed evaluation is authenticated and reused'
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
    -Description 'Resume the sole authenticated Goal 3 locked-holdout evaluation; never tunes or places orders.' `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
"installed=$TaskName selection=$selectionId"
