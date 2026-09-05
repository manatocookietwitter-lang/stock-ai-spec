[CmdletBinding()]
param(
    [ValidateSet('install', 'remove', 'show')]
    [string]$Action = 'show',
    [string]$TaskName = 'StockAI-Goal3-Phased'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runner = (Resolve-Path (Join-Path $PSScriptRoot 'goal3-phase-runner.ps1')).Path
$WindowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$escapedRunner = $Runner.Replace('"', '\"')
$arguments = "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$escapedRunner`" -Action run"

if ($Action -eq 'show') {
    [pscustomobject]@{
        task_name = $TaskName
        executable = $WindowsPowerShell
        arguments = $arguments
        sequence = 'boundary seed17 -> 1d screen -> 5d/20d screen -> evaluation-ready'
        current_worker_action = 'none; waits on existing runner lock'
        progress_command = 'powershell -NoProfile -ExecutionPolicy Bypass -File runner/goal3-phase-runner.ps1 -Action status'
        locked_holdout_accessed = $false
    } | Format-List | Out-String -Width 4096 | Write-Output
    exit 0
}

if ($Action -eq 'remove') {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    "removed=$TaskName"
    exit 0
}

$actionDefinition = New-ScheduledTaskAction -Execute $WindowsPowerShell -Argument $arguments -WorkingDirectory $ProjectRoot
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
    -Description 'Run precommitted Goal 3 screening phases after the active seed17 reaches its safe boundary; never opens locked holdout.' `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
"installed=$TaskName current_worker_action=none holdout_accessed=false"
