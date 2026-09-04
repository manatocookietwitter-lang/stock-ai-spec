[CmdletBinding()]
param(
    [ValidateSet('install', 'remove', 'show')]
    [string]$Action = 'show',
    [string]$TaskName = 'StockAI-Goal3-Research',
    [string]$Manifest = 'artifacts/campaigns/goal3-base-v3.json'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runner = (Resolve-Path (Join-Path $PSScriptRoot 'research-runner.ps1')).Path
$ManifestPath = if ([System.IO.Path]::IsPathRooted($Manifest)) {
    [System.IO.Path]::GetFullPath($Manifest)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Manifest))
}
$WindowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$escapedRunner = $Runner.Replace('"', '\"')
$escapedManifest = $ManifestPath.Replace('"', '\"')
$arguments = "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$escapedRunner`" -Action run -Manifest `"$escapedManifest`""

if ($Action -eq 'show') {
    [pscustomobject]@{
        task_name = $TaskName
        executable = $WindowsPowerShell
        arguments = $arguments
        manifest = $ManifestPath
        schedule = 'at logon and every 5 minutes; overlapping instances ignored'
    } | Format-List
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
    -Description 'Resume authenticated Goal 3 research without Codex; never opens the locked holdout.' `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
"installed=$TaskName manifest=$ManifestPath"
