function Get-WorkerProcessState {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        $Batch,
        [scriptblock]$ProcessResolver = $null,
        [double]$StartToleranceSeconds = 120
    )

    if ($null -eq $Batch.child_pid -or $null -eq $Batch.started_at) {
        return 'UNKNOWN'
    }
    try {
        $workerPid = [int]$Batch.child_pid
        if ($workerPid -le 0) {
            return 'UNKNOWN'
        }
    } catch {
        return 'UNKNOWN'
    }

    if ($null -eq $ProcessResolver) {
        $ProcessResolver = {
            param([int]$RequestedPid)
            Get-Process -Id $RequestedPid -ErrorAction Stop
        }
    }
    try {
        $process = & $ProcessResolver $workerPid
    } catch {
        # Windows PowerShell 5.1 does not reliably resolve
        # Microsoft.PowerShell.Commands.ProcessCommandException as a catch
        # type. Match the exact Get-Process "PID absent" error identity; every
        # other failure (including WinError 87 and AccessDenied) is UNKNOWN.
        if (
            $_.FullyQualifiedErrorId -like 'NoProcessFoundForGivenId,*' -and
            [string]$_.CategoryInfo.Reason -eq 'ProcessCommandException'
        ) {
            return 'DEAD'
        }
        return 'UNKNOWN'
    }
    if ($null -eq $process) {
        return 'DEAD'
    }

    try {
        if ([string]$process.ProcessName -notlike 'python*') {
            return 'DEAD'
        }
        $recordedStart = [DateTimeOffset]::Parse([string]$Batch.started_at).UtcDateTime
        $observedStart = $process.StartTime.ToUniversalTime()
    } catch {
        return 'UNKNOWN'
    }
    if ([Math]::Abs(($observedStart - $recordedStart).TotalSeconds) -gt $StartToleranceSeconds) {
        return 'DEAD'
    }
    return 'ALIVE'
}

function Get-EffectiveWorkerStatus {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$StoredStatus,
        [string]$WorkerState = $null
    )

    if ($StoredStatus -ne 'RUNNING') {
        return $StoredStatus
    }
    switch ($WorkerState) {
        'ALIVE' { return 'RUNNING' }
        'DEAD' { return 'INTERRUPTED' }
        default { return 'UNKNOWN' }
    }
}

function ConvertTo-WorkerAlive {
    [CmdletBinding()]
    param([string]$WorkerState = $null)

    switch ($WorkerState) {
        'ALIVE' { return $true }
        'DEAD' { return $false }
        default { return $null }
    }
}
