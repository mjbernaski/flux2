# Register the "FluxServerWatchdog" scheduled task, which runs watchdog.ps1 at
# logon and every few minutes after, restarting the FLUX server whenever it is
# found down. Re-running this replaces the existing task.
#
# Usage:  .\install_watchdog.ps1                 install/refresh (5 min interval)
#         .\install_watchdog.ps1 -Minutes 2      check more often
#         .\install_watchdog.ps1 -Uninstall      remove the task
#
# Logon-type note: this registers as an Interactive task, which needs no stored
# password but only runs while this user is logged on. Covering a logged-off or
# freshly-rebooted-to-login-screen box needs "run whether user is logged on or
# not", which means either the account password (schtasks /RU /RP) or SYSTEM -
# and SYSTEM has its own profile, so it would need HF_HOME and the HF token
# pointed at this user's cache. Interactive is the honest default here; see
# README if that ever needs to change.

param(
    [int]$Minutes = 5,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$TaskName = "FluxServerWatchdog"
$FluxRoot = $PSScriptRoot

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    } else {
        Write-Host "No scheduled task '$TaskName' to remove." -ForegroundColor Yellow
    }
    exit 0
}

$watchdog = Join-Path $FluxRoot "watchdog.ps1"
if (-not (Test-Path $watchdog)) {
    Write-Host "watchdog.ps1 not found next to this script." -ForegroundColor Red
    exit 1
}

# PowerShell 7 if present, Windows PowerShell otherwise - watchdog.ps1 runs
# under both.
#
# The path recorded here has to outlive PowerShell updates, and Get-Command's
# .Source alone does not: on a Store install it resolves to the *versioned*
# package directory (WindowsApps\Microsoft.PowerShell_7.6.5.0_x64__...\pwsh.exe),
# which the next update deletes. The task then fails every interval with
# 0x80070002 (file not found), silently - Task Scheduler has no opinion about a
# task that cannot start - so the server stays down until a human notices. That
# is exactly what happened between 2026-09-09 and 2026-09-13, across a 7.6.5 ->
# 7.6.6 update. Prefer paths that carry no version: the per-user execution
# alias, then an MSI install, then Windows PowerShell in System32.
$candidates = @(
    (Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\pwsh.exe'),
    (Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe'),
    (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe')
)
$shell = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $shell) {
    # Last resort: whatever is on PATH, versioned or not.
    $pwshCmd = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    $shell = if ($pwshCmd) { $pwshCmd.Source } else { (Get-Command powershell.exe).Source }
}
Write-Host "Task will run: $shell" -ForegroundColor Cyan

$action = New-ScheduledTaskAction -Execute $shell `
    -Argument "-NoProfile -NoLogo -NonInteractive -WindowStyle Hidden -File `"$watchdog`"" `
    -WorkingDirectory $FluxRoot

# Two triggers: the logon one closes the gap between signing in and the first
# interval tick; the repeating one is what actually keeps it up.
#
# Duration is 10 years rather than [TimeSpan]::MaxValue: MaxValue serializes to
# P99999999DT23H59M59S, which Task Scheduler rejects outright ("value which is
# incorrectly formatted or out of range"). There is no "forever" this cmdlet
# will emit, so pick a number longer than the machine.
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$repeating = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $Minutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

# IgnoreNew: a check that overruns its interval must not stack up copies of
# itself. StartWhenAvailable catches up after the machine was asleep. The time
# limit is generous but finite - watchdog.ps1 returns in seconds because it
# launches the server with -NoWait.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

# Register-ScheduledTask reports an invalid task XML as a non-terminating error
# even under $ErrorActionPreference = "Stop", so a bad trigger would otherwise
# print the failure and the "Registered" banner right after it. Force it to
# throw, then confirm the task is really there.
Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger @($atLogon, $repeating) -Settings $settings -Principal $principal `
    -Description "Restarts the FLUX image server (run_server.ps1) whenever it is found not listening." `
    -Force -ErrorAction Stop | Out-Null

if (-not (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
    Write-Host "Registration reported success but '$TaskName' does not exist." -ForegroundColor Red
    exit 1
}

Write-Host "Registered '$TaskName': at logon, then every $Minutes minutes." -ForegroundColor Green
Write-Host "  Check:     Get-ScheduledTask $TaskName | Get-ScheduledTaskInfo" -ForegroundColor Cyan
Write-Host "  Run now:   Start-ScheduledTask $TaskName" -ForegroundColor Cyan
Write-Host "  Log:       $FluxRoot\watchdog.log" -ForegroundColor Cyan
Write-Host "  Remove:    .\install_watchdog.ps1 -Uninstall" -ForegroundColor Cyan
