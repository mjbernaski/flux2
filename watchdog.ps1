# Bring the FLUX server back if it is not running. Meant to be driven by the
# "FluxServerWatchdog" scheduled task (install_watchdog.ps1), which fires this
# at logon and every few minutes after.
#
# Why this exists on top of run_server.ps1's own retry loop: that loop only
# covers the server process dying while the supervisor lives. The failure mode
# actually seen on this box killed both at once - torch's Intel OpenMP runtime
# installs a Windows console control handler, so closing the terminal the
# server was launched from aborted python ("forrtl: error (200): program
# aborting due to window-CLOSE event") and took the supervisor with it. In 50MB
# of server.log the retry loop never once fired; every recovery was a human
# relaunching it. run_server.ps1 now disables that handler, but a supervisor
# that dies for any other reason still leaves nothing to restart it.
#
# Usage:  .\watchdog.ps1           check, relaunch if down
#         .\watchdog.ps1 -WhatIf   report what it would do, change nothing

[CmdletBinding(SupportsShouldProcess)]
param()

$FluxRoot = if ($env:FLUX_HOME) { $env:FLUX_HOME } else { $PSScriptRoot }
$LogFile = Join-Path $FluxRoot "watchdog.log"

function Write-WatchdogLog([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    # Best-effort: a locked or unwritable log must not stop a restart.
    try { Add-Content -Path $LogFile -Value $line -ErrorAction Stop } catch {}
}

# Same port resolution as run_server.ps1, kill_flux.ps1 and f14.ps1.
$Port = 2222
if ($env:PORT -match '^\d+$') {
    $Port = [int]$env:PORT
} elseif (Test-Path (Join-Path $FluxRoot ".env")) {
    $m = Select-String -Path (Join-Path $FluxRoot ".env") -Pattern '^PORT=(\d+)' -ErrorAction SilentlyContinue
    if ($m) { $Port = [int]$m.Matches[0].Groups[1].Value }
}

# netstat, not Get-NetTCPConnection: the latter goes through CIM, which is
# access-denied in some shells here (see kill_flux.ps1).
if (netstat -ano | Select-String ":$Port\s.*LISTENING") {
    Write-Verbose "Port $Port is listening - nothing to do."
    exit 0
}

# A deliberate stop must stay stopped: kill_flux.ps1 leaves this sentinel behind
# and run_server.ps1 clears it on the next start, so "down" and "wanted down"
# are distinguishable without a second switch to remember.
$stopFile = Join-Path $FluxRoot ".flux_stopped"
if (Test-Path $stopFile) {
    Write-Verbose "Port $Port down, but .flux_stopped is present - stopped on purpose, leaving it."
    exit 0
}

# The port being down does not mean the server is gone: a cold start spends
# minutes loading weights before it binds, and this task fires every few
# minutes. run_server.ps1 writes the supervisor's own PID to server.pid and
# clears it on exit, so a live PID there means a start is already in flight.
# Without this check a slow load would be relaunched on top of itself, and
# run_server.ps1's Stop-ExistingServer would kill the instance that was almost
# ready - a loop that never converges.
$pidFile = Join-Path $FluxRoot "server.pid"
if (Test-Path $pidFile) {
    $supervisorPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($supervisorPid -match '^\d+$' -and (Get-Process -Id ([int]$supervisorPid) -ErrorAction SilentlyContinue)) {
        Write-Verbose "Supervisor $supervisorPid is alive - still loading, leaving it alone."
        exit 0
    }
}

# Come back on whatever config was last run, so a deliberate switch to another
# model survives a restart. run_server.ps1 keeps this current.
$Config = 9
$lastConfigFile = Join-Path $FluxRoot ".last_config"
if (Test-Path $lastConfigFile) {
    $last = (Get-Content $lastConfigFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($last -match '^([1-9]|1[0-4])$') { $Config = [int]$last }
}

$launcher = Join-Path $FluxRoot "f14.ps1"
if (-not (Test-Path $launcher)) {
    Write-WatchdogLog "f14.ps1 not found under $FluxRoot - cannot restart."
    exit 1
}

if (-not $PSCmdlet.ShouldProcess("FLUX server config $Config", "restart")) { exit 0 }

Write-WatchdogLog "Port $Port down and no live supervisor - restarting config $Config."
try {
    # -NoWait: this task should return immediately, not block a scheduler slot
    # for the whole model load.
    & $launcher $Config -NoWait
} catch {
    Write-WatchdogLog "Restart failed: $($_.Exception.Message)"
    exit 1
}
exit 0
