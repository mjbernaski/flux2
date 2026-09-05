<#
.SYNOPSIS
    Undo a deliberate FLUX server stop - re-arm the watchdog and bring the server back.

.DESCRIPTION
    Companion to kill_flux.ps1. A deliberate stop leaves two marks on this box:

      1. .flux_stopped   - the sentinel kill_flux.ps1 writes so FluxServerWatchdog
                           leaves a stopped server stopped (watchdog.ps1 checks it).
      2. FluxServerWatchdog disabled - belt-and-braces, only if the stop disabled it.

    This script clears both and relaunches on whatever config was last run
    (.last_config, written by run_server.ps1), so a deliberate model switch survives.

    run_server.ps1 also removes .flux_stopped on start, so the sentinel is not the
    thing that has to be undone by hand - the watchdog being disabled is. Running
    this is idempotent: if the server is already listening it only re-arms the
    watchdog and returns.

.PARAMETER Config
    Menu number from SERVER_OPTIONS.md. Defaults to .last_config (currently 9).

.PARAMETER NoStart
    Only re-arm: clear the sentinel and enable the task, then let the watchdog pick
    the server up on its next tick (every few minutes). Nothing is launched here.

.PARAMETER TimeoutSec
    Seconds to wait for port 2222 after launching. Default 300 - a cold HF cache on
    the 32B configs takes minutes.

.EXAMPLE
    E:\flux2\restore_flux.ps1
    Re-arm the watchdog and start the last-used config, waiting for the port.

.EXAMPLE
    E:\flux2\restore_flux.ps1 -NoStart
    Re-arm only; the watchdog restarts it within a few minutes.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateRange(1, 14)]
    [int]$Config,

    [switch]$NoStart,

    [int]$TimeoutSec = 300
)

$ErrorActionPreference = 'Stop'
$TaskName = 'FluxServerWatchdog'

$FluxRoot = if ($env:FLUX_HOME) { $env:FLUX_HOME }
            elseif ($PSScriptRoot) { $PSScriptRoot }
            else { 'E:\flux2' }

if (-not (Test-Path $FluxRoot)) {
    Write-Host "FLUX root '$FluxRoot' not found - set FLUX_HOME to the checkout." -ForegroundColor Red
    exit 1
}

# Same port resolution as run_server.ps1 / kill_flux.ps1 / watchdog.ps1.
$Port = 2222
$envFile = Join-Path $FluxRoot '.env'
if ($env:PORT -match '^\d+$') {
    $Port = [int]$env:PORT
} elseif (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern '^PORT=(\d+)' -ErrorAction SilentlyContinue
    if ($m) { $Port = [int]$m.Matches[0].Groups[1].Value }
}

# netstat, not Get-NetTCPConnection: the latter goes through CIM, which is
# access-denied in some shells on this box (see kill_flux.ps1's note).
function Test-PortUp([int]$ListenPort) {
    return [bool](netstat -ano | Select-String ":$ListenPort\s.*LISTENING")
}

# --- 1. Re-arm the watchdog ---------------------------------------------------
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "Scheduled task '$TaskName' is not registered." -ForegroundColor Yellow
    Write-Host "Re-create it with:  & '$FluxRoot\install_watchdog.ps1'" -ForegroundColor Yellow
} elseif ($task.State -eq 'Disabled') {
    if ($PSCmdlet.ShouldProcess($TaskName, 'enable scheduled task')) {
        # Enabling needs the same rights that disabling did; surface it rather than
        # failing silently half-way through the restore.
        try {
            Enable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
            Write-Host "Re-enabled scheduled task '$TaskName'." -ForegroundColor Green
        } catch {
            Write-Host "Could not enable '$TaskName': $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Re-run this script from an elevated PowerShell." -ForegroundColor Yellow
        }
    }
} else {
    Write-Verbose "Scheduled task '$TaskName' is already $($task.State) - nothing to enable."
}

# --- 2. Clear the deliberate-stop sentinel ------------------------------------
$stopFile = Join-Path $FluxRoot '.flux_stopped'
if (Test-Path $stopFile) {
    if ($PSCmdlet.ShouldProcess($stopFile, 'remove stop sentinel')) {
        Remove-Item $stopFile -Force -ErrorAction SilentlyContinue
        Write-Host "Cleared .flux_stopped - the watchdog will guard the server again." -ForegroundColor Green
    }
} else {
    Write-Verbose "No .flux_stopped sentinel - the stop was not marked deliberate."
}

# --- 3. Bring the server back -------------------------------------------------
if (Test-PortUp $Port) {
    Write-Host "FLUX server already listening on port $Port - nothing to start." -ForegroundColor Green
    exit 0
}

if ($NoStart) {
    Write-Host "Re-armed only. FluxServerWatchdog will restart the server on its next tick." -ForegroundColor Cyan
    exit 0
}

# Come back on the config that was last run, matching watchdog.ps1's behaviour, so
# a deliberate model switch is not silently reverted to the default.
if (-not $PSBoundParameters.ContainsKey('Config')) {
    $Config = 9
    $lastConfigFile = Join-Path $FluxRoot '.last_config'
    if (Test-Path $lastConfigFile) {
        $last = (Get-Content $lastConfigFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($last -match '^([1-9]|1[0-4])$') { $Config = [int]$last }
    }
}

$launcher = Join-Path $FluxRoot 'f14.ps1'
if (-not (Test-Path $launcher)) {
    Write-Host "f14.ps1 not found under $FluxRoot - cannot start." -ForegroundColor Red
    exit 1
}

if (-not $PSCmdlet.ShouldProcess("FLUX server config $Config", 'start')) { exit 0 }

Write-Host "Starting FLUX server, config $Config (waiting up to ${TimeoutSec}s for port $Port)..." -ForegroundColor Cyan
& $launcher $Config -TimeoutSec $TimeoutSec

if (Test-PortUp $Port) {
    Write-Host "FLUX server is up on port $Port." -ForegroundColor Green
    exit 0
}

Write-Host "Port $Port is still down. Check $FluxRoot\server.log." -ForegroundColor Red
exit 1
