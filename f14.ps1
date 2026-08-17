# Start the FLUX server, config 14 (FLUX.2-klein-4B, NF4 encoder), detached
# from the calling terminal - runnable from any directory on this PC.
#
# Usage:  f14                    launch config 14, wait until the port is up
#         f14 9                  launch a different config instead
#         f14 -NoWait            fire and forget, don't wait for the port
#         f14 -Foreground        run the supervisor in this window (Ctrl+C stops it)
#
# Why Start-Process rather than just calling run_server.ps1: run_server.ps1 is a
# restart supervisor that blocks for the life of the server, so invoking it
# directly ties the server to whatever shell you typed it in. Start-Process with
# a hidden window gives the supervisor its own console-less process, which
# survives the launching terminal closing. Its stdout/stderr are redirected
# because a hidden process has nowhere else to put them - the server's own
# output still goes to server.log via the supervisor's Tee-Object.
#
# Stop it with:  E:\flux2\kill_flux.ps1

param(
    # Menu number from SERVER_OPTIONS.md / run_server.ps1's Get-ConfigArgs.
    [ValidateRange(1, 14)]
    [int]$Config = 14,

    # Return as soon as the supervisor is spawned, without waiting for the
    # server to bind its port (model load is ~10s for klein-4B, minutes for the
    # 32B configs on a cold HF cache).
    [switch]$NoWait,

    # Run the supervisor attached to this window instead - the plain
    # `run_server.ps1 <n>` behaviour, kept here so one command covers both.
    [switch]$Foreground,

    # Seconds to wait for the port when not -NoWait.
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"

# FLUX_HOME lets a checkout somewhere else reuse this script; the literal path
# is what makes it work from any CWD.
$FluxRoot = if ($env:FLUX_HOME) { $env:FLUX_HOME } else { "E:\flux2" }
$Launcher = Join-Path $FluxRoot "run_server.ps1"

if (-not (Test-Path $Launcher)) {
    Write-Host "run_server.ps1 not found under $FluxRoot - set FLUX_HOME to the checkout." -ForegroundColor Red
    exit 1
}

# Same port resolution as run_server.ps1 and kill_flux.ps1.
$Port = 2222
if ($env:PORT -match '^\d+$') {
    $Port = [int]$env:PORT
} elseif (Test-Path (Join-Path $FluxRoot ".env")) {
    $m = Select-String -Path (Join-Path $FluxRoot ".env") -Pattern '^PORT=(\d+)' -ErrorAction SilentlyContinue
    if ($m) { $Port = [int]$m.Matches[0].Groups[1].Value }
}

function Test-PortUp([int]$ListenPort) {
    # netstat, not Get-NetTCPConnection: the latter goes through CIM, which is
    # access-denied in some shells here (see kill_flux.ps1).
    return [bool](netstat -ano | Select-String ":$ListenPort\s.*LISTENING")
}

if ($Foreground) {
    & $Launcher $Config
    exit $LASTEXITCODE
}

# PowerShell 7 if it is installed, Windows PowerShell otherwise. run_server.ps1
# works under both. Written the long way so this script still parses under
# 5.1, where `?.` is a syntax error.
$Shell = $null
$pwshCmd = Get-Command pwsh.exe -ErrorAction SilentlyContinue
if ($pwshCmd) { $Shell = $pwshCmd.Source } else { $Shell = (Get-Command powershell.exe).Source }

if (Test-PortUp $Port) {
    Write-Host "Something is already listening on port $Port - run_server.ps1 will stop it first." -ForegroundColor Yellow
}

$proc = Start-Process -FilePath $Shell `
    -ArgumentList "-NoProfile", "-NoLogo", "-File", $Launcher, "$Config" `
    -WorkingDirectory $FluxRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $FluxRoot "supervisor.out.log") `
    -RedirectStandardError  (Join-Path $FluxRoot "supervisor.err.log")

Write-Host "Supervisor started detached (PID $($proc.Id)), config $Config." -ForegroundColor Green
Write-Host "Logs: $FluxRoot\server.log  |  Stop: $FluxRoot\kill_flux.ps1" -ForegroundColor Cyan

if ($NoWait) { exit 0 }

Write-Host -NoNewline "Waiting for port $Port "
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    if (Test-PortUp $Port) {
        Write-Host ""
        Write-Host "Server is up: http://localhost:$Port/" -ForegroundColor Green
        exit 0
    }
    # A dead supervisor means the server failed to start (bad venv, OOM, retries
    # exhausted); no point waiting out the full timeout.
    if ($proc.HasExited) {
        Write-Host ""
        Write-Host "Supervisor exited with code $($proc.ExitCode) - last lines of server.log:" -ForegroundColor Red
        Get-Content (Join-Path $FluxRoot "server.log") -Tail 20 -ErrorAction SilentlyContinue
        exit 1
    }
    Start-Sleep -Seconds 2
    Write-Host -NoNewline "."
}

Write-Host ""
Write-Host "Port $Port not up after ${TimeoutSec}s - still loading? Check $FluxRoot\server.log" -ForegroundColor Yellow
exit 1
