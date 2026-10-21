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
function Get-EnvPort([string]$Name, [int]$Default) {
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if ($fromEnv -match '^\d+$') { return [int]$fromEnv }
    $envFile = Join-Path $FluxRoot ".env"
    if (Test-Path $envFile) {
        $m = Select-String -Path $envFile -Pattern "^$Name=(\d+)" -ErrorAction SilentlyContinue
        if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    }
    return $Default
}

$Port = Get-EnvPort "PORT" 2222
$ComfyPort = Get-EnvPort "COMFY_PORT" 8188
$VideoServerPort = Get-EnvPort "VIDEO_SERVER_PORT" 2224

# netstat, not Get-NetTCPConnection: the latter goes through CIM, which is
# access-denied in some shells here (see kill_flux.ps1).
function Test-PortUp([int]$ListenPort) {
    return [bool](netstat -ano | Select-String ":$ListenPort\s.*LISTENING")
}

function Test-LivePid([string]$PidPath) {
    if (-not (Test-Path $PidPath)) { return $false }
    $recorded = (Get-Content $PidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    return ($recorded -match '^\d+$' -and
            (Get-Process -Id ([int]$recorded) -ErrorAction SilentlyContinue))
}

# A switch in flight is stopping one tenant and starting the other, which looks
# exactly like "both down" for a minute or two. Relaunching into that window is
# how two models end up on one card, so stay out of it entirely.
$switchingFile = Join-Path $FluxRoot ".switching"
if (Test-Path $switchingFile) {
    $age = (Get-Date) - (Get-Item $switchingFile).LastWriteTime
    if ($age.TotalMinutes -lt 15) {
        Write-Verbose "A mode switch has been in flight for $([int]$age.TotalSeconds)s - leaving it alone."
        exit 0
    }
    Write-WatchdogLog "Ignoring a stale .switching ($([int]$age.TotalMinutes) minutes old)."
}

# video_server.py holds no model and belongs up in both modes: it is the UI the
# mode switch is driven from, and in video mode it is the only way back. Both
# supervisors start it, but neither is running when it is the thing that died.
if (-not (Test-PortUp $VideoServerPort) -and (Test-Path (Join-Path $FluxRoot "video_server.py"))) {
    $venvPython = Join-Path $FluxRoot ".venv\Scripts\python.exe"
    if ((Test-Path $venvPython) -and $PSCmdlet.ShouldProcess("video_server.py", "start")) {
        Write-WatchdogLog "Port $VideoServerPort down - starting video_server.py."
        try {
            Start-Process -FilePath $venvPython `
                -ArgumentList "-u", "video_server.py", "--port", "$VideoServerPort" `
                -WorkingDirectory $FluxRoot -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $FluxRoot "video_server.log") `
                -RedirectStandardError (Join-Path $FluxRoot "video_server.err.log")
        } catch {
            Write-WatchdogLog "Could not start video_server.py: $($_.Exception.Message)"
        }
    }
}

# Which GPU tenant is wanted. switch_mode.ps1 is the only writer; absent means
# flux, so a box that has never switched behaves as it always did.
$Mode = "flux"
$modeFile = Join-Path $FluxRoot ".mode"
if (Test-Path $modeFile) {
    $wanted = (Get-Content $modeFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($wanted) { $wanted = $wanted.Trim().ToLower() }
    if ($wanted -eq "video") { $Mode = "video" }
}

if ($Mode -eq "video") {
    # --- video mode: supervise ComfyUI, and never start the generator --------
    if (Test-PortUp $ComfyPort) {
        Write-Verbose "Port $ComfyPort is listening - nothing to do."
        exit 0
    }
    if (Test-Path (Join-Path $FluxRoot ".video_stopped")) {
        Write-Verbose "ComfyUI down, but .video_stopped is present - stopped on purpose, leaving it."
        exit 0
    }
    if (Test-LivePid (Join-Path $FluxRoot "video.pid")) {
        Write-Verbose "run_video.ps1 supervisor is alive - still loading, leaving it alone."
        exit 0
    }
    # Flux resident while .mode says video means someone started it by hand.
    # Its weights are on the card, so starting ComfyUI now is the double-tenant
    # stall. Report rather than fight: a human picked this.
    if (Test-PortUp $Port) {
        Write-WatchdogLog "Mode is 'video' but the generator is up on port $Port - leaving both alone. Run switch_mode.ps1 to settle it."
        exit 0
    }

    $videoLauncher = Join-Path $FluxRoot "run_video.ps1"
    if (-not (Test-Path $videoLauncher)) {
        Write-WatchdogLog "run_video.ps1 not found under $FluxRoot - cannot restart ComfyUI."
        exit 1
    }
    if (-not $PSCmdlet.ShouldProcess("ComfyUI (video mode)", "restart")) { exit 0 }

    Write-WatchdogLog "Port $ComfyPort down and no live supervisor - restarting ComfyUI."
    $shell = $null
    $pwshCmd = Get-Command pwsh.exe -ErrorAction SilentlyContinue
    if ($pwshCmd) { $shell = $pwshCmd.Source } else { $shell = (Get-Command powershell.exe).Source }
    try {
        Start-Process -FilePath $shell `
            -ArgumentList "-NoProfile", "-NoLogo", "-File", $videoLauncher `
            -WorkingDirectory $FluxRoot -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $FluxRoot "video_supervisor.out.log") `
            -RedirectStandardError (Join-Path $FluxRoot "video_supervisor.err.log")
    } catch {
        Write-WatchdogLog "ComfyUI restart failed: $($_.Exception.Message)"
        exit 1
    }
    exit 0
}

# --- flux mode ---------------------------------------------------------------
if (Test-PortUp $Port) {
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

# ComfyUI resident while .mode says flux is the mirror of the case above.
if (Test-PortUp $ComfyPort) {
    Write-WatchdogLog "Mode is 'flux' but ComfyUI is up on port $ComfyPort - not starting a model on top of it. Run switch_mode.ps1 flux."
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
    if ($last -match '^([1-9]|1[0-5])$') { $Config = [int]$last }
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
