# Video mode supervisor: runs ComfyUI (MiniMax H3) under a restart loop, the
# same shape run_server.ps1 gives the image generator.
#
# Usage:  .\run_video.ps1                 start ComfyUI, auto-restart on crash
#         .\run_video.ps1 -Port 8188      bind a different port
#         .\run_video.ps1 -Foreground     (default) blocks; stop with kill_video.ps1
#
# Normally started detached by switch_mode.ps1, not by hand - starting ComfyUI
# while the image generator is still resident is the thing the mode machinery
# exists to prevent, so this script refuses to run when port 2222 is up rather
# than trusting the caller.
#
# COMFY_HOME points at the ComfyUI checkout. It defaults to E:\ComfyUI-h3 (the
# separate, newer checkout H3 needs) and falls back to E:\ComfyUI, which is the
# older install running Wan. H3 wants ComfyUI 0.30+ for the base templates,
# 0.35 for sparse attention and 0.38 for the pruned_w6a8 build, so the two are
# deliberately not the same tree: upgrading the Wan install in place is what
# the second checkout avoids.

param(
    [int]$Port = 0,
    [string]$ComfyHome = "",
    [switch]$NoImageManager
)

Set-Location $PSScriptRoot

$LogFile = Join-Path $PSScriptRoot "comfy.log"
$PidFile = Join-Path $PSScriptRoot "video.pid"
$ModeFile = Join-Path $PSScriptRoot ".mode"
$MaxRetries = 5
$RetryDelay = 3

function Write-Log([string]$Message, [string]$Color = "White") {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line -ForegroundColor $Color
    try { Add-Content -Path $LogFile -Value $line -ErrorAction Stop } catch {}
}

function Get-EnvPort([string]$Name, [int]$Default) {
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if ($fromEnv -match '^\d+$') { return [int]$fromEnv }
    if (Test-Path ".env") {
        $m = Select-String -Path ".env" -Pattern "^$Name=(\d+)" -ErrorAction SilentlyContinue
        if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    }
    return $Default
}

function Test-PortUp([int]$ListenPort) {
    return [bool](netstat -ano | Select-String ":$ListenPort\s.*LISTENING")
}

if ($Port -eq 0) { $Port = Get-EnvPort "COMFY_PORT" 8188 }
$FluxPort = Get-EnvPort "PORT" 2222
$VideoServerPort = Get-EnvPort "VIDEO_SERVER_PORT" 2224

# --- Resolve the checkout ----------------------------------------------------
if (-not $ComfyHome) {
    $ComfyHome = $env:COMFY_HOME
}
if (-not $ComfyHome) {
    foreach ($candidate in @("E:\ComfyUI-h3", "E:\ComfyUI")) {
        if (Test-Path (Join-Path $candidate "main.py")) { $ComfyHome = $candidate; break }
    }
}
if (-not $ComfyHome -or -not (Test-Path (Join-Path $ComfyHome "main.py"))) {
    Write-Log "No ComfyUI checkout found (tried COMFY_HOME, E:\ComfyUI-h3, E:\ComfyUI). Set COMFY_HOME." "Red"
    exit 1
}

# Each checkout owns its venv: the H3 tree needs a newer torch than the Wan
# tree is running, which is the other half of why they are separate.
$ComfyPython = $null
foreach ($rel in @("comfy-venv\Scripts\python.exe", ".venv\Scripts\python.exe", "venv\Scripts\python.exe")) {
    $candidate = Join-Path $ComfyHome $rel
    if (Test-Path $candidate) { $ComfyPython = $candidate; break }
}
if (-not $ComfyPython) {
    Write-Log "No venv python under $ComfyHome (looked for comfy-venv, .venv, venv)." "Red"
    exit 1
}

$version = "unknown"
$versionFile = Join-Path $ComfyHome "comfyui_version.py"
if (Test-Path $versionFile) {
    $m = Select-String -Path $versionFile -Pattern '__version__\s*=\s*"([^"]+)"' -ErrorAction SilentlyContinue
    if ($m) { $version = $m.Matches[0].Groups[1].Value }
}

# --- Refuse to double up on the card ----------------------------------------
if (Test-PortUp $FluxPort) {
    Write-Log "The image generator is listening on port $FluxPort. Both resident at once is the sysmem-fallback stall, so not starting. Use switch_mode.ps1 video." "Red"
    exit 1
}
if (Test-PortUp $Port) {
    Write-Log "Something is already listening on port $Port - ComfyUI may already be up." "Yellow"
    exit 0
}

# --- Environment -------------------------------------------------------------
# Windows RedirectionGuard can refuse to traverse the HF cache's symlinks, so
# store real files (same reason run_server.ps1 sets it).
$env:HF_HUB_DISABLE_SYMLINKS = "1"
# torch ships Intel's OpenMP runtime, which installs a Windows console control
# handler that aborts the process when its console closes or the user logs off
# ("forrtl: error (200)"). That took the generator AND its supervisor down
# together, so the retry loop never ran; the same applies verbatim here.
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
$env:KMP_HANDLE_SIGNALS = "0"

Set-Content -Path $ModeFile -Value "video" -Encoding ascii -ErrorAction SilentlyContinue
# kill_video.ps1 leaves .video_stopped behind so the watchdog does not
# resurrect a deliberate stop; starting again is the intent that cancels it.
Remove-Item (Join-Path $PSScriptRoot ".video_stopped") -Force -ErrorAction SilentlyContinue

# video_server.py (port 2224) is the UI and the only way back to image mode, so
# it has to be up in video mode too. run_server.ps1 ensures it the same way on
# the other side; whichever supervisor is running, one of them has.
function Confirm-VideoServer {
    if (Test-PortUp $VideoServerPort) { return }
    $fluxPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $fluxPython)) {
        Write-Log "No .venv python for video_server.py - the switch-back UI will not be available." "Yellow"
        return
    }
    Write-Log "Starting video_server.py on port $VideoServerPort." "Cyan"
    Start-Process -FilePath $fluxPython `
        -ArgumentList "-u", "video_server.py", "--port", "$VideoServerPort" `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput "video_server.log" -RedirectStandardError "video_server.err.log"
}

Confirm-VideoServer

# --- Supervise ---------------------------------------------------------------
Write-Log "ComfyUI $version at $ComfyHome" "Green"
Write-Log "python: $ComfyPython" "Blue"
Write-Log "Auto-restart enabled (max $MaxRetries retries). Stop with kill_video.ps1." "Yellow"

$PID | Out-File $PidFile -Encoding ascii
$exitCode = 0
try {
    $attempt = 0
    while ($true) {
        $attempt++
        $startTime = Get-Date
        if ($attempt -gt 1) {
            Write-Log "Restart attempt $attempt of $MaxRetries" "Yellow"
        }

        # --listen 0.0.0.0 so a phone on the LAN can reach it, matching how the
        # other two apps bind. -u for unbuffered diagnostics into comfy.log.
        & $ComfyPython -u (Join-Path $ComfyHome "main.py") `
            --port $Port --listen 0.0.0.0 2>&1 |
            Tee-Object -FilePath $LogFile -Append
        $exitCode = $LASTEXITCODE
        $duration = [int]((Get-Date) - $startTime).TotalSeconds

        if ($exitCode -eq 0) {
            Write-Log "ComfyUI exited cleanly." "Green"
            break
        }

        Write-Log "ComfyUI exited with code $exitCode after ${duration}s" "Red"

        # A run that lasted means this was a crash, not a broken install.
        if ($duration -gt 60) {
            Write-Log "Ran for ${duration}s. Resetting retry counter." "Green"
            $attempt = 0
        }

        # Unlike run_server.ps1 there is no fallback config to hop to: there is
        # one video backend. Giving up leaves .mode at 'video', which is
        # correct - the tenant is still the wanted one, it is just down, and
        # watchdog.ps1 will try again on its next pass.
        if ($attempt -ge $MaxRetries -and $attempt -ne 0) {
            Write-Log "Maximum retries ($MaxRetries) reached. Giving up - see comfy.log." "Red"
            $exitCode = 1
            break
        }

        Write-Log "Restarting in ${RetryDelay}s... (attempt $($attempt + 1)/$MaxRetries)" "Red"
        Start-Sleep -Seconds $RetryDelay
    }
} finally {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}
exit $exitCode
