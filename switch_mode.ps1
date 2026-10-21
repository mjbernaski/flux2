# Switch the box between its two GPU tenants: the image generator (web_server.py
# on 2222, via run_server.ps1) and MiniMax H3 video (ComfyUI on 8188, via
# run_video.ps1). Exactly one of them may be resident, so switching is
# stop-one-then-start-the-other and never both at once.
#
# Usage:  .\switch_mode.ps1 video     stop flux, start ComfyUI + H3
#         .\switch_mode.ps1 flux      stop ComfyUI, bring the generator back
#         .\switch_mode.ps1 flux -Config 9   ... on a specific config
#         .\switch_mode.ps1 video -WhatIf    report the plan, change nothing
#
# This script is the ONLY writer of .mode, which records which tenant is
# wanted. Everything else reads it: watchdog.ps1 decides what to resurrect from
# it, run_server.ps1 refuses to start against it, and video_server.py reports
# it. Keeping one writer is what stops "wanted" and "running" from disagreeing
# in two places at once.
#
# Normally invoked by video_server.py (POST /mode), which runs it detached and
# returns immediately, because a switch takes minutes: a process stop, a wait
# for the driver to actually release the VRAM, then a cold model load.
#
# Why the VRAM wait is not optional: a process can be gone from the process
# table while the driver still holds its allocations for a moment. Starting the
# next tenant into that window is how both end up resident, and on Windows that
# does not fail - WDDM silently pages one of them to system RAM and every step
# runs 8-200x slow with no error anywhere (see the sysmem-fallback note in
# CLAUDE.md). A slow start is strictly better than that.

[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('flux', 'video')]
    [string]$Mode,

    # Which image config to come back on. Defaults to .last_config, the same
    # file watchdog.ps1 reads, so a deliberate model choice survives a round
    # trip through video mode.
    [ValidateRange(1, 15)]
    [int]$Config = 0,

    # Seconds to wait for the outgoing tenant's VRAM to be released.
    [int]$FreeTimeoutSec = 120,

    # Seconds to wait for the incoming tenant to bind its port. A cold H3 load
    # reads ~32GB off disk; a 32B image config can be slower still.
    [int]$UpTimeoutSec = 900
)

Set-Location $PSScriptRoot

$ModeFile = Join-Path $PSScriptRoot ".mode"
$SwitchingFile = Join-Path $PSScriptRoot ".switching"
$LogFile = Join-Path $PSScriptRoot "switch_mode.log"
$LastConfigFile = Join-Path $PSScriptRoot ".last_config"

# VRAM considered "released". Not zero: the desktop compositor and any browser
# hold a few hundred MB on this card at all times, and ollama may hold more.
$FreeBelowMiB = 4096

function Write-SwitchLog([string]$Message, [string]$Color = "White") {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Mode] $Message"
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

$FluxPort = Get-EnvPort "PORT" 2222
$ComfyPort = Get-EnvPort "COMFY_PORT" 8188

# netstat, not Get-NetTCPConnection: the latter goes through CIM, which is
# access-denied in some shells here (see kill_flux.ps1).
function Test-PortUp([int]$ListenPort) {
    return [bool](netstat -ano | Select-String ":$ListenPort\s.*LISTENING")
}

function Get-UsedVramMiB {
    # Returns $null when nvidia-smi is missing or unreadable, which makes the
    # VRAM wait a best-effort check rather than a hard gate - a box without it
    # should still be able to switch.
    try {
        $out = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        $first = @($out)[0]
        if ($first -match '(\d+)') { return [int]$Matches[1] }
    } catch {}
    return $null
}

function Wait-PortDown([int]$ListenPort, [int]$TimeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-PortUp $ListenPort)) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Wait-PortUp([int]$ListenPort, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortUp $ListenPort) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Wait-VramFree([int]$TimeoutSec) {
    $used = Get-UsedVramMiB
    if ($null -eq $used) {
        Write-SwitchLog "nvidia-smi unavailable - skipping the VRAM wait." "Yellow"
        Start-Sleep -Seconds 5
        return $true
    }
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $used = Get-UsedVramMiB
        if ($null -eq $used -or $used -lt $FreeBelowMiB) {
            Write-SwitchLog "VRAM down to ${used}MiB - the card is free." "Green"
            return $true
        }
        Start-Sleep -Seconds 2
    }
    Write-SwitchLog "VRAM still at ${used}MiB after ${TimeoutSec}s. Something else is holding it (ollama? LM Studio? a stuck python?). Starting anyway would risk the sysmem-fallback stall, so stopping here." "Red"
    return $false
}

# --- Guard: one switch at a time -------------------------------------------
# .switching is also what video_server.py reports as "in flight", so it is
# written before anything is stopped and removed in the finally below whatever
# happens. A crashed switch leaves it behind; video_server treats one older
# than 15 minutes as stale rather than blocking forever.
if (Test-Path $SwitchingFile) {
    $age = (Get-Date) - (Get-Item $SwitchingFile).LastWriteTime
    if ($age.TotalMinutes -lt 15) {
        Write-SwitchLog "Another switch has been in flight for $([int]$age.TotalSeconds)s - refusing to start a second one." "Red"
        exit 1
    }
    Write-SwitchLog "Clearing a stale .switching ($([int]$age.TotalMinutes) minutes old)." "Yellow"
}

if (-not $PSCmdlet.ShouldProcess("GPU tenant", "switch to $Mode")) {
    Write-Host "Would switch to $Mode." -ForegroundColor Cyan
    exit 0
}

Set-Content -Path $SwitchingFile -Value "$Mode $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" `
    -Encoding ascii -ErrorAction SilentlyContinue

# The body is a function so that every early exit is a `return` the finally
# below is guaranteed to see. A bare `exit` here would be betting on whether
# PowerShell runs finally blocks on script exit, and leaving .switching behind
# blocks the next switch for fifteen minutes.
function Invoke-Switch {
    if ($Mode -eq 'video') {
        # --- flux -> video ---------------------------------------------------
        Write-SwitchLog "Stopping the image generator (port $FluxPort)." "Cyan"
        # Not -All: image_manager (2223) and video_server (2224) hold no VRAM
        # and must survive, the second one especially - it is the page the
        # switch back is asked for from. kill_flux.ps1 spares both by port.
        & (Join-Path $PSScriptRoot "kill_flux.ps1")

        if (-not (Wait-PortDown $FluxPort 60)) {
            Write-SwitchLog "Port $FluxPort is still listening - aborting rather than running both." "Red"
            return 1
        }
        if (-not (Wait-VramFree $FreeTimeoutSec)) { return 1 }

        # Written only once the card is actually free, so .mode never claims a
        # tenant that has not been given room yet.
        Set-Content -Path $ModeFile -Value "video" -Encoding ascii
        Write-SwitchLog "Mode is now 'video'. Starting ComfyUI." "Green"

        $launcher = Join-Path $PSScriptRoot "run_video.ps1"
        if (-not (Test-Path $launcher)) {
            Write-SwitchLog "run_video.ps1 not found - nothing to start." "Red"
            return 1
        }
        $shell = $null
        $pwshCmd = Get-Command pwsh.exe -ErrorAction SilentlyContinue
        if ($pwshCmd) { $shell = $pwshCmd.Source } else { $shell = (Get-Command powershell.exe).Source }
        # Detached and hidden, for the same reason f14.ps1 does it: run_video.ps1
        # is a supervisor that blocks for the life of ComfyUI, and torch's Intel
        # OpenMP console handler aborts the process when its console closes.
        $proc = Start-Process -FilePath $shell `
            -ArgumentList "-NoProfile", "-NoLogo", "-File", $launcher `
            -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $PSScriptRoot "video_supervisor.out.log") `
            -RedirectStandardError  (Join-Path $PSScriptRoot "video_supervisor.err.log")
        Write-SwitchLog "ComfyUI supervisor started detached (PID $($proc.Id))."

        if (Wait-PortUp $ComfyPort $UpTimeoutSec) {
            Write-SwitchLog "ComfyUI is up on port $ComfyPort." "Green"
        } else {
            Write-SwitchLog "ComfyUI did not bind port $ComfyPort within ${UpTimeoutSec}s - check comfy.log." "Yellow"
            return 1
        }
    } else {
        # --- video -> flux ---------------------------------------------------
        Write-SwitchLog "Stopping ComfyUI (port $ComfyPort)." "Cyan"
        $killer = Join-Path $PSScriptRoot "kill_video.ps1"
        if (Test-Path $killer) { & $killer } else {
            Write-SwitchLog "kill_video.ps1 not found - cannot stop ComfyUI cleanly." "Red"
            return 1
        }

        if (-not (Wait-PortDown $ComfyPort 60)) {
            Write-SwitchLog "Port $ComfyPort is still listening - aborting rather than running both." "Red"
            return 1
        }
        if (-not (Wait-VramFree $FreeTimeoutSec)) { return 1 }

        Set-Content -Path $ModeFile -Value "flux" -Encoding ascii

        if ($Config -eq 0) {
            $Config = 9
            if (Test-Path $LastConfigFile) {
                $last = (Get-Content $LastConfigFile -ErrorAction SilentlyContinue | Select-Object -First 1)
                if ($last -match '^([1-9]|1[0-5])$') { $Config = [int]$last }
            }
        }
        Write-SwitchLog "Mode is now 'flux'. Starting config $Config." "Green"

        # f14.ps1 is the detached launcher; run_server.ps1 clears .flux_stopped
        # on start, so the kill_flux sentinel from the other direction is
        # cancelled by the act of starting rather than by being cleaned up here.
        $launcher = Join-Path $PSScriptRoot "f14.ps1"
        if (-not (Test-Path $launcher)) {
            Write-SwitchLog "f14.ps1 not found - cannot restart the generator." "Red"
            return 1
        }
        & $launcher $Config -NoWait

        if (Wait-PortUp $FluxPort $UpTimeoutSec) {
            Write-SwitchLog "Image generator is up on port $FluxPort." "Green"
        } else {
            Write-SwitchLog "Port $FluxPort not up within ${UpTimeoutSec}s - check server.log." "Yellow"
            return 1
        }
    }
    return 0
}

$exitCode = 1
try {
    $exitCode = Invoke-Switch
} catch {
    Write-SwitchLog "Switch failed: $($_.Exception.Message)" "Red"
    $exitCode = 1
} finally {
    Remove-Item $SwitchingFile -Force -ErrorAction SilentlyContinue
}

exit $exitCode
