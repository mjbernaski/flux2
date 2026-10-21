# Stop ComfyUI and its supervisor - the video-mode counterpart of
# kill_flux.ps1.
#
# Usage:  .\kill_video.ps1            stop ComfyUI (port 8188)
#         .\kill_video.ps1 -All       also stop video_server.py (port 2224)
#         .\kill_video.ps1 -Verbose   explain each step
#         .\kill_video.ps1 -WhatIf    report the plan, kill nothing
#
# Same two Windows facts as kill_flux.ps1 shape this: run_video.ps1 is a
# restart supervisor, so it must die before its python child or the loop
# respawns ComfyUI; and a terminated parent does not take its children with it,
# so every process is killed explicitly.
#
# video_server.py is spared by default and that is the point - it holds no VRAM
# and it is the page the switch back to image mode is asked for from. Killing
# it leaves the box in video mode with no UI to leave video mode.

[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$All
)

Set-Location $PSScriptRoot

$PidFile = Join-Path $PSScriptRoot "video.pid"
$killed = $false

function Get-EnvPort([string]$Name, [int]$Default) {
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if ($fromEnv -match '^\d+$') { return [int]$fromEnv }
    if (Test-Path ".env") {
        $m = Select-String -Path ".env" -Pattern "^$Name=(\d+)" -ErrorAction SilentlyContinue
        if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    }
    return $Default
}

$Port = Get-EnvPort "COMFY_PORT" 8188
$VideoServerPort = Get-EnvPort "VIDEO_SERVER_PORT" 2224

# netstat rather than Get-NetTCPConnection: the latter goes through CIM, which
# is access-denied in some shells here.
function Get-PortListeners([int]$ListenPort) {
    return @(netstat -ano | Select-String ":$ListenPort\s.*LISTENING\s+(\d+)" |
        ForEach-Object { [int]$_.Matches[0].Groups[1].Value } |
        Select-Object -Unique)
}

$AllProcs = @(Get-Process -ErrorAction SilentlyContinue)

function Get-ParentOf($Proc) {
    # Process.Parent (PowerShell 6+) reads the parent without WMI. Guard
    # against PID reuse: a recycled parent would have started after its child.
    $parent = $null
    try { $parent = $Proc.Parent } catch { return $null }
    if (-not $parent) { return $null }
    try { if ($parent.StartTime -gt $Proc.StartTime) { return $null } } catch {}
    return $parent
}

# ComfyUI may itself be a python-under-python pair depending on how the venv
# was made, so collect the whole python chain a seed PID belongs to - up
# through python ancestors, then back down through python descendants.
function Get-PythonChain([int]$Seed) {
    $proc = Get-Process -Id $Seed -ErrorAction SilentlyContinue
    if (-not $proc) { return @() }

    $root = $proc
    while ($true) {
        $parent = Get-ParentOf $root
        if (-not $parent -or $parent.ProcessName -notlike 'python*') { break }
        $root = $parent
    }

    $chain = [System.Collections.Generic.List[int]]::new()
    $queue = [System.Collections.Generic.Queue[int]]::new()
    $queue.Enqueue($root.Id)
    while ($queue.Count -gt 0) {
        $id = $queue.Dequeue()
        if ($chain.Contains($id)) { continue }
        $chain.Add($id)
        foreach ($child in $AllProcs) {
            if ($child.ProcessName -notlike 'python*') { continue }
            $parent = Get-ParentOf $child
            if ($parent -and $parent.Id -eq $id) { $queue.Enqueue($child.Id) }
        }
    }
    return $chain.ToArray()
}

function Expand-Chains([int[]]$Seeds) {
    $out = @()
    foreach ($seed in ($Seeds | Where-Object { $_ } | Select-Object -Unique)) {
        $out += Get-PythonChain $seed
    }
    return @($out | Select-Object -Unique)
}

function Stop-Pids([int[]]$Ids, [string]$What) {
    $stopped = @()
    foreach ($id in ($Ids | Where-Object { $_ -and $_ -ne $PID } | Select-Object -Unique)) {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        if (-not $PSCmdlet.ShouldProcess("$($proc.ProcessName) PID $id", "Stop $What")) { continue }
        try {
            Stop-Process -Id $id -Force -ErrorAction Stop
            $stopped += $id
        } catch {
            # A PID can vanish between the check and the kill when its parent
            # took it down first. That is the outcome we wanted.
            if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
                Write-Host "Could not kill ${What} PID ${id}: $($_.Exception.Message)" -ForegroundColor Red
            } else {
                Write-Verbose "PID $id already exited with its parent"
            }
        }
    }
    if ($stopped) {
        Write-Host "Killed ${What}: $($stopped -join ', ')" -ForegroundColor Yellow
        $script:killed = $true
    }
    return $stopped
}

# --- Work out the targets before killing anything ---------------------------
# The video server's chain is excluded by port, the same way kill_flux.ps1
# spares the image manager, and it has to be known before anything dies.
$keepChain = @()
if (-not $All) {
    $keepChain = Expand-Chains (Get-PortListeners $VideoServerPort)
    if ($keepChain) { Write-Verbose "sparing video_server.py chain (port ${VideoServerPort}): $($keepChain -join ', ')" }
}

$comfyChain = @(Expand-Chains (Get-PortListeners $Port) | Where-Object { $keepChain -notcontains $_ })
Write-Verbose "ComfyUI chain (port ${Port}): $($comfyChain -join ', ')"

# Supervisors: run_video.ps1 records its own PID, plus the non-python parent of
# anything we are about to kill. These die first or the loop respawns ComfyUI.
$supervisors = @()
if (Test-Path $PidFile) {
    $recorded = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($recorded -match '^\d+$' -and (Get-Process -Id ([int]$recorded) -ErrorAction SilentlyContinue)) {
        Write-Verbose "PID $recorded is alive - the run_video.ps1 restart loop"
        $supervisors += [int]$recorded
    } else {
        Write-Verbose "PID $recorded is no longer running - stale pid file"
    }
    if (-not $WhatIfPreference) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
}

foreach ($id in $comfyChain) {
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    $parent = Get-ParentOf $proc
    if ($parent -and $parent.ProcessName -match '^(pwsh|powershell)$') { $supervisors += $parent.Id }
}

$targets = $comfyChain
if ($All) { $targets = @($targets + (Expand-Chains (Get-PortListeners $VideoServerPort)) | Select-Object -Unique) }

# The video-mode half of .flux_stopped: watchdog.ps1 restarts whichever tenant
# .mode names, so a deliberate stop of ComfyUI needs its own way of saying
# "down on purpose" or the next watchdog pass brings it straight back.
# run_video.ps1 clears it on the next start, so nothing has to be undone by
# hand. Note .flux_stopped does NOT cover this: switch_mode.ps1 writes that one
# on its way *into* video mode, where flux being down is the whole point.
if (-not $WhatIfPreference) {
    Set-Content -Path ".video_stopped" -Value "stopped by kill_video.ps1 at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" `
        -Encoding ascii -ErrorAction SilentlyContinue
    Write-Verbose "Wrote .video_stopped - the watchdog will leave ComfyUI down"
}

Stop-Pids $supervisors "video supervisor" | Out-Null
Stop-Pids $targets "ComfyUI" | Out-Null

if ($WhatIfPreference) {
    Write-Host "Dry run - nothing was killed." -ForegroundColor Cyan
    exit 0
}

if (-not $killed) {
    Write-Host "No ComfyUI processes found." -ForegroundColor Green
    exit 0
}

Start-Sleep -Seconds 1
$remaining = @(Get-PortListeners $Port | Where-Object { $_ -ne $PID })
if ($remaining) {
    Write-Host "Still listening after kill: $($remaining -join ', ') - retrying" -ForegroundColor Red
    Stop-Pids $remaining "survivor" | Out-Null
    Start-Sleep -Seconds 1
    $stubborn = @(Get-PortListeners $Port | Where-Object { $_ -ne $PID })
    if ($stubborn) {
        Write-Host "Could not stop: $($stubborn -join ', '). Try an elevated shell." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Done." -ForegroundColor Green
exit 0
