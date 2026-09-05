# Kill all running flux server processes (both the supervisor and the python
# server) - Windows/PowerShell port of sh/kill_flux.sh.
#
# Usage:  .\kill_flux.ps1            stop the generation server (port 2222)
#         .\kill_flux.ps1 -All       also stop the image manager (port 2223)
#         .\kill_flux.ps1 -Verbose   explain each step
#
# Three Windows facts shape this script:
#
# 1. run_server.ps1 is a restart supervisor, so it must die before its python
#    child or the loop simply respawns the server.
# 2. There is no SIGTERM: Stop-Process terminates outright, and a terminated
#    parent does NOT take its children with it (no process groups as in bash).
#    Every process is therefore killed explicitly.
# 3. .venv\Scripts\python.exe is a trampoline - it re-execs the base
#    interpreter (e.g. miniconda's python.exe) as a child, so each server is a
#    *pair* of processes and the one actually holding the port does not match
#    the venv path. Matching on image path alone finds the trampoline and
#    misses the server; matching on the port alone finds the server and leaves
#    an orphaned trampoline. Both halves are found by walking the process tree
#    from whichever half we can identify.
#
#    5200 pwsh (run_server.ps1)
#    |- 20704 .venv\...\python.exe -> 12424 miniconda\python.exe  LISTENING 2222
#    \- 47076 .venv\...\python.exe -> 32412 miniconda\python.exe  LISTENING 2223
#
# That tree is also why the image manager can be spared by default: its chain
# is identified by port 2223 and excluded, rather than guessed at by path.

[CmdletBinding(SupportsShouldProcess)]
param(
    # Also stop image_manager.py (port 2223). Off by default so the gallery
    # survives a server restart, matching what run_server.ps1 assumes.
    [switch]$All
)

Set-Location $PSScriptRoot

$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
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

$Port = Get-EnvPort "PORT" 2222
$ImPort = Get-EnvPort "IMAGE_MANAGER_PORT" 2223

# netstat rather than Get-NetTCPConnection: the latter goes through CIM, which
# is access-denied here for the same reason Win32_Process is.
function Get-PortListeners([int]$ListenPort) {
    return @(netstat -ano | Select-String ":$ListenPort\s.*LISTENING\s+(\d+)" |
        ForEach-Object { [int]$_.Matches[0].Groups[1].Value } |
        Select-Object -Unique)
}

# Command lines are the only way to name a process exactly, but Win32_Process
# can be access-denied depending on how the shell was launched (run_server.ps1
# hits the same wall). Best-effort: when it works we seed from command lines,
# otherwise from listening ports.
function Get-ProcessTable {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object { $_.CommandLine })
    } catch {
        Write-Verbose "Win32_Process unavailable ($($_.Exception.Message)) - seeding from ports instead"
        return $null
    }
}

$AllProcs = @(Get-Process -ErrorAction SilentlyContinue)

function Get-ParentOf($Proc) {
    # Process.Parent (PowerShell 6+) reads the parent without WMI. Guard against
    # PID reuse: a recycled parent PID would have started after its "child".
    $parent = $null
    try { $parent = $Proc.Parent } catch { return $null }
    if (-not $parent) { return $null }
    try { if ($parent.StartTime -gt $Proc.StartTime) { return $null } } catch {}
    return $parent
}

# Both halves of a trampoline pair, from either half: walk up through python
# ancestors (stopping at the pwsh supervisor), then back down through python
# descendants. Returns the whole python chain a seed PID belongs to.
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
        # -WhatIf turns the whole script into a dry run that reports the process
        # tree it worked out without touching it.
        if (-not $PSCmdlet.ShouldProcess("$($proc.ProcessName) PID $id", "Stop $What")) { continue }
        try {
            Stop-Process -Id $id -Force -ErrorAction Stop
            $stopped += $id
        } catch {
            # Killing a trampoline usually takes its interpreter child with it,
            # so a PID can vanish between the check above and the kill. That is
            # the outcome we wanted, not a failure.
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

$procs = Get-ProcessTable

# --- Work out what belongs to whom before killing anything --------------------
# The image manager's chain has to be known up front: once its processes are
# dead there is nothing left to tell them apart from the server's.
$imSeeds = if ($procs) {
    @($procs | Where-Object { $_.CommandLine -match 'image_manager\.py' } |
        ForEach-Object { [int]$_.ProcessId })
} else {
    Get-PortListeners $ImPort
}
$imChain = Expand-Chains $imSeeds
Write-Verbose "image_manager.py chain (port ${ImPort}): $($imChain -join ', ')"

$serverSeeds = if ($procs) {
    @($procs | Where-Object { $_.CommandLine -match 'web_server\.py' } |
        ForEach-Object { [int]$_.ProcessId })
} else {
    Get-PortListeners $Port
}
$serverChain = Expand-Chains $serverSeeds
Write-Verbose "web_server.py chain (port ${Port}): $($serverChain -join ', ')"

# A server killed mid-startup has not bound the port yet, so also sweep venv
# pythons that belong to no known chain - minus the manager's, which is the
# one venv python we deliberately leave alone.
$stragglers = @(Expand-Chains @($AllProcs |
    Where-Object { $_.ProcessName -like 'python*' -and $_.Path -eq $PythonExe } |
    ForEach-Object { $_.Id }) | Where-Object { $imChain -notcontains $_ })
if ($stragglers) { Write-Verbose "venv python stragglers: $($stragglers -join ', ')" }

$targets = @(@($serverChain) + $stragglers | Select-Object -Unique)
if ($All) { $targets = @($targets + $imChain | Select-Object -Unique) }

# Supervisors: the non-python parent of any chain we are about to kill, plus
# whatever server.pid records. Killing these first stops the restart loop.
$supervisors = @()
foreach ($id in $targets) {
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    $parent = Get-ParentOf $proc
    if ($parent -and $parent.ProcessName -match '^(pwsh|powershell)$') { $supervisors += $parent.Id }
}

Write-Verbose "Checking for server.pid file..."
if (Test-Path "server.pid") {
    $recorded = (Get-Content "server.pid" -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($recorded -match '^\d+$' -and (Get-Process -Id ([int]$recorded) -ErrorAction SilentlyContinue)) {
        Write-Verbose "PID $recorded is alive - this is the run_server.ps1 restart loop"
        $supervisors += [int]$recorded
    } else {
        Write-Verbose "PID $recorded is no longer running - stale pid file"
    }
    Remove-Item "server.pid" -Force -ErrorAction SilentlyContinue
    Write-Verbose "Removed server.pid"
} else {
    Write-Verbose "No server.pid file found"
}

if ($procs) {
    $supervisors += @($procs | Where-Object { $_.CommandLine -match 'run_server\.ps1' } |
        ForEach-Object { [int]$_.ProcessId })
}

# The FluxServerWatchdog scheduled task (install_watchdog.ps1) restarts the
# server whenever it finds port $Port down, which would undo this script within
# minutes. Drop a sentinel so a deliberate stop stays stopped; run_server.ps1
# clears it on the next start, so nothing has to be un-done by hand.
if (-not $WhatIfPreference) {
    Set-Content -Path ".flux_stopped" -Value "stopped by kill_flux.ps1 at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" `
        -Encoding ascii -ErrorAction SilentlyContinue
    Write-Verbose "Wrote .flux_stopped - the watchdog will leave the server down"
}

# --- Kill: supervisors first, then the python they would otherwise respawn ----
Stop-Pids $supervisors "supervisor" | Out-Null
Stop-Pids $targets "flux server" | Out-Null

# --- Verify -------------------------------------------------------------------
if ($WhatIfPreference) {
    Write-Host "Dry run - nothing was killed." -ForegroundColor Cyan
    exit 0
}

if (-not $killed) {
    Write-Host "No flux server processes found." -ForegroundColor Green
    exit 0
}

Start-Sleep -Seconds 1
Write-Verbose "Re-checking for survivors..."
$remaining = Get-PortListeners $Port
if ($All) { $remaining += Get-PortListeners $ImPort }
$remaining = @($remaining | Where-Object { $_ -ne $PID } | Select-Object -Unique)

if ($remaining) {
    Write-Host "Still listening after kill: $($remaining -join ', ') - retrying" -ForegroundColor Red
    Stop-Pids $remaining "survivor" | Out-Null
    Start-Sleep -Seconds 1
    $stubborn = @(Get-PortListeners $Port | Where-Object { $_ -ne $PID })
    if ($stubborn) {
        Write-Host "Could not stop: $($stubborn -join ', '). Try an elevated shell." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Verbose "All processes exited"
}

Write-Host "Done." -ForegroundColor Green
exit 0
