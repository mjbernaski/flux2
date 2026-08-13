# Compact UI launcher (Windows) - serves static-v2/ on port 3333 and proxies
# /api/v1 through to the generation server on 2222.
# Usage:  .\run_new_ui.ps1 [extra new_ui.py args]
#
# Binds 0.0.0.0, so it is reachable over the LAN once the firewall rule from
# enable_wan_access.ps1 exists (it adds TCP 3333 alongside 2222/2223).

Set-Location $PSScriptRoot

$Port = if ($env:FLUX_UI_PORT) { [int]$env:FLUX_UI_PORT } else { 3333 }
$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }

# Kill any prior instance. Find it by its listening port rather than by
# matching the script name - CIM/WMI command-line queries are access-denied in
# some shells here, and killing all venv pythons would take down web_server.py.
$listeners = netstat -ano | Select-String ":$Port\s.*LISTENING\s+(\d+)" |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Unique
foreach ($listenerPid in $listeners) {
    Write-Host "Killing existing UI on port ${Port}: PID $listenerPid" -ForegroundColor Yellow
    try { Stop-Process -Id $listenerPid -Force -ErrorAction Stop } catch {}
}
if ($listeners) { Start-Sleep -Seconds 1 }

$lan = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1 -ExpandProperty IPAddress)
if ($lan) { Write-Host "LAN URL: http://${lan}:$Port" -ForegroundColor Green }

& $PythonExe new_ui.py --port $Port @args
exit $LASTEXITCODE
