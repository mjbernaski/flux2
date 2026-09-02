# Image Manager launcher (Windows) - PowerShell port of sh/run_image_manager.sh
# Browse/move/delete/crop images under web-generated/ on port 2223.
# Usage:  .\run_image_manager.ps1 [extra image_manager.py args]

Set-Location $PSScriptRoot

$Port = if ($env:IMAGE_MANAGER_PORT) { [int]$env:IMAGE_MANAGER_PORT } else { 2223 }
$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }

# Kill any prior instance. CIM/WMI command-line queries can be access-denied,
# so find it by its listening port instead of matching "image_manager.py".
# Do NOT kill all venv pythons here - that would take down web_server.py too.
$listeners = netstat -ano | Select-String ":$Port\s.*LISTENING\s+(\d+)" |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Unique
foreach ($listenerPid in $listeners) {
    Write-Host "Killing existing image manager on port ${Port}: PID $listenerPid" -ForegroundColor Yellow
    try { Stop-Process -Id $listenerPid -Force -ErrorAction Stop } catch {}
}
if ($listeners) { Start-Sleep -Seconds 1 }

& $PythonExe image_manager.py --port $Port @args
exit $LASTEXITCODE
