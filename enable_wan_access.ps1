# enable_wan_access.ps1 — one-time setup so the eero's port-forward reaches
# this machine (see 2026-08-08: forward targets the eero DHCP reservation
# 192.168.4.92, but the wired adapter had a hand-set static 192.168.6.40).
#
# 1. Puts the Ethernet adapter back on DHCP so eero assigns the reserved
#    192.168.4.92 again (LAN URL becomes 192.168.4.92:2222; the servers bind
#    0.0.0.0, so they need no change).
# 2. Opens Windows Firewall for the flux web server (2222), image
#    manager (2223) and the compact UI (3333) on the private/domain profiles.
#
# Needs admin; relaunches itself elevated if it isn't.

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Not elevated - relaunching as administrator..." -ForegroundColor Yellow
    Start-Process -FilePath "pwsh.exe" -ArgumentList "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"" -Verb RunAs
    exit
}

Write-Host "== Switching Ethernet to DHCP (eero will assign the reserved 192.168.4.92) ==" -ForegroundColor Cyan
netsh interface ip set address name="Ethernet" source=dhcp
netsh interface ip set dnsservers name="Ethernet" source=dhcp

Write-Host "== Adding firewall rules for ports 2222/2223/3333 ==" -ForegroundColor Cyan
netsh advfirewall firewall add rule name="FLUX web server (TCP 2222)" dir=in action=allow protocol=TCP localport=2222 profile=private,domain
netsh advfirewall firewall add rule name="FLUX image manager (TCP 2223)" dir=in action=allow protocol=TCP localport=2223 profile=private,domain
netsh advfirewall firewall add rule name="FLUX compact UI (TCP 3333)" dir=in action=allow protocol=TCP localport=3333 profile=private,domain

Write-Host "== Result ==" -ForegroundColor Cyan
Start-Sleep -Seconds 5   # give DHCP a moment before showing the new address
ipconfig | Select-String "IPv4"
netsh advfirewall firewall show rule name="FLUX web server (TCP 2222)" | Select-String "Rule Name|Enabled|LocalPort|Action"

Write-Host ""
Write-Host "Done. Test from off-LAN: http://66.33.11.152:2222 (LAN is now http://192.168.4.92:2222)" -ForegroundColor Green
Read-Host "Press Enter to close"
