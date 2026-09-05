# FLUX Server Launcher (Windows) - PowerShell port of run_server.sh
# Usage:  .\run_server.ps1          interactive menu
#         .\run_server.ps1 9        launch config 9 directly
#         .\run_server.ps1 last     resume the most recently run config
# Keep the config table in sync with run_server.sh, SERVER_OPTIONS.md and
# the SERVER_CONFIGS table in web_server.py.

Set-Location $PSScriptRoot

# Windows RedirectionGuard can refuse to traverse the symlinks the HF cache
# uses ("untrusted mount point"), so store real files instead of symlinks.
$env:HF_HUB_DISABLE_SYMLINKS = "1"

# klein-9B leaves only ~7GB VRAM headroom; generating at varying resolutions
# fragments the caching allocator until it creeps past 32GB and the Windows
# driver spills to system RAM (~7x slower steps, no error). Expandable
# segments lets the allocator reuse blocks across differing shapes.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# torch ships Intel's OpenMP runtime (torch\lib\libiomp5md.dll), which installs
# a Windows console control handler. When the console this server is attached to
# gets CTRL_CLOSE_EVENT (terminal window/tab closed) or CTRL_LOGOFF_EVENT (user
# logs off), that handler kills the process outright:
#   forrtl: error (200): program aborting due to window-CLOSE event
# It is not a crash - the server was healthy and serving requests the second
# before - and it is unrecoverable, because the same console event also takes
# down this supervisor, so the auto-restart loop below never gets to run. These
# two vars stop the Intel runtimes from installing their handlers at all, which
# leaves Python's own (SIGINT -> KeyboardInterrupt) as the only one.
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
$env:KMP_HANDLE_SIGNALS = "0"

$SwitchExitCode = 86
$SwitchConfigFile = ".next_config"
$LastConfigFile = ".last_config"
$LogFile = "server.log"
$MaxRetries = 5
$RetryDelay = 3
$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Get-ConfigArgs([int]$Config) {
    switch ($Config) {
        1  { @{ Args = @();                                        Desc = "FLUX.1 4-bit BNB" } }
        2  { @{ Args = @("--full-model");                          Desc = "FLUX.1 Full" } }
        3  { @{ Args = @("--gguf", "q8", "--local-encoder");       Desc = "FLUX.1 GGUF Q8" } }
        4  { @{ Args = @("--schnell", "--local-encoder");          Desc = "FLUX.1-schnell" } }
        5  { @{ Args = @("--uncensored");                          Desc = "FLUX.1 + U-LoRA" } }
        6  { @{ Args = @("--flux2");                               Desc = "FLUX.2 4-bit BNB" } }
        7  { @{ Args = @("--flux2", "--full-model", "--turbo");    Desc = "FLUX.2 Full + Turbo" } }
        8  { @{ Args = @("--flux2", "--full-model", "--no-turbo"); Desc = "FLUX.2 Full (no Turbo)" } }
        # --quantize-encoder: 17GB transformer + 16GB Qwen3 encoder don't both fit
        # a 32GB card in bf16; Windows sysmem fallback then makes steps ~10x slower.
        # NF4 encoder (~5GB) keeps the transformer bf16 and everything in VRAM.
        9  { @{ Args = @("--klein", "--quantize-encoder");         Desc = "FLUX.2-klein-9B (NF4 encoder)" } }
        10 { @{ Args = @("--kontext");                             Desc = "FLUX.1 Kontext (editor)" } }
        11 { @{ Args = @("--kontext", "--full-model");             Desc = "FLUX.1 Kontext Full (editor, bf16)" } }
        12 { @{ Args = @("--kontext", "--full-model", "--uncensored"); Desc = "FLUX.1 Kontext Full + U-LoRA" } }
        13 { @{ Args = @("--sdxl");                                Desc = "SDXL (photoreal)" } }
        # Same reasoning as config 9, and it bites harder here than the small
        # transformer suggests: klein-4B's Qwen3 encoder is the same ~16GB in
        # bf16 as the 9B's, so a 4B config was still filling a 32GB card and
        # leaving nothing for the full-resolution VAE decode. Measured with the
        # bf16 encoder: 31.8/32.6GB resident at 1.25MP, and generations at
        # 1.75MP+ stalled on the last step paging to system RAM.
        14 { @{ Args = @("--klein-4b", "--quantize-encoder");      Desc = "FLUX.2-klein-4B (NF4 encoder)" } }
        default { $null }
    }
}

function Show-Menu {
    Clear-Host
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "       FLUX Image Generator - Server Launcher" -ForegroundColor Green
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  FLUX.1 (12B)" -ForegroundColor Yellow
    Write-Host "    1) 4-bit            Low VRAM"
    Write-Host "    2) Full             Best quality"
    Write-Host "    3) GGUF Q8          DGX Spark optimized"
    Write-Host "    4) schnell          4-step fast (Apache 2.0)"
    Write-Host "    5) U-LoRA           Full model + LoRA"
    Write-Host ""
    Write-Host "  FLUX.2 (32B / klein 9B/4B)" -ForegroundColor Yellow
    Write-Host "    6) 4-bit            Low VRAM"
    Write-Host "    7) Full (Turbo)     8-step fast inference"
    Write-Host "    8) Full (no Turbo)  Max quality, slower"
    Write-Host "    9) klein-9B         [default] faster 9B"
    Write-Host "   14) klein-4B         Lowest-VRAM FLUX.2, ~13GB bf16"
    Write-Host ""
    Write-Host "  Editing" -ForegroundColor Yellow
    Write-Host "   10) Kontext          Instruction editing (FLUX.1, 4-bit)"
    Write-Host "   11) Kontext Full     Instruction editing (FLUX.1, full bf16)"
    Write-Host "   12) Kontext U-LoRA   Kontext Full + U-LoRA (edit refs)"
    Write-Host ""
    Write-Host "  Stable Diffusion (SDXL)" -ForegroundColor Yellow
    Write-Host "   13) SDXL photoreal   Photoreal checkpoint, negative prompts"
    Write-Host ""
    Write-Host "    q) Quit" -ForegroundColor Red
    Write-Host ""
}

function Stop-ExistingServer {
    if (Test-Path "server.pid") {
        $oldPid = Get-Content "server.pid" -ErrorAction SilentlyContinue
        if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            Write-Host "Killing existing server (PID $oldPid)..." -ForegroundColor Yellow
            try { Stop-Process -Id $oldPid -Force -ErrorAction Stop } catch {}
            Start-Sleep -Seconds 1
        }
        Remove-Item "server.pid" -Force -ErrorAction SilentlyContinue
    }
    # CIM/WMI queries can be access-denied, so find the old server without
    # them: whatever is listening on the port, plus any python from our venv.
    $port = 2222
    if (Test-Path ".env") {
        $m = Select-String -Path ".env" -Pattern '^PORT=(\d+)' -ErrorAction SilentlyContinue
        if ($m) { $port = [int]$m.Matches[0].Groups[1].Value }
    }
    $killed = $false
    $listeners = netstat -ano | Select-String ":$port\s.*LISTENING\s+(\d+)" |
        ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Unique
    foreach ($listenerPid in $listeners) {
        Write-Host "Killing process listening on port ${port}: $listenerPid" -ForegroundColor Yellow
        try { Stop-Process -Id $listenerPid -Force -ErrorAction Stop; $killed = $true } catch {}
    }
    foreach ($p in (Get-Process python* -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $PythonExe })) {
        Write-Host "Killing leftover venv python: $($p.Id)" -ForegroundColor Yellow
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop; $killed = $true } catch {}
    }
    if ($killed) { Start-Sleep -Seconds 1 }
}

# The critique/describe VLM lives at OLLAMA_URL, which despite the name is not
# necessarily ollama and not necessarily local: edit_loop also speaks the
# OpenAI dialect, so the endpoint may be vLLM or llama.cpp on another box. Only
# a local ollama is ours to start — anything else is another host's process, so
# probe it and report. Without this distinction the launcher claimed
# "ollama not installed" on a box correctly pointed at a remote vLLM.
function Confirm-Ollama {
    $url = $env:OLLAMA_URL
    if (-not $url -and (Test-Path ".env")) {
        $m = Select-String -Path ".env" -Pattern '^OLLAMA_URL=(.+)$' -ErrorAction SilentlyContinue
        if ($m) { $url = $m.Matches[0].Groups[1].Value.Trim() }
    }
    if (-not $url) { $url = "http://127.0.0.1:11434" }
    $url = $url.TrimEnd('/')

    if ($url -notmatch '^https?://(127\.0\.0\.1|localhost)(:11434)?$') {
        # Remote and/or OpenAI-compatible. Probe both dialects' cheap endpoints
        # — whichever answers, the VLM features have a backend. "$url/models"
        # covers a URL that already ends in /v1, which edit_loop accepts (and
        # treats as OpenAI without probing).
        foreach ($probe in @("$url/api/version", "$url/v1/models", "$url/models")) {
            try {
                Invoke-RestMethod -Uri $probe -TimeoutSec 3 | Out-Null
                Write-Host "VLM endpoint $url is up (critique/describe available)." -ForegroundColor Green
                return
            } catch {}
        }
        Write-Host "VLM endpoint $url is not responding - critique/describe may be unavailable" -ForegroundColor Yellow
        return
    }

    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Host "ollama not installed - VLM critique/describe will be unavailable" -ForegroundColor Yellow
        return
    }
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null
        return
    } catch {}
    Write-Host "Starting ollama (VLM backend, logging to ollama.log)..." -ForegroundColor Cyan
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput "ollama.log" -RedirectStandardError "ollama.err.log"
    foreach ($i in 1..20) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 1 | Out-Null
            Write-Host "ollama is up." -ForegroundColor Green
            return
        } catch { Start-Sleep -Milliseconds 500 }
    }
    Write-Host "ollama did not come up within 10s - VLM features may be unavailable" -ForegroundColor Yellow
}

# The gallery/crop tool (image_manager.py, port 2223) should be up whenever
# the generation server is. Skip if something is already listening there.
# Must run after Stop-ExistingServer: its venv-python sweep would kill a
# manager started earlier.
function Confirm-ImageManager {
    $imPort = if ($env:IMAGE_MANAGER_PORT) { [int]$env:IMAGE_MANAGER_PORT } else { 2223 }
    $listening = netstat -ano | Select-String ":$imPort\s.*LISTENING"
    if ($listening) { return }
    Write-Host "Starting image manager on port $imPort (logging to image_manager.log)..." -ForegroundColor Cyan
    Start-Process -FilePath $PythonExe -ArgumentList "-u", "image_manager.py", "--port", "$imPort" `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput "image_manager.log" -RedirectStandardError "image_manager.err.log"
}

function Write-Log([string]$Message, [string]$Color = "White") {
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $Message
}

function Start-FluxServer([int]$Config, [string[]]$ExtraArgs = @()) {
    $cfg = Get-ConfigArgs $Config
    if (-not $cfg) {
        Write-Host "Invalid selection" -ForegroundColor Red
        return 1
    }

    if (-not (Test-Path $PythonExe)) {
        Write-Host "No .venv found - create it first: uv venv .venv --python 3.11; uv pip install --python .venv\Scripts\python.exe -r requirements.txt" -ForegroundColor Red
        return 1
    }

    Write-Host ""
    Write-Host "Starting $($cfg.Desc)..." -ForegroundColor Green
    Write-Host "Command: python web_server.py $($cfg.Args -join ' ')" -ForegroundColor Blue
    Write-Host "Auto-restart enabled (max $MaxRetries retries on failure)" -ForegroundColor Yellow
    Write-Host "Logging to: $LogFile" -ForegroundColor Cyan
    Write-Host ""

    Stop-ExistingServer
    Confirm-Ollama
    Confirm-ImageManager

    # kill_flux.ps1 leaves .flux_stopped behind so the FluxServerWatchdog task
    # does not resurrect a deliberate stop. Starting again is the intent that
    # cancels it - clear it here so the watchdog guards this run too.
    Remove-Item ".flux_stopped" -Force -ErrorAction SilentlyContinue

    $PID | Out-File "server.pid" -Encoding ascii
    try {
        $attempt = 0
        $firstStart = $true
        while ($true) {
            $attempt++
            $startTime = Get-Date

            if (-not $firstStart) {
                Write-Log ""
                Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Restart attempt $attempt of $MaxRetries" "Yellow"
            } else {
                $firstStart = $false
                Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Initial server start" "Green"
            }

            $Config | Out-File $LastConfigFile -Encoding ascii
            $env:FLUX_CONFIG = "$Config"
            # -u: unbuffered stdout so diagnostics land in the log immediately.
            & $PythonExe -u web_server.py @($cfg.Args) @ExtraArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
            $exitCode = $LASTEXITCODE

            $duration = [int]((Get-Date) - $startTime).TotalSeconds

            # Model switch requested from the web UI: relaunch with the new
            # config's flags. Not a crash - reset the retry counter.
            if ($exitCode -eq $SwitchExitCode -and (Test-Path $SwitchConfigFile)) {
                $newConfig = (Get-Content $SwitchConfigFile -Raw).Trim()
                Remove-Item $SwitchConfigFile -Force
                $newCfg = $null
                if ($newConfig -match '^([1-9]|1[0-4])$') { $newCfg = Get-ConfigArgs ([int]$newConfig) }
                if ($newCfg) {
                    $Config = [int]$newConfig
                    $cfg = $newCfg
                } else {
                    Write-Log "Invalid switch request '$newConfig' - restarting current model" "Red"
                }
                Write-Log ""
                Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Model switch - relaunching as: $($cfg.Desc)" "Cyan"
                $attempt = 0
                $firstStart = $true
                continue
            }

            if ($exitCode -eq 0) {
                Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Server exited cleanly." "Green"
                break
            }

            Write-Log ""
            Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Server crashed with exit code $exitCode" "Red"

            if ($duration -gt 60) {
                Write-Log "Server ran for $duration seconds. Resetting retry counter." "Green"
                $attempt = 0
            }

            if ($attempt -ge $MaxRetries -and $attempt -ne 0) {
                Write-Log "Maximum retries ($MaxRetries) reached. Giving up." "Red"
                return 1
            }

            Write-Log "Restarting in $RetryDelay seconds... (attempt $($attempt + 1)/$MaxRetries). Ctrl+C to abort." "Red"
            Start-Sleep -Seconds $RetryDelay
        }
    } finally {
        Remove-Item "server.pid" -Force -ErrorAction SilentlyContinue
    }
    return 0
}

# --- Main ---
$first = $args | Select-Object -First 1
$rest = @($args | Select-Object -Skip 1)

if ($first -eq "last") {
    $lastConfig = Get-Content $LastConfigFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($lastConfig -notmatch '^([1-9]|1[0-4])$') { $lastConfig = 9 }
    exit (Start-FluxServer ([int]$lastConfig) $rest)
}

if ($first -match '^([1-9]|1[0-4])$') {
    exit (Start-FluxServer ([int]$first) $rest)
}

# No arguments and no interactive console -> launch the default (klein-9B)
# directly, mirroring run_server.sh.
if (-not $first -and [Console]::IsInputRedirected) {
    exit (Start-FluxServer 9)
}

while ($true) {
    Show-Menu
    $choice = Read-Host "Select configuration [1-14, default=9, q to quit]"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "9" }
    if ($choice -match '^[qQ]$') { Write-Host "Goodbye!" -ForegroundColor Green; exit 0 }
    if ($choice -match '^([1-9]|1[0-4])$') {
        exit (Start-FluxServer ([int]$choice))
    }
    Write-Host "Invalid selection. Press Enter to continue..." -ForegroundColor Red
    Read-Host | Out-Null
}
