# free_gpu.ps1 — unload Ollama and LM Studio models to free the 5090's VRAM.
# The apps themselves keep running; they reload models on next use.

Write-Host "== Unloading Ollama models =="
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $loaded = ollama ps 2>$null | Select-Object -Skip 1 |
        ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ }
    if ($loaded) {
        foreach ($m in $loaded) {
            Write-Host "  ollama stop $m"
            ollama stop $m
        }
    } else {
        Write-Host "  no models loaded"
    }
} else {
    Write-Host "  ollama CLI not found, skipping"
}

Write-Host "== Unloading LM Studio models =="
if (Get-Command lms -ErrorAction SilentlyContinue) {
    lms unload --all
} else {
    Write-Host "  lms CLI not found, skipping"
}

# Fallback: kill any llama-server engine still holding VRAM (parent apps survive)
Start-Sleep -Seconds 2
$stragglers = Get-Process llama-server -ErrorAction SilentlyContinue
if ($stragglers) {
    Write-Host "== Killing leftover llama-server processes =="
    $stragglers | ForEach-Object { Write-Host "  PID $($_.Id)" }
    $stragglers | Stop-Process -Force
}

Write-Host ""
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
