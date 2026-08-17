#!/usr/bin/env bash
#
# Start vLLM server (if not already running) and launch the chat interface.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
VLLM_HOST="localhost"
VLLM_PORT="8000"
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
MODEL="Qwen/Qwen3.5-35B-A3B"
LOG="$SCRIPT_DIR/vllm_server.log"

# ─── Helpers ─────────────────────────────────────────────────────────────────

info()  { printf '\033[36m%s\033[0m\n' "$*"; }
warn()  { printf '\033[33m%s\033[0m\n' "$*"; }
err()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }

check_health() {
    curl -sf "${VLLM_URL}/models" >/dev/null 2>&1
}

wait_for_ready() {
    local timeout="${1:-600}"  # default 10 minutes — model loading is slow
    local elapsed=0
    local interval=5

    info "Waiting for vLLM to load model and become ready..."
    while ! check_health; do
        if [ "$elapsed" -ge "$timeout" ]; then
            err "Timed out after ${timeout}s waiting for vLLM to start."
            err "Check the log: $LOG"
            exit 1
        fi
        # Make sure the server process is still alive
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM process died during startup. Check the log:"
            tail -20 "$LOG"
            exit 1
        fi
        printf '.'
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo
}

# ─── Ensure vLLM is running ─────────────────────────────────────────────────

if check_health; then
    info "vLLM is already running at ${VLLM_URL}"
else
    info "Starting vLLM server with model ${MODEL}..."
    info "Logging to: ${LOG}"

    "$VENV/python" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --max-model-len 8192 \
        --enforce-eager \
        --reasoning-parser qwen3 \
        --enable-prefix-caching \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        > "$LOG" 2>&1 &

    VLLM_PID=$!
    echo "$VLLM_PID" > "$SCRIPT_DIR/.vllm.pid"
    info "vLLM server started (PID: ${VLLM_PID})"

    wait_for_ready 600
fi

# ─── Confirm model is loaded ────────────────────────────────────────────────

MODEL_INFO=$(curl -sf "${VLLM_URL}/models" | "$VENV/python" -c "
import json, sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
print(', '.join(models) if models else 'none')
" 2>/dev/null || echo "unknown")

info "Model loaded: ${MODEL_INFO}"

# ─── Launch chat ─────────────────────────────────────────────────────────────

info "Starting chat interface...\n"
exec "$VENV/python" "$SCRIPT_DIR/chat.py"
