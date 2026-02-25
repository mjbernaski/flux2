#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=5000
PIDFILE="$SCRIPT_DIR/.mg_api.pid"
LOGFILE="$SCRIPT_DIR/mg_api.log"
RESTART_DELAY=5

kill_existing() {
    # Kill any process tracked by our pidfile
    if [[ -f "$PIDFILE" ]]; then
        local old_pid
        old_pid=$(cat "$PIDFILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Killing existing instance (PID $old_pid)..."
            kill "$old_pid"
            wait "$old_pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi

    # Kill anything else listening on our port
    local pids
    pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "Killing processes on port $PORT: $pids"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 1
        # Force-kill stragglers
        pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    fi
}

cleanup() {
    echo "Shutting down..."
    rm -f "$PIDFILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

cd "$SCRIPT_DIR"

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

kill_existing

echo "Starting MedGemma API server on port $PORT..."
echo "Logs: $LOGFILE"

while true; do
    python mg_api.py >> "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PIDFILE"
    echo "[$(date)] Server started (PID $SERVER_PID)"

    # Wait for it to exit and capture the exit code
    wait "$SERVER_PID" || true
    EXIT_CODE=$?
    rm -f "$PIDFILE"

    echo "[$(date)] Server exited with code $EXIT_CODE"

    # Kill anything left on the port before restarting
    kill_existing

    echo "Restarting in ${RESTART_DELAY}s..."
    sleep "$RESTART_DELAY"
done
