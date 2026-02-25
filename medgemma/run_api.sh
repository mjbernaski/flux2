#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_PORT=5000
WEB_PORT=3000
API_PIDFILE="$SCRIPT_DIR/.mg_api.pid"
WEB_PIDFILE="$SCRIPT_DIR/.mg_web.pid"
API_LOGFILE="$SCRIPT_DIR/mg_api.log"
WEB_LOGFILE="$SCRIPT_DIR/mg_web.log"
RESTART_DELAY=5

kill_pid_file() {
    local pidfile="$1"
    if [[ -f "$pidfile" ]]; then
        local old_pid
        old_pid=$(cat "$pidfile")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Killing existing instance (PID $old_pid)..."
            kill "$old_pid"
            wait "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

kill_port() {
    local port="$1"
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 1
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    fi
}

kill_existing() {
    kill_pid_file "$API_PIDFILE"
    kill_pid_file "$WEB_PIDFILE"
    kill_port "$API_PORT"
    kill_port "$WEB_PORT"
}

start_web() {
    # Kill old web process if running
    kill_pid_file "$WEB_PIDFILE"
    kill_port "$WEB_PORT"

    node "$SCRIPT_DIR/web/server.js" >> "$WEB_LOGFILE" 2>&1 &
    WEB_PID=$!
    echo "$WEB_PID" > "$WEB_PIDFILE"
    echo "[$(date)] Web frontend started (PID $WEB_PID) on port $WEB_PORT"
}

cleanup() {
    echo "Shutting down..."
    # Kill web server
    if [[ -f "$WEB_PIDFILE" ]]; then
        kill "$(cat "$WEB_PIDFILE")" 2>/dev/null || true
        rm -f "$WEB_PIDFILE"
    fi
    rm -f "$API_PIDFILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

cd "$SCRIPT_DIR"

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

kill_existing

echo "Starting MedGemma API server on port $API_PORT..."
echo "Starting web frontend on port $WEB_PORT..."
echo "API logs:  $API_LOGFILE"
echo "Web logs:  $WEB_LOGFILE"

start_web

while true; do
    python mg_api.py >> "$API_LOGFILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$API_PIDFILE"
    echo "[$(date)] API server started (PID $SERVER_PID)"

    # Restart web frontend if it died
    if [[ -f "$WEB_PIDFILE" ]]; then
        local_web_pid=$(cat "$WEB_PIDFILE")
        if ! kill -0 "$local_web_pid" 2>/dev/null; then
            echo "[$(date)] Web frontend died, restarting..."
            start_web
        fi
    else
        start_web
    fi

    wait "$SERVER_PID" || true
    EXIT_CODE=$?
    rm -f "$API_PIDFILE"

    echo "[$(date)] API server exited with code $EXIT_CODE"

    kill_port "$API_PORT"

    echo "Restarting API in ${RESTART_DELAY}s..."
    sleep "$RESTART_DELAY"
done
