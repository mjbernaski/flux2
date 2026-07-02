#!/bin/bash
# Kill all running flux server processes (both the supervisor and the python server)

cd "$(dirname "$0")"

VERBOSE=0
if [ "$1" = "--verbose" ] || [ "$1" = "-v" ]; then
    VERBOSE=1
fi

log() {
    if [ $VERBOSE -eq 1 ]; then
        echo "[verbose] $*"
    fi
}

killed=0

# Kill the supervisor process via server.pid
log "Checking for server.pid file..."
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    log "Found server.pid with PID $PID"
    if kill -0 "$PID" 2>/dev/null; then
        log "PID $PID is alive — this is the supervisor shell (run_server.sh) that auto-restarts the python server"
        echo "Killing supervisor (PID $PID)..."
        kill "$PID"
        log "Sent SIGTERM to $PID — this stops the restart loop so the server won't come back"
        killed=1
    else
        log "PID $PID is no longer running — stale pid file"
    fi
    rm -f server.pid
    log "Removed server.pid"
else
    log "No server.pid file found"
fi

# Kill any run_server.sh supervisor processes
log "Searching for supervisor bash scripts (run_server.sh) via pgrep..."
SUPERVISORS=$(pgrep -f "bash.*run_server\.sh" 2>/dev/null)
if [ -n "$SUPERVISORS" ]; then
    log "Found supervisor script PIDs: $SUPERVISORS"
    log "These are the bash processes that run the restart loop — killing them prevents the python server from being respawned"
    echo "Killing supervisor scripts: $SUPERVISORS"
    kill $SUPERVISORS 2>/dev/null
    killed=1
else
    log "No supervisor bash scripts found running"
fi

# Kill any python web_server.py processes
log "Searching for python web_server.py processes via pgrep..."
SERVERS=$(pgrep -f "python.*web_server\.py" 2>/dev/null)
if [ -n "$SERVERS" ]; then
    log "Found web_server.py PIDs: $SERVERS"
    log "These are the actual Flask/web server processes serving image generation requests"
    echo "Killing web_server.py processes: $SERVERS"
    kill $SERVERS 2>/dev/null
    killed=1
else
    log "No web_server.py processes found running"
fi

if [ $killed -eq 0 ]; then
    echo "No flux server processes found."
else
    log "Waiting 1 second for processes to exit gracefully..."
    sleep 1
    # Verify everything is dead
    log "Checking if any processes survived the SIGTERM..."
    REMAINING=$(pgrep -f "(run_server\.sh|python.*web_server\.py)" 2>/dev/null)
    if [ -n "$REMAINING" ]; then
        log "Processes still alive: $REMAINING — escalating to SIGKILL (force kill, cannot be caught)"
        echo "Force-killing remaining processes: $REMAINING"
        kill -9 $REMAINING 2>/dev/null
    else
        log "All processes exited cleanly"
    fi
    echo "Done."
fi
