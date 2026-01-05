#!/bin/bash
cd "$(dirname "$0")"

# Check for and kill any existing server process
if [ -f server.pid ]; then
    OLD_PID=$(cat server.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing existing server (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 1
    fi
    rm -f server.pid
fi

# Also check for any web_server.py processes
EXISTING=$(pgrep -f "python.*web_server.py" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "Killing existing web_server.py processes: $EXISTING"
    pkill -f "python.*web_server.py"
    sleep 1
fi

source .venv/bin/activate

# Save PID and start server with FLUX.2 4-bit quantized model
echo $$ > server.pid
python web_server.py --flux2 "$@"
