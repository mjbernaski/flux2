#!/bin/bash
cd "$(dirname "$0")"

# Check for and kill any existing flux_cli.py processes
EXISTING=$(pgrep -f "python.*flux_cli.py" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "Killing existing flux_cli.py processes: $EXISTING"
    pkill -f "python.*flux_cli.py"
    sleep 1
fi

source .venv/bin/activate
python flux_cli.py --full-model "$@"
