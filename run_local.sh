#!/bin/bash
cd "$(dirname "$0")"

# Check for and kill any existing fl24bit.py processes
EXISTING=$(pgrep -f "python.*fl24bit.py" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "Killing existing fl24bit.py processes: $EXISTING"
    pkill -f "python.*fl24bit.py"
    sleep 1
fi

source .venv/bin/activate
python fl24bit.py --local-encoder "$@"
