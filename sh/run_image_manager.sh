#!/bin/bash
# Image Manager launcher — browse/move/delete/crop images under web-generated/
set -e
cd "$(dirname "$0")"

PORT="${IMAGE_MANAGER_PORT:-2223}"

# Kill any prior instance on this port
EXISTING=$(pgrep -f "python.*image_manager.py" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "Killing existing image_manager.py (PIDs: $EXISTING)"
  pkill -f "python.*image_manager.py" || true
  sleep 1
fi

# Activate venv if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

exec python image_manager.py --port "$PORT" "$@"
