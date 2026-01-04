#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python web_server.py --gguf q8 --local-encoder "$@"
