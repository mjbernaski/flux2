#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

# Performance optimizations for Blackwell/DGX Spark
export TORCH_CUDA_ARCH_LIST="12.1"
export CUDA_LAUNCH_BLOCKING=0

python web_server.py --gguf q8 --local-encoder "$@"
