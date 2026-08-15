#!/usr/bin/env bash
# Serve nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4 across the two DGX Spark nodes
# (TP=2), bare-metal vLLM (no Docker) using the same .venv-vllm environment as
# the Gemma serve_*.sh scripts.
#
# Companion to DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/ (which
# is text-only). This is the vision-capable alternative: same two-node
# cluster, same RoCE fabric, different model. Only one of the two can run at
# a time -- each needs most of both nodes' unified memory.
#
# Self-detects head (spark-1, 192.168.101.10) vs worker (spark-2,
# 192.168.101.11) by local IP, so this same script works run from either
# node. Worker-first launch order avoids a race during multi-node init.
#
#   ./serve_qwen3vl_235b_dualnode.sh          # launch (worker-first, then head)
#   ./stop_qwen3vl_235b_dualnode.sh           # stop both
#
# Validated 2026-08-09: full 262144-token native context (not YaRN-scaled --
# this checkpoint's rope_scaling is "default"/untuned-extension, so unlike
# DeepSeek's YaRN ceiling this one has no extrapolation risk), text + vision
# both work, survived a stress sweep of varied image aspect ratios after the
# fixes below. ~18 tok/s single-stream decode (eager mode -- see below).
#
# Hard-won fixes, in the order they were needed:
#   1. GLOO_SOCKET_IFNAME must be set explicitly alongside NCCL_SOCKET_IFNAME.
#      Without it, Gloo (used for the out-of-band multi-node handshake,
#      separate from the NCCL data path) picks inconsistent IPv4/IPv6
#      addresses across the two nodes and fails with
#      "ss1.ss_family == ss2.ss_family" -- spark-2 has both address families
#      on the fabric interface, spark-1 apparently resolves differently.
#   2. HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1. Without it, a processor
#      auto-load (video_preprocessor_config.json, unused by this text+image
#      workload) tries to hit the network on every boot to revalidate a file
#      that's already fully cached locally, and a transient network hiccup
#      there is a full boot failure. All files needed are already local.
#   3. --mm-processor-kwargs max_pixels: this checkpoint's default image cap
#      is ~16.7M pixels (near-full-resolution). A real photo at that cap
#      generates enough vision tokens to push a request into an execution
#      path outside what startup warmup covers, which under load caused a
#      cross-node RPC timeout that killed the whole server. Capped to 262144
#      (~512x512-equivalent area, aspect ratio preserved) as a safety
#      default -- this measurably loses detail on dense/text-heavy images;
#      raise it if you need more fidelity and are willing to re-validate.
#   4. --enable-chunked-prefill: matches the working DeepSeek config; keeps a
#      large prompt (image + text) from becoming one giant synchronous step.
#   5. VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90 (default 300): even after (3),
#      a request can still land on an execution shape the CUDA-graph/kernel
#      warmup didn't cover and stall a worker mid-decode. This bounds that to
#      a fast, clear failure instead of a multi-minute hang -- it does NOT
#      prevent the stall, just fails fast when one happens.
#   6. --enforce-eager: the actual fix for the stalls in (5), not just a
#      faster-failing workaround. Disables CUDA graph capture entirely, which
#      removes the whole "request shape wasn't in the captured/warmed set"
#      failure class this checkpoint kept hitting (piecewise MoE routing
#      produces a lot of shape variety). Costs real throughput -- selective
#      re-enablement via an explicit --cudagraph-capture-sizes list tuned to
#      real traffic shapes is the natural follow-up if 18 tok/s isn't enough.
#
# Before running: this needs DeepSeek stopped first if it's running --
# ../DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/stop-deepseek-v4-flash-dspark.sh
# -- both nodes are at ~1-2GB free RAM under DeepSeek, no room for this too.
set -euo pipefail
cd "$(dirname "$0")"

HEAD_IP="192.168.101.10"
WORKER_IP="192.168.101.11"
MASTER_PORT="${MASTER_PORT:-25100}"
API_PORT="${API_PORT:-8899}"
MODEL="nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4"
SERVED_NAME="qwen3-vl-235b"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PROJECT_DIR="$(pwd)"

post_status_note() {
  local text="$1"
  curl -s --max-time 2 -X POST http://localhost:9999/note \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"text":"%s"}' "$text")" >/dev/null 2>&1 || true
  curl -s --max-time 2 -X POST http://192.168.101.11:9999/note \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"text":"%s"}' "$text")" >/dev/null 2>&1 || true
}

common_env=(
  env
  CUDA_HOME=/usr/local/cuda-13.0
  FLASHINFER_CUDA_ARCH_LIST=12.1a
  CUTE_DSL_ARCH=sm_121a
  "PATH=${PROJECT_DIR}/.venv-vllm/bin:${PATH}"
  HF_HUB_DISABLE_XET=1
  MAX_JOBS=4
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90
  NCCL_NET=IB
  NCCL_IB_DISABLE=0
  NCCL_IB_HCA=rocep1s0f1
  NCCL_SOCKET_IFNAME=enp1s0f1np1
  GLOO_SOCKET_IFNAME=enp1s0f1np1
  NCCL_IB_GID_INDEX=0
  NCCL_CROSS_NIC=1
)

vllm_flags=(
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$API_PORT"
  --trust-remote-code
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --kv-cache-dtype fp8
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --limit-mm-per-prompt "{\"image\": 4}"
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}"
  --enable-chunked-prefill
  --enforce-eager
  --distributed-executor-backend mp
  --nnodes 2 --master-addr "$HEAD_IP" --master-port "$MASTER_PORT"
)

launch_worker() {
  echo "Launching worker (this node, $WORKER_IP, node-rank 1, headless)..."
  post_status_note "qwen3-vl-235b: worker starting on spark-2, node-rank 1"
  "${common_env[@]}" VLLM_HOST_IP="$WORKER_IP" \
    nohup .venv-vllm/bin/vllm serve "$MODEL" "${vllm_flags[@]}" \
    --node-rank 1 --headless \
    > /tmp/qwen3vl235b-worker.log 2>&1 &
  disown
  echo "worker pid $!"
}

launch_head() {
  echo "Launching head (this node, $HEAD_IP, node-rank 0)..."
  post_status_note "qwen3-vl-235b: head starting on spark-1, node-rank 0, warmup ~10min"
  "${common_env[@]}" VLLM_HOST_IP="$HEAD_IP" \
    nohup .venv-vllm/bin/vllm serve "$MODEL" "${vllm_flags[@]}" \
    --node-rank 0 \
    > /tmp/qwen3vl235b-head.log 2>&1 &
  disown
  echo "head pid $!"
  echo "API will be at http://${HEAD_IP}:${API_PORT}/v1 once ready (weight load + eager-mode warmup takes ~10 min)."
}

start_remote_worker() {
  if ssh "$WORKER_IP" "pgrep -f 'vllm serve $MODEL .*--node-rank 1'" >/dev/null 2>&1; then
    echo "Worker already running on spark-2 ($WORKER_IP)."
    return
  fi
  echo "Starting worker on spark-2 ($WORKER_IP) via ssh..."
  post_status_note "qwen3-vl-235b: launching worker on spark-2 via ssh"
  ssh "$WORKER_IP" "cd '$PROJECT_DIR' && nohup ./$(basename "$0") >/tmp/qwen3vl235b-worker-launch.log 2>&1 < /dev/null &"
  sleep 5
  if ! ssh "$WORKER_IP" "pgrep -f 'vllm serve $MODEL .*--node-rank 1'" >/dev/null 2>&1; then
    echo "Worker did not start; check /tmp/qwen3vl235b-worker-launch.log and /tmp/qwen3vl235b-worker.log on spark-2." >&2
    post_status_note "qwen3-vl-235b: worker FAILED to start on spark-2"
    exit 1
  fi
  echo "Worker launched on spark-2."
}

LOCAL_IPS=" $(hostname -I) "
post_status_note "qwen3-vl-235b: launch script starting on $(hostname)"
if [[ "$LOCAL_IPS" == *" $WORKER_IP "* ]]; then
  launch_worker
elif [[ "$LOCAL_IPS" == *" $HEAD_IP "* ]]; then
  start_remote_worker
  launch_head
else
  echo "This node's IPs ($(hostname -I)) match neither head ($HEAD_IP) nor worker ($WORKER_IP). Refusing to guess." >&2
  exit 1
fi
