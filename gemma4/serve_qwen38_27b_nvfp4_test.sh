#!/usr/bin/env bash
# TEST launch of unsloth/Qwen3.8-27B-NVFP4 on spark-1 (TP=1), using the
# .venv-vllm-0271 clone (vLLM 0.27.1) -- NOT the production .venv-vllm used
# by serve_qwen3vl_235b_dualnode.sh and serve_qwen3vl_32b_nvfp4_singlenode.sh.
#
# Why: RedHatAI/Qwen3-VL-32B-Instruct-NVFP4 (currently on 8899 via the
# production venv) predates Qwen3.8 by ~10 months. Qwen3.8-27B is a newer,
# similarly-sized (27B vs 32B) dense vision-language model with a DGX
# Spark-validated NVFP4 quant showing ~24 tok/s decode vs this hardware's
# ~3.6 tok/s on the old dense-bf16 32B fallback. No official Qwen/RedHatAI/
# nvidia NVFP4 checkpoint exists yet for Qwen3.8-27B -- using unsloth's
# (best-maintained third-party quant; had a tokenizer truncation bug at
# 2048 tokens, already patched upstream).
#
# Flags below adapt serve_qwen3vl_32b_nvfp4_singlenode.sh's vision config
# (--limit-mm-per-prompt, --mm-processor-kwargs) onto the DGX Spark forum's
# validated text-serving flags (TP1, gpu-memory-utilization 0.45 -- this
# checkpoint is ~24.6GiB vs the 32B's ~22GB but leaves far more headroom
# since it's dense-27B not 32B, --speculative-config for MTP). Vision
# correctness on Qwen3.8 is UNVALIDATED here -- the recipe this was sourced
# from only covered text serving. Test images before trusting this for VL
# work.
#
#   ./serve_qwen38_27b_nvfp4_test.sh   # launch (spark-1 only, test venv)
#   ./stop_qwen38_27b_nvfp4_test.sh    # stop
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-vllm-0271"
API_PORT="${API_PORT:-8899}"
MODEL="unsloth/Qwen3.8-27B-NVFP4"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
# vLLM auto-selects block_size=1600 when left unset on this build, which
# makes --enable-prefix-caching a no-op in practice (a request only gets
# cache credit once it shares a full 1600-token-aligned block; confirmed via
# /metrics: vllm:prefix_cache_hits_total stayed at 0 across 4.4M queried
# tokens). 16 is vLLM's normal default granularity and lets prefix caching
# actually hit on shared system prompts / repeated chat history.
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PROJECT_DIR="$(pwd)"
LOG_FILE="/tmp/qwen38-27b-nvfp4-test.log"

app_gpu_mem() {
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '/[Vv][Ll][Ll][Mm]/ { sum += $2 } END { if (sum > 0) printf "%.1fG", sum / 1024 }'
}

post_status_note() {
  local text="$1"
  local mem
  mem="$(app_gpu_mem)"
  [[ -n "$mem" ]] && text="${text} | GPU mem ${mem}"
  curl -s --max-time 2 -X POST http://localhost:9999/note \
    -H 'Content-Type: application/json' \
    -d "$(printf '{"text":"%s","channel":"app"}' "$text")" >/dev/null 2>&1 || true
}

if pgrep -f "vllm serve $MODEL" >/dev/null 2>&1; then
  echo "Already running (pid $(pgrep -f "vllm serve $MODEL" | tr '\n' ' '))."
  exit 0
fi

vllm_flags=(
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$API_PORT"
  --trust-remote-code
  --tensor-parallel-size 1
  --kv-cache-dtype fp8
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --limit-mm-per-prompt "{\"image\": 4}"
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}"
  --enable-chunked-prefill
  --enable-prefix-caching
  --async-scheduling
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  # CUDA graphs re-enabled (no --enforce-eager): dense model, not MoE, so the
  # shape-variety stalls that forced eager on the 235B MoE script shouldn't
  # apply. With spec decode, capture size must be a multiple of
  # (num_speculative_tokens + 1) or vLLM's default ladder can silently floor
  # to a size that only covers one concurrent request (see
  # DeepSeek-v4-Flash-.../CREDITS.md "CUDA-Graph Capture-Size Fix"). The 4
  # here = num_speculative_tokens(3) + 1 below -- keep them in sync.
  --max-cudagraph-capture-size $((MAX_NUM_SEQS * 4))
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

echo "Launching $MODEL (test venv $VENV, spark-1 only) on port $API_PORT..."
post_status_note "qwen3.8-27b-nvfp4 TEST: single-node launch starting on spark-1 ($VENV), warmup a few min"

# FLASHINFER_DISABLE_VERSION_CHECK: flashinfer-cubin is stuck at 0.6.13 on
# PyPI while vllm 0.27.1 pins flashinfer-python==0.6.16.post3 exactly -- no
# matching cubin exists to install anywhere. This is vLLM's own documented
# bypass for that specific skew, not a correctness workaround we invented.
env \
  CUDA_HOME=/usr/local/cuda-13.0 \
  FLASHINFER_CUDA_ARCH_LIST=12.1a \
  CUTE_DSL_ARCH=sm_121a \
  "PATH=${PROJECT_DIR}/${VENV}/bin:${PATH}" \
  HF_HUB_DISABLE_XET=1 \
  MAX_JOBS=4 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=90 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  nohup "${VENV}/bin/vllm" serve "$MODEL" "${vllm_flags[@]}" \
  > "$LOG_FILE" 2>&1 &
disown
echo "$!" > .vllm.pid
echo "pid $!"
echo "API will be at http://0.0.0.0:${API_PORT}/v1 once ready. Log: $LOG_FILE"
