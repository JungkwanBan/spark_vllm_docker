#!/bin/bash
set -euo pipefail

# =============================================================================
# vLLM Spark Unified Entrypoint
#
# Automatically handles:
#   - TP1 standalone (no Ray, direct vllm serve)
#   - TP2+ head (Ray head → wait for workers → vllm serve)
#   - TP2+ worker (Ray worker --block)
#
# Required environment variables:
#   ROLE              - "head" or "worker"
#   TP_SIZE           - tensor parallel size (1 = standalone, 2+ = multi-node)
#   MODEL_CONTAINER_PATH  - model path inside container
#   SERVED_MODEL_NAME     - model name for API
#
# Optional (with defaults):
#   HOST_PORT, MAX_MODEL_LEN, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION,
#   MAX_NUM_BATCHED_TOKENS, VLLM_EXTRA_ARGS
# =============================================================================

: "${ROLE:?ROLE must be set to 'head' or 'worker'}"
: "${TP_SIZE:=1}"

# ---- Worker: just join Ray and block ----
if [ "${ROLE}" = "worker" ]; then
    echo "[entrypoint] Starting Ray WORKER → ${HEAD_ROCE_IP}:${RAY_PORT}"
    exec ray start \
        --address="${HEAD_ROCE_IP}:${RAY_PORT}" \
        --node-ip-address="${WORKER_ROCE_IP}" \
        --block
fi

# ---- Head: standalone or multi-node ----
if [ "${ROLE}" != "head" ]; then
    echo "[entrypoint] ERROR: ROLE must be 'head' or 'worker', got '${ROLE}'"
    exit 1
fi

# Build vllm serve command
VLLM_CMD=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN:-32768}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --trust-remote-code
    --host 0.0.0.0
    --port "${HOST_PORT:-8000}"
    --dtype auto
    --enable-prefix-caching
)

if [ "${TP_SIZE}" -ge 2 ]; then
    # ---- Multi-node: start Ray head, wait for workers, then serve ----
    echo "[entrypoint] Starting Ray HEAD (TP_SIZE=${TP_SIZE})..."
    ray start --head --port="${RAY_PORT}" \
        --node-ip-address="${HEAD_ROCE_IP}" \
        --dashboard-host=0.0.0.0 \
        --disable-usage-stats

    echo "[entrypoint] Waiting for ${TP_SIZE} node(s) to join Ray cluster..."
    while true; do
        NODE_COUNT=$(ray status 2>/dev/null | grep -c 'node_' || echo 0)
        if [ "${NODE_COUNT}" -ge "${TP_SIZE}" ]; then
            echo "[entrypoint] All ${TP_SIZE} nodes joined! Starting vLLM..."
            break
        fi
        sleep 5
    done

    VLLM_CMD+=(
        --tensor-parallel-size "${TP_SIZE}"
        --distributed-executor-backend ray
    )
else
    # ---- Standalone: direct serve, no Ray ----
    echo "[entrypoint] Starting vLLM standalone (TP_SIZE=1)..."
fi

# Append model-specific extra args (split on whitespace)
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    VLLM_CMD+=(${VLLM_EXTRA_ARGS})
fi

echo "[entrypoint] Running: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
