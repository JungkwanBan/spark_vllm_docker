#!/bin/bash
# Hot-patch script: Apply patches #1 (AOT cache fix), #4 (nogds), #5 (MoE tuning)
# to running vLLM containers on spark01 and spark02
# These patches modify Python files in-place - container restart required after.

set -e

VLLM_PKG="/usr/local/lib/python3.12/dist-packages/vllm"
MOE_CONFIGS_DIR="${VLLM_PKG}/model_executor/layers/fused_moe/configs"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

CONTAINERS=(
  "spark01:vllm-qwen3.5-122b-fp8-head"
  "spark02:vllm-qwen3.5-122b-fp8-worker"
)

for entry in "${CONTAINERS[@]}"; do
  HOST="${entry%%:*}"
  CONTAINER="${entry##*:}"
  echo "========================================="
  echo "Patching ${CONTAINER} on ${HOST}"
  echo "========================================="

  # Patch #1: AOT compile cache fix
  echo "[1/3] Applying AOT cache fix (caching.py)..."
  ssh "$HOST" "docker cp - ${CONTAINER}:/tmp/" < "${PATCH_DIR}/aot_cache_fix.patch"
  ssh "$HOST" "docker exec ${CONTAINER} bash -c 'cd / && patch -p1 --dry-run < /tmp/aot_cache_fix.patch && patch -p1 < /tmp/aot_cache_fix.patch'" 2>&1 || \
    echo "  WARNING: AOT cache patch may already be applied or not applicable"

  # Patch #4: Force nogds=True
  echo "[2/3] Applying nogds force patch (weight_utils.py)..."
  ssh "$HOST" "docker cp - ${CONTAINER}:/tmp/" < "${PATCH_DIR}/nogds_force.patch"
  ssh "$HOST" "docker exec ${CONTAINER} bash -c 'cd / && patch -p1 --dry-run < /tmp/nogds_force.patch && patch -p1 < /tmp/nogds_force.patch'" 2>&1 || \
    echo "  WARNING: nogds patch may already be applied or not applicable"

  # Patch #5: GB10 MoE tuning configs
  echo "[3/3] Installing GB10 MoE tuning configs..."
  ssh "$HOST" "docker cp - ${CONTAINER}:${MOE_CONFIGS_DIR}/" < <(
    cd "${PATCH_DIR}" && tar cf - \
      --transform='s/moe_config_e256.json/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json/' \
      --transform='s/moe_config_e512.json/E=512,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json/' \
      moe_config_e256.json moe_config_e512.json
  )
  ssh "$HOST" "docker exec ${CONTAINER} ls ${MOE_CONFIGS_DIR}/*GB10* 2>/dev/null" && echo "  MoE configs installed." || echo "  WARNING: MoE config install failed"

  echo ""
done

echo "All patches applied. Containers need restart to take effect."
echo ""
echo "Restart commands:"
echo "  cd /home/bjk110/docker/vllm-qwen3.5-dgx-spark"
echo "  ssh spark02 'cd /home/bjk110/docker/vllm-qwen3.5-dgx-spark && docker compose --profile worker down && docker compose --profile worker up -d'"
echo "  ssh spark01 'cd /home/bjk110/docker/vllm-qwen3.5-dgx-spark && docker compose --profile head down && docker compose --profile head up -d'"
