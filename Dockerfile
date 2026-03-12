# syntax=docker/dockerfile:1.6
#
# vLLM + Qwen3.5-122B-A10B-FP8 on NVIDIA DGX Spark (GB10, SM121)
#
# Multi-stage build:
#   1. flashinfer-builder : Compile FlashInfer from upstream for SM121
#   2. runner             : Minimal runtime with vLLM nightly + Qwen3.5 patches
#
# Usage:
#   docker buildx build -t vllm-spark:qwen3.5-fp8 --load .
#
# Build args (override with --build-arg):
#   VLLM_VERSION        - vLLM nightly wheel specifier
#   FLASHINFER_REF      - FlashInfer git ref (branch/tag/SHA)
#   CUTLASS_REF         - CUTLASS git ref
#   TRANSFORMERS_VER    - transformers version for Qwen3.5 MoE
#   BUILD_JOBS          - parallel build jobs (default: 16)
# =========================================================================

ARG BUILD_JOBS=16

# =========================================================================
# STAGE 1: FlashInfer Builder
# =========================================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS flashinfer-builder

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# SM121 = NVIDIA Blackwell (GB10 / DGX Spark)
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}

# Build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl vim ninja-build git ccache && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv && pip uninstall -y flash-attn

ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# FlashInfer build dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# ---- Clone FlashInfer (upstream) ----
ARG FLASHINFER_REF=v0.6.1
ARG FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git

RUN --mount=type=cache,id=git-flashinfer,target=/git-cache/flashinfer \
    if [ -d /git-cache/flashinfer/.git ]; then \
        cp -a /git-cache/flashinfer /workspace/flashinfer && \
        cd /workspace/flashinfer && \
        git fetch origin && git fetch origin --tags; \
    else \
        git clone ${FLASHINFER_REPO} /workspace/flashinfer && \
        cp -a /workspace/flashinfer /git-cache/flashinfer; \
    fi && \
    cd /workspace/flashinfer && \
    (git checkout --detach origin/${FLASHINFER_REF} 2>/dev/null || git checkout ${FLASHINFER_REF})

RUN cd /workspace/flashinfer && \
    git submodule update --init 3rdparty/spdlog

# ---- Clone CUTLASS (upstream) ----
ARG CUTLASS_REF=main
ARG CUTLASS_REPO=https://github.com/NVIDIA/cutlass.git

RUN --mount=type=cache,id=git-cutlass,target=/git-cache/cutlass \
    cd /workspace/flashinfer && \
    rm -rf 3rdparty/cutlass && \
    if [ -d /git-cache/cutlass/.git ]; then \
        cp -a /git-cache/cutlass 3rdparty/cutlass && \
        cd 3rdparty/cutlass && \
        git fetch origin && git fetch origin --tags; \
    else \
        git clone ${CUTLASS_REPO} 3rdparty/cutlass && \
        cp -a 3rdparty/cutlass/. /git-cache/cutlass/; \
    fi && \
    cd /workspace/flashinfer/3rdparty/cutlass && \
    (git checkout --detach origin/${CUTLASS_REF} 2>/dev/null || git checkout ${CUTLASS_REF})

# ---- Optional: FlashInfer cubin cache patch ----
COPY patches/flashinfer_cache.patch /workspace/flashinfer/
RUN cd /workspace/flashinfer && \
    (patch -p1 --dry-run < flashinfer_cache.patch &>/dev/null && \
     patch -p1 < flashinfer_cache.patch || \
     echo "FlashInfer cache patch not applicable, skipping.")

# ---- Build FlashInfer wheels ----
WORKDIR /workspace/flashinfer

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' \
           -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-cubin && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v


# =========================================================================
# STAGE 2: Runner
# =========================================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV VLLM_BASE_DIR=/workspace/vllm

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Minimal runtime packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl vim git libxcb1 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv && pip uninstall -y flash-attn

WORKDIR ${VLLM_BASE_DIR}

# Tiktoken encodings (offline)
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken \
      "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken \
      "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
ENV TIKTOKEN_ENCODINGS_BASE=${VLLM_BASE_DIR}/tiktoken_encodings

# ---- Install FlashInfer wheels from builder stage ----
RUN --mount=type=bind,from=flashinfer-builder,source=/workspace/wheels,target=/mount/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install /mount/wheels/*.whl

# ---- Install vLLM nightly (cu130 / aarch64) ----
ARG VLLM_VERSION="vllm==0.17.0rc1.dev212+gc88510083.cu130"
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    pip install "${VLLM_VERSION}" \
      --index-url https://wheels.vllm.ai/nightly/cu130 \
      --no-deps --quiet || \
    pip install vllm \
      --index-url https://wheels.vllm.ai/nightly/cu130 \
      --quiet

# ---- Runtime dependencies ----
# Install vLLM deps automatically, excluding torch/torchvision/torchaudio (NGC provides these)
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    DEPS=$(pip show vllm 2>/dev/null | grep ^Requires | sed 's/Requires: //' | tr ',' '\n' | \
      sed 's/^ *//;s/ *$//' | grep -v -E '^(torch|torchvision|torchaudio|flashinfer)' | \
      tr '\n' ' ') && \
    echo "Installing vLLM deps: $DEPS" && \
    uv pip install $DEPS \
      ray[default] \
      fastsafetensors \
      nvidia-nvshmem-cu13 \
      uvloop

# ---- Qwen3.5 MoE compatibility ----
ARG TRANSFORMERS_VER=5.2.0
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install transformers==${TRANSFORMERS_VER} --upgrade --no-deps --system && \
    uv pip install huggingface-hub --upgrade --no-deps --system

# ---- Patches ----

# 1. fastsafetensors natural-sort patch for multi-node weight loading
COPY patches/fastsafetensors_natural_sort.patch /tmp/
RUN cd / && \
    (patch -p1 --dry-run < /tmp/fastsafetensors_natural_sort.patch &>/dev/null && \
     patch -p1 < /tmp/fastsafetensors_natural_sort.patch && \
     echo "Applied fastsafetensors natural-sort patch." || \
     echo "Patch not applicable (may already be fixed upstream), skipping.")

# 2. Qwen3.5 MoE rope validation fix (transformers 5.x uses | operator on sets)
COPY patches/qwen3_5_moe_rope_fix.py /tmp/
RUN python3 /tmp/qwen3_5_moe_rope_fix.py

# 3. AOT compile cache serialization fix (torch.fx.Node pickling)
COPY patches/aot_cache_fix.patch /tmp/
RUN cd / && \
    (patch -p1 --dry-run < /tmp/aot_cache_fix.patch &>/dev/null && \
     patch -p1 < /tmp/aot_cache_fix.patch && \
     echo "Applied AOT cache fix patch." || \
     echo "AOT cache fix patch not applicable, skipping.")

# 4. Force nogds=True for GB10 (no GDS support)
COPY patches/nogds_force.patch /tmp/
RUN cd / && \
    (patch -p1 --dry-run < /tmp/nogds_force.patch &>/dev/null && \
     patch -p1 < /tmp/nogds_force.patch && \
     echo "Applied nogds force patch." || \
     echo "nogds force patch not applicable, skipping.")

# 5. GB10 MoE tuning configs (E=256,N=512 and E=512,N=512)
COPY patches/moe_config_e256.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
COPY patches/moe_config_e512.json /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=512,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json

# 6. SM121 compatibility patches (is_blackwell_class, nvfp4 split, TRITON_PTXAS)
COPY patches/apply_sm121_patches.py /tmp/
RUN python3 /tmp/apply_sm121_patches.py

# Remove incompatible triton-kernels if present
RUN uv pip uninstall triton-kernels 2>/dev/null || true

# ---- Scripts ----
COPY scripts/run-cluster-node.sh ${VLLM_BASE_DIR}/
RUN chmod +x ${VLLM_BASE_DIR}/run-cluster-node.sh
ENV PATH=${VLLM_BASE_DIR}:$PATH

EXPOSE 8000
