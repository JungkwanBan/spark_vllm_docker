# vLLM Spark — Unified Serving for DGX Spark (GB10)

Unified vLLM serving configuration for NVIDIA DGX Spark dual-node cluster (GB10 × 2).
Supports multiple Qwen3.5 models with different quantizations via `.env` presets.

## Supported Models

| Preset | Model | Quant | TP | Notes |
|--------|-------|-------|----|-------|
| `qwen3.5-397b-int4.env` | Qwen3.5-397B-A17B | INT4 AutoRound | 2 | Production config |
| `qwen3.5-122b-fp8.env` | Qwen3.5-122B-A10B | FP8 | 2 | Chunked prefill |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 | 1 | Single-node |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 | 2 | Multi-node |

## Quick Start

### TP2 Multi-Node (e.g., 397B INT4)

```bash
# On both spark01 and spark02:
cp models/qwen3.5-397b-int4.env .env
# Edit MODEL_PATH if needed

# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

### TP1 Single-Node (e.g., NVFP4 122B on spark01)

```bash
cp models/qwen3.5-122b-nvfp4.env .env
docker compose --profile head up -d
```

## Building Images

### Base image (FP8/INT4)

```bash
docker buildx build -t vllm-spark:v020-fi064 --load .
```

### NVFP4 extension

```bash
docker buildx build -f Dockerfile.nvfp4 \
  --build-arg BASE_IMAGE=vllm-spark:v020-fi064 \
  -t vllm-spark:v020-fi064-nvfp4 --load .
```

## Architecture

- **`entrypoint.sh`** — Smart entrypoint that auto-detects TP1 (standalone) vs TP2+ (Ray cluster)
- **`docker-compose.yml`** — Unified compose with `head` and `worker` profiles
- **`models/*.env`** — Validated presets for each model/quantization combo
- **`patches/`** — SM121 (Blackwell) compatibility patches for vLLM/FlashInfer
- **`scripts/run-cluster-node.sh`** — Manual Ray cluster bootstrap (for use inside containers)

## Configuration

All configuration is via `.env`. See `.env.example` for full documentation of every variable.

Key variables:
- `VLLM_IMAGE` — Pre-built Docker image to use
- `TP_SIZE` — 1 for standalone, 2+ for multi-node Ray
- `VLLM_EXTRA_ARGS` — Model-specific vllm serve flags (e.g., `--kv-cache-dtype fp8`)
- `VLLM_MARLIN_USE_ATOMIC_ADD` — Enable for INT4 AutoRound models
