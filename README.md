# vLLM Spark — Unified Serving for DGX Spark (GB10)

**[한국어](README.ko.md)** | English

Unified vLLM serving configuration for NVIDIA DGX Spark dual-node cluster (GB10 x 2).
Supports multiple Qwen3.5 models with different quantizations via `.env` presets — one repo, one Dockerfile, one compose file.

## Hardware

| Node | Role | GPU | Memory | Interconnect |
|---|---|---|---|---|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB unified | 200Gbps RoCE |

## Software Stack

| Component | Version |
|---|---|
| Base Image | NGC PyTorch 26.01 |
| vLLM | 0.17.0rc1.dev212 (nightly, cu130) |
| FlashInfer | v0.6.1 (SM121 build) |
| PyTorch | 2.10.0a0 |
| CUDA | 13.1 |
| NCCL | 2.29 |
| Python | 3.12 |
| Transformers | 5.2.0 |

## Supported Models

| Preset | Model | Quantization | TP | Image |
|---|---|---|---|---|
| `qwen3.5-397b-int4.env` | Qwen3.5-397B-A17B | INT4 AutoRound (Marlin) | 2 | base |
| `qwen3.5-122b-fp8.env` | Qwen3.5-122B-A10B | FP8 | 2 | base |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 | 1 | nvfp4 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 | 2 | nvfp4 |

## Quick Start

### 0. Build the Docker Image

```bash
# Base image (FP8 / INT4)
docker buildx build -t vllm-spark:v020-fi064 --load .

# NVFP4 extension (optional, only for NVFP4 models)
docker buildx build -f Dockerfile.nvfp4 \
  --build-arg BASE_IMAGE=vllm-spark:v020-fi064 \
  -t vllm-spark:v020-fi064-nvfp4 --load .
```

Build arguments:

| Argument | Default | Description |
|---|---|---|
| `BUILD_JOBS` | 16 | Parallel build jobs |
| `FLASHINFER_REF` | v0.6.1 | FlashInfer git ref |
| `CUTLASS_REF` | main | CUTLASS git ref |
| `VLLM_VERSION` | 0.17.0rc1.dev212+gc88510083.cu130 | vLLM nightly wheel |
| `TRANSFORMERS_VER` | 5.2.0 | Transformers version |
| `TORCH_CUDA_ARCH_LIST` | 12.1a | Target CUDA arch (Blackwell) |

### 1. Choose a Model Preset

```bash
cp models/qwen3.5-397b-int4.env .env
# Edit MODEL_PATH to match your local model weights path
```

### 2. Start Services

#### TP2 Multi-Node (e.g., 397B INT4)

```bash
# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

The head node automatically waits for the worker to join the Ray cluster before launching vLLM.

#### TP1 Single-Node (e.g., NVFP4 122B)

```bash
cp models/qwen3.5-122b-nvfp4.env .env
docker compose --profile head up -d
```

When `TP_SIZE=1`, the entrypoint skips Ray entirely and runs `vllm serve` directly.

### 3. Verify

```bash
curl http://spark01:8000/health
```

## Architecture

```
spark01 (head)                    spark02 (worker)
┌─────────────────────┐          ┌─────────────────────┐
│  Ray Head (6379)    │          │  Ray Worker          │
│  vLLM API (:8000)   │◄────────►│                      │
│  GB10 GPU            │ 200Gbps │  GB10 GPU            │
│  TP rank 0           │  RoCE   │  TP rank 1           │
└─────────────────────┘          └─────────────────────┘
```

### How the Entrypoint Works

`entrypoint.sh` routes automatically based on `ROLE` and `TP_SIZE`:

| ROLE | TP_SIZE | Behavior |
|---|---|---|
| `head` | 1 | Direct `vllm serve` (no Ray) |
| `head` | 2+ | Ray head → wait for workers → `vllm serve --distributed-executor-backend ray` |
| `worker` | any | `ray start --block` (joins head) |

### Repository Structure

```
vllm-spark/
├── docker-compose.yml          # Unified compose (head + worker profiles)
├── entrypoint.sh               # Smart entrypoint (TP1/TP2 auto-routing)
├── Dockerfile                  # Base image build (FP8/INT4)
├── Dockerfile.nvfp4            # NVFP4 extension
├── .env.example                # Full configuration template
├── models/                     # Validated model presets
│   ├── qwen3.5-397b-int4.env
│   ├── qwen3.5-122b-fp8.env
│   ├── qwen3.5-122b-nvfp4.env
│   └── qwen3.5-122b-nvfp4-tp2.env
├── patches/                    # SM121 (Blackwell) compatibility patches
└── scripts/
    └── run-cluster-node.sh     # Manual Ray cluster bootstrap
```

## Configuration

All configuration is via `.env`. See [`.env.example`](.env.example) for full documentation.

### Key Variables

| Variable | Description | Example |
|---|---|---|
| `VLLM_IMAGE` | Pre-built Docker image | `vllm-spark:v020-fi064` |
| `MODEL_PATH` | Host path to model weights | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | Container mount point | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API model name | `Qwen/Qwen3.5-397B-...` |
| `TP_SIZE` | Tensor parallel size (1=standalone, 2+=Ray) | `2` |
| `VLLM_EXTRA_ARGS` | Model-specific vllm serve flags | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | Enable for INT4 AutoRound | `1` (or empty to disable) |

## Patches

The Dockerfile applies SM121 (Blackwell) compatibility patches:

| Patch | Purpose |
|---|---|
| `fastsafetensors_natural_sort` | Multi-node weight loading order fix |
| `qwen3_5_moe_rope_fix` | RoPE validation fix for transformers 5.x |
| `aot_cache_fix` | torch.fx.Node pickling fix for AOT cache |
| `nogds_force` | Force `nogds=True` (GB10 has no GDS support) |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 split, TRITON_PTXAS |
| `moe_config_e256/e512` | GB10-tuned MoE kernel configs |

## Benchmark Results (397B INT4 TP2)

Measured with [llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4.

### Single Request (concurrency=1)

| Test | Throughput (t/s) | TTFT (ms) |
|---|---|---|
| pp512 | 967 ± 33 | 543 ± 25 |
| pp1024 | 1,349 ± 2 | 776 ± 2 |
| pp2048 | 1,704 ± 9 | 1,224 ± 7 |
| tg128 | 27.0 ± 0.1 | — |

### Concurrent Requests — Total Decode Throughput (t/s)

| Concurrency | tg128 total | tg128 peak |
|---|---|---|
| 1 | 27.0 | 28 |
| 2 | 45.3 | 52 |
| 4 | 60~67 | 85~88 |
| 8 | 59~91 | 152~160 |

## System Tuning

Recommended OS-level settings for DGX Spark:

```bash
# Reduce swap pressure (unified memory)
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## License

Configuration files are provided as-is for reference. Models are subject to their respective licenses ([Qwen License](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)).
