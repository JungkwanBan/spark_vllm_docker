# vLLM Spark — DGX Spark (GB10) 통합 서빙

한국어 | **[English](README.md)**

NVIDIA DGX Spark 듀얼 노드 클러스터(GB10 x 2)를 위한 통합 vLLM 서빙 구성입니다.
여러 Qwen3.5 모델을 다양한 양자화 방식으로 `.env` 프리셋 하나로 전환할 수 있습니다 — 리포 하나, Dockerfile 하나, compose 파일 하나.

## 하드웨어

| 노드 | 역할 | GPU | 메모리 | 인터커넥트 |
|---|---|---|---|---|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 119 GiB 통합 메모리 | 200Gbps RoCE |

## 소프트웨어 스택

| 구성요소 | 버전 |
|---|---|
| 베이스 이미지 | NGC PyTorch 26.01 |
| vLLM | 0.17.0rc1.dev212 (nightly, cu130) |
| FlashInfer | v0.6.1 (SM121 빌드) |
| PyTorch | 2.10.0a0 |
| CUDA | 13.1 |
| NCCL | 2.29 |
| Python | 3.12 |
| Transformers | 5.2.0 |

## 지원 모델

| 프리셋 | 모델 | 양자화 | TP | 이미지 |
|---|---|---|---|---|
| `qwen3.5-397b-int4.env` | Qwen3.5-397B-A17B | INT4 AutoRound (Marlin) | 2 | base |
| `qwen3.5-122b-fp8.env` | Qwen3.5-122B-A10B | FP8 | 2 | base |
| `qwen3.5-122b-nvfp4.env` | Qwen3.5-122B-A10B | NVFP4 | 1 | nvfp4 |
| `qwen3.5-122b-nvfp4-tp2.env` | Qwen3.5-122B-A10B | NVFP4 | 2 | nvfp4 |

## 빠른 시작

### 0. Docker 이미지 준비

#### 방법 A: GHCR에서 빌드된 이미지 Pull

```bash
# 베이스 이미지 (FP8 / INT4)
docker pull ghcr.io/jungkwanban/vllm-spark:v020-fi064

# NVFP4 확장 (선택, NVFP4 모델에만 필요)
docker pull ghcr.io/jungkwanban/vllm-spark:v020-fi064-nvfp4
```

#### 방법 B: 소스에서 빌드

```bash
# 베이스 이미지 (FP8 / INT4)
docker buildx build -t vllm-spark:v020-fi064 --load .

# NVFP4 확장 (선택, NVFP4 모델에만 필요)
docker buildx build -f Dockerfile.nvfp4 \
  --build-arg BASE_IMAGE=vllm-spark:v020-fi064 \
  -t vllm-spark:v020-fi064-nvfp4 --load .
```

빌드 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `BUILD_JOBS` | 16 | 병렬 빌드 작업 수 |
| `FLASHINFER_REF` | v0.6.1 | FlashInfer git ref |
| `CUTLASS_REF` | main | CUTLASS git ref |
| `VLLM_VERSION` | 0.17.0rc1.dev212+gc88510083.cu130 | vLLM nightly wheel |
| `TRANSFORMERS_VER` | 5.2.0 | Transformers 버전 |
| `TORCH_CUDA_ARCH_LIST` | 12.1a | 타겟 CUDA 아키텍처 (Blackwell) |

### 1. 모델 프리셋 선택

```bash
cp models/qwen3.5-397b-int4.env .env
# MODEL_PATH를 로컬 모델 가중치 경로에 맞게 수정
```

### 2. 서비스 시작

#### TP2 멀티노드 (예: 397B INT4)

```bash
# spark01 (head):
docker compose --profile head up -d

# spark02 (worker):
docker compose --profile worker up -d
```

Head 노드는 Worker가 Ray 클러스터에 참여할 때까지 자동으로 대기한 후 vLLM을 시작합니다.

#### TP1 싱글노드 (예: NVFP4 122B)

```bash
cp models/qwen3.5-122b-nvfp4.env .env
docker compose --profile head up -d
```

`TP_SIZE=1`이면 Ray 없이 `vllm serve`를 직접 실행합니다.

### 3. 동작 확인

```bash
curl http://spark01:8000/health
```

## 아키텍처

```
spark01 (head)                    spark02 (worker)
┌─────────────────────┐          ┌─────────────────────┐
│  Ray Head (6379)    │          │  Ray Worker          │
│  vLLM API (:8000)   │◄────────►│                      │
│  GB10 GPU            │ 200Gbps │  GB10 GPU            │
│  TP rank 0           │  RoCE   │  TP rank 1           │
└─────────────────────┘          └─────────────────────┘
```

### 엔트리포인트 동작 방식

`entrypoint.sh`는 `ROLE`과 `TP_SIZE`에 따라 자동으로 분기합니다:

| ROLE | TP_SIZE | 동작 |
|---|---|---|
| `head` | 1 | `vllm serve` 직접 실행 (Ray 없음) |
| `head` | 2+ | Ray head 시작 → 워커 대기 → `vllm serve --distributed-executor-backend ray` |
| `worker` | any | `ray start --block` (head에 참여) |

### 리포지토리 구조

```
vllm-spark/
├── docker-compose.yml          # 통합 compose (head + worker 프로필)
├── entrypoint.sh               # 스마트 엔트리포인트 (TP1/TP2 자동 분기)
├── Dockerfile                  # 베이스 이미지 빌드 (FP8/INT4)
├── Dockerfile.nvfp4            # NVFP4 확장
├── .env.example                # 전체 설정 템플릿
├── models/                     # 검증된 모델 프리셋
│   ├── qwen3.5-397b-int4.env
│   ├── qwen3.5-122b-fp8.env
│   ├── qwen3.5-122b-nvfp4.env
│   └── qwen3.5-122b-nvfp4-tp2.env
├── patches/                    # SM121 (Blackwell) 호환성 패치
└── scripts/
    └── run-cluster-node.sh     # 수동 Ray 클러스터 부트스트랩
```

## 설정

모든 설정은 `.env`를 통해 관리합니다. 전체 문서는 [`.env.example`](.env.example)을 참고하세요.

### 주요 변수

| 변수 | 설명 | 예시 |
|---|---|---|
| `VLLM_IMAGE` | 사전 빌드된 Docker 이미지 | `vllm-spark:v020-fi064` |
| `MODEL_PATH` | 호스트의 모델 가중치 경로 | `/home/user/Models/Qwen/...` |
| `MODEL_CONTAINER_PATH` | 컨테이너 내 마운트 경로 | `/models/Qwen3.5-397B-...` |
| `SERVED_MODEL_NAME` | API 모델 이름 | `Qwen/Qwen3.5-397B-...` |
| `TP_SIZE` | 텐서 병렬 크기 (1=단독, 2+=Ray) | `2` |
| `VLLM_EXTRA_ARGS` | 모델별 vllm serve 추가 플래그 | `--kv-cache-dtype fp8 --reasoning-parser qwen3` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | INT4 AutoRound 활성화 | `1` (비활성화: 빈 값) |

## 패치

Dockerfile에서 적용하는 SM121 (Blackwell) 호환성 패치:

| 패치 | 목적 |
|---|---|
| `fastsafetensors_natural_sort` | 멀티노드 가중치 로딩 순서 수정 |
| `qwen3_5_moe_rope_fix` | transformers 5.x RoPE 검증 수정 |
| `aot_cache_fix` | AOT 캐시 torch.fx.Node pickling 수정 |
| `nogds_force` | `nogds=True` 강제 (GB10은 GDS 미지원) |
| `apply_sm121_patches` | `is_blackwell_class`, NVFP4 분리, TRITON_PTXAS |
| `moe_config_e256/e512` | GB10 튜닝 MoE 커널 설정 |

## 벤치마크 결과 (397B INT4 TP2)

[llama-benchy](https://github.com/eugr/llama-benchy) v0.3.4로 측정했습니다.

### 단일 요청 (concurrency=1)

| 테스트 | 처리량 (t/s) | TTFT (ms) |
|---|---|---|
| pp512 | 967 ± 33 | 543 ± 25 |
| pp1024 | 1,349 ± 2 | 776 ± 2 |
| pp2048 | 1,704 ± 9 | 1,224 ± 7 |
| tg128 | 27.0 ± 0.1 | — |

### 동시 요청 — 총 Decode 처리량 (t/s)

| 동시 요청 수 | tg128 총합 | tg128 피크 |
|---|---|---|
| 1 | 27.0 | 28 |
| 2 | 45.3 | 52 |
| 4 | 60~67 | 85~88 |
| 8 | 59~91 | 152~160 |

## 시스템 튜닝

DGX Spark에 권장하는 OS 수준 설정:

```bash
# Swap 부담 감소 (통합 메모리)
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## 라이선스

설정 파일은 참고용으로 제공됩니다. 모델은 해당 라이선스를 따릅니다 ([Qwen 라이선스](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)).
