#!/usr/bin/env python3
"""
Apply SM121 (GB10 / DGX Spark) compatibility patches to vLLM.
Based on https://github.com/seli-equinix/vllm/tree/feature/sm121-gb10-support

Patches applied:
  #3  Split has_flashinfer_nvfp4 from has_flashinfer_cutlass_fused_moe
  #6  Auto-configure TRITON_PTXAS_PATH for SM121
  #9  Add is_blackwell_class() helper (SM10x/SM11x/SM12x)

Run during Docker build:
  RUN python3 /tmp/apply_sm121_patches.py
"""
import os
import sys
import glob
import functools

VLLM_PKG = None
for p in glob.glob("/usr/local/lib/python3.*/dist-packages/vllm"):
    VLLM_PKG = p
    break

if not VLLM_PKG:
    print("ERROR: vllm package not found")
    sys.exit(1)

print(f"vLLM package: {VLLM_PKG}")
applied = []
skipped = []


def patch_file(path, old, new, label):
    with open(path) as f:
        src = f.read()
    if old not in src:
        if new in src:
            skipped.append(f"{label} (already applied)")
        else:
            skipped.append(f"{label} (anchor not found)")
        return False
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    applied.append(label)
    return True


# ============================================================
# Patch #9: Add is_blackwell_class() to cuda.py and interface.py
# ============================================================
print("\n[1/3] Patch #9: is_blackwell_class()")

# 9a. Add helper function to cuda.py
cuda_py = os.path.join(VLLM_PKG, "platforms", "cuda.py")
with open(cuda_py) as f:
    src = f.read()

if "def _is_blackwell_class" not in src:
    HELPER_FUNC = '''

def _is_blackwell_class(device_capability: "DeviceCapability") -> bool:
    """Check if a device capability represents a Blackwell-class GPU.

    Blackwell architecture includes:
    - SM100/SM101: B100, B200 (major=10)
    - SM120/SM121: GB10 DGX Spark (major=12)

    Note: SM11x may be used by future Blackwell variants.
    """
    return device_capability.major in (10, 11, 12)

'''
    if "class CudaPlatformBase" in src:
        src = src.replace("class CudaPlatformBase", HELPER_FUNC + "class CudaPlatformBase", 1)
        with open(cuda_py, "w") as f:
            f.write(src)
        applied.append("#9a: _is_blackwell_class helper")
    else:
        skipped.append("#9a: CudaPlatformBase not found")
else:
    skipped.append("#9a: _is_blackwell_class already exists")

# 9b. Add classmethod to CudaPlatformBase
with open(cuda_py) as f:
    src = f.read()

if "def is_blackwell_class" not in src:
    insert_after = "    def is_device_capability_family"
    if insert_after in src:
        idx = src.index(insert_after)
        next_def = src.find("\n    @", idx + len(insert_after))
        if next_def == -1:
            next_def = src.find("\n    def ", idx + len(insert_after))

        CLASSMETHOD = '''
    @classmethod
    def is_blackwell_class(cls, device_id: int = 0) -> bool:
        """Check if device is a Blackwell-class GPU (SM10x/SM11x/SM12x)."""
        return _is_blackwell_class(cls.get_device_capability(device_id))

'''
        if next_def > 0:
            src = src[:next_def] + CLASSMETHOD + src[next_def:]
            with open(cuda_py, "w") as f:
                f.write(src)
            applied.append("#9b: is_blackwell_class classmethod")
        else:
            skipped.append("#9b: insertion point not found")
    else:
        skipped.append("#9b: is_device_capability_family not found")
else:
    skipped.append("#9b: is_blackwell_class already exists")

# 9c. Add is_blackwell_class to Platform interface (interface.py)
interface_py = os.path.join(VLLM_PKG, "platforms", "interface.py")
with open(interface_py) as f:
    iface_src = f.read()

if "def is_blackwell_class" not in iface_src:
    marker = "    def is_device_capability_family"
    if marker in iface_src:
        idx = iface_src.index(marker)
        next_def = iface_src.find("\n    @", idx + len(marker))
        if next_def == -1:
            next_def = iface_src.find("\n    def ", idx + len(marker))
        IFACE_METHOD = '''
    @classmethod
    def is_blackwell_class(cls, device_id: int = 0) -> bool:
        """Check if device is a Blackwell-class GPU (SM10x/SM11x/SM12x)."""
        cap = cls.get_device_capability(device_id)
        return cap is not None and cap.major in (10, 11, 12)

'''
        if next_def > 0:
            iface_src = iface_src[:next_def] + IFACE_METHOD + iface_src[next_def:]
            with open(interface_py, "w") as f:
                f.write(iface_src)
            applied.append("#9c: is_blackwell_class in Platform interface")
        else:
            skipped.append("#9c: interface.py insertion point not found")
    else:
        skipped.append("#9c: is_device_capability_family not in interface.py")
else:
    skipped.append("#9c: is_blackwell_class already in interface.py")

# 9d-9g. Replace is_device_capability_family(100) with is_blackwell_class()
replacements = [
    (
        os.path.join(VLLM_PKG, "model_executor", "layers", "batch_invariant.py"),
        "        current_platform.is_device_capability_family(100)\n        or current_platform.is_device_capability(80)",
        "        current_platform.is_blackwell_class()\n        or current_platform.is_device_capability(80)",
        "#9d: batch_invariant.py",
    ),
    (
        os.path.join(VLLM_PKG, "model_executor", "layers", "fused_moe", "batched_deep_gemm_moe.py"),
        "            and current_platform.is_device_capability_family(100)\n        )",
        "            and current_platform.is_blackwell_class()\n        )",
        "#9e: batched_deep_gemm_moe.py",
    ),
]
for path, old, new, label in replacements:
    if os.path.exists(path):
        patch_file(path, old, new, label)
    else:
        skipped.append(f"{label} (file not found)")

# mxfp4.py - multiple replacements
mxfp4_py = os.path.join(VLLM_PKG, "model_executor", "layers", "quantization", "mxfp4.py")
if os.path.exists(mxfp4_py):
    with open(mxfp4_py) as f:
        mxfp4_src = f.read()
    mxfp4_orig = mxfp4_src
    mxfp4_src = mxfp4_src.replace(
        "            current_platform.is_device_capability_family(100)\n            and has_flashinfer()\n            and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
        "            current_platform.is_blackwell_class()\n            and has_flashinfer()\n            and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
        1)
    mxfp4_src = mxfp4_src.replace(
        '        elif current_platform.is_device_capability_family(100) and has_flashinfer():\n            logger.info_once(\n                "Using FlashInfer MXFP4 BF16 backend for SM100',
        '        elif current_platform.is_blackwell_class() and has_flashinfer():\n            logger.info_once(\n                "Using FlashInfer MXFP4 BF16 backend for Blackwell-class GPU',
        1)
    if mxfp4_src != mxfp4_orig:
        with open(mxfp4_py, "w") as f:
            f.write(mxfp4_src)
        applied.append("#9f: mxfp4.py is_blackwell_class")
    else:
        skipped.append("#9f: mxfp4.py (no changes needed)")

# mla_attention.py
mla_py = os.path.join(VLLM_PKG, "model_executor", "layers", "attention", "mla_attention.py")
if os.path.exists(mla_py):
    patch_file(mla_py,
        "        and current_platform.is_device_capability_family(100)\n        and has_nvidia_artifactory()",
        "        and current_platform.is_blackwell_class()\n        and has_nvidia_artifactory()",
        "#9g: mla_attention.py cudnn_prefill")

# flashmla.py - update error message (SM12x excluded from FlashMLA Sparse)
flashmla_py = os.path.join(VLLM_PKG, "v1", "attention", "ops", "flashmla.py")
if os.path.exists(flashmla_py):
    patch_file(flashmla_py,
        "        current_platform.is_device_capability_family(90)\n        or current_platform.is_device_capability_family(100)\n    ):\n        return (\n            False,\n            \"FlashMLA Sparse is only supported on Hopper and Blackwell devices.\"",
        '        current_platform.is_device_capability_family(90)\n        or current_platform.is_device_capability_family(100)\n    ):\n        return (\n            False,\n            "FlashMLA Sparse is only supported on SM90 (Hopper) "\n            "and SM100 (Blackwell B200/GB200)."',
        "#9h: flashmla.py error message")


# ============================================================
# Patch #3: Split has_flashinfer_nvfp4
# ============================================================
print("\n[2/3] Patch #3: Split has_flashinfer_nvfp4")
fi_py = os.path.join(VLLM_PKG, "utils", "flashinfer.py")

with open(fi_py) as f:
    fi_src = f.read()
fi_orig = fi_src

OLD_CUTLASS_CHECK = '''def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True'''

NEW_CUTLASS_CHECK = '''def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE engine is available.

    Only checks for the core CUTLASS MoE entry point. FP4-specific
    utilities are checked separately via has_flashinfer_nvfp4().
    """
    if not has_flashinfer_moe():
        return False

    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_nvfp4() -> bool:
    """Return `True` if FlashInfer NVFP4 quantization utilities are available."""
    required_functions = [
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True'''

if OLD_CUTLASS_CHECK in fi_src:
    fi_src = fi_src.replace(OLD_CUTLASS_CHECK, NEW_CUTLASS_CHECK, 1)
    applied.append("#3a: Split has_flashinfer_nvfp4")
else:
    skipped.append("#3a: cutlass_fused_moe check (already modified or not found)")

if '"has_flashinfer_cutlass_fused_moe",' in fi_src and '"has_flashinfer_nvfp4",' not in fi_src:
    fi_src = fi_src.replace(
        '"has_flashinfer_cutlass_fused_moe",',
        '"has_flashinfer_cutlass_fused_moe",\n    "has_flashinfer_nvfp4",',
        1)
    applied.append("#3b: Added has_flashinfer_nvfp4 to exports")

if "import functools" not in fi_src:
    fi_src = "import functools\n" + fi_src
    applied.append("#3c: Added functools import")

if fi_src != fi_orig:
    with open(fi_py, "w") as f:
        f.write(fi_src)

# 3d. Gate nvfp4 in flashinfer_cutlass_moe.py
fimoe_py = os.path.join(VLLM_PKG, "model_executor", "layers", "fused_moe", "flashinfer_cutlass_moe.py")
with open(fimoe_py) as f:
    fimoe_src = f.read()

if "has_flashinfer_nvfp4" not in fimoe_src:
    fimoe_src = fimoe_src.replace(
        "from vllm.utils.flashinfer import (\n    flashinfer_cutlass_fused_moe,\n    has_flashinfer_cutlass_fused_moe,\n)",
        "from vllm.utils.flashinfer import (\n    flashinfer_cutlass_fused_moe,\n    has_flashinfer_cutlass_fused_moe,\n    has_flashinfer_nvfp4,\n)",
        1)
    fimoe_src = fimoe_src.replace(
        "                and p.has_device_capability(100)\n            )\n        )",
        "                and p.has_device_capability(100)\n                and (scheme != (kNvfp4Static, kNvfp4Dynamic) or has_flashinfer_nvfp4())\n            )\n        )",
        1)
    with open(fimoe_py, "w") as f:
        f.write(fimoe_src)
    applied.append("#3d: Gate nvfp4 scheme on has_flashinfer_nvfp4()")
else:
    skipped.append("#3d: has_flashinfer_nvfp4 already present")


# ============================================================
# Patch #6: Auto-configure TRITON_PTXAS_PATH
# ============================================================
print("\n[3/3] Patch #6: Auto-configure TRITON_PTXAS_PATH")
triton_importing_py = os.path.join(VLLM_PKG, "triton_utils", "importing.py")

with open(triton_importing_py) as f:
    ti_src = f.read()

if "_configure_triton_ptxas_for_new_gpus" not in ti_src:
    if "import shutil" not in ti_src:
        ti_src = ti_src.replace("import os\n", "import os\nimport shutil\nimport subprocess\n", 1)

    AUTO_CONFIG_FUNC = '''

def _configure_triton_ptxas_for_new_gpus():
    """Auto-configure TRITON_PTXAS_PATH for SM>=110 GPUs."""
    if os.environ.get("TRITON_PTXAS_PATH"):
        return

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    system_ptxas_paths = [
        os.path.join(cuda_home, "bin", "ptxas"),
        "/usr/local/cuda/bin/ptxas",
        shutil.which("ptxas"),
    ]

    system_ptxas = None
    for path in system_ptxas_paths:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            system_ptxas = path
            break

    if not system_ptxas:
        return

    try:
        from triton.backends import backends
        nvidia_backend = backends.get("nvidia")
        if nvidia_backend is None or nvidia_backend.driver is None:
            return
        if not nvidia_backend.driver.is_active():
            return
        driver_instance = nvidia_backend.driver()
        target = driver_instance.get_current_target()
        if target.arch >= 110:
            result = subprocess.run(
                [system_ptxas, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                major, minor = divmod(target.arch, 10)
                logger.info(
                    "Detected GPU arch=%d (SM%d.%d). "
                    "Configuring TRITON_PTXAS_PATH=%s",
                    target.arch, major, minor, system_ptxas,
                )
    except Exception as e:
        logger.debug("Failed to auto-configure TRITON_PTXAS_PATH: %s", e)


_configure_triton_ptxas_for_new_gpus()

'''
    ti_src = ti_src.replace("\nHAS_TRITON", AUTO_CONFIG_FUNC + "HAS_TRITON", 1)

    with open(triton_importing_py, "w") as f:
        f.write(ti_src)
    applied.append("#6: Auto-configure TRITON_PTXAS_PATH")
else:
    skipped.append("#6: already present")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SM121 PATCH SUMMARY")
print("=" * 60)
print(f"\nApplied ({len(applied)}):")
for a in applied:
    print(f"  + {a}")
if skipped:
    print(f"\nSkipped ({len(skipped)}):")
    for s in skipped:
        print(f"  - {s}")

if not applied:
    print("\n  All patches already applied.")
print()
