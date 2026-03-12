#!/usr/bin/env python3
"""
Apply patches #2, #3, #6, #7, #9 from feature/sm121-gb10-support branch.
Run inside container:
  docker exec <container> python3 /tmp/apply_patches_round2.py
"""
import os
import glob
import functools
import re

VLLM_PKG = None
for p in glob.glob("/usr/local/lib/python3.*/dist-packages/vllm"):
    VLLM_PKG = p
    break

if not VLLM_PKG:
    print("ERROR: vllm package not found")
    exit(1)

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
# Patch #9: Add is_blackwell_class() to cuda.py
# Must be applied BEFORE other patches that reference it
# ============================================================
print("\n[1/5] Patch #9: Add is_blackwell_class() to cuda.py")
cuda_py = os.path.join(VLLM_PKG, "platforms", "cuda.py")

with open(cuda_py) as f:
    src = f.read()

# Add the standalone helper function if not present
if "def _is_blackwell_class" not in src:
    # Find a good insertion point - before the CudaPlatform class
    # Insert after imports, before class definition
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
    # Insert before the class definition
    insert_marker = "class CudaPlatformBase"
    if insert_marker in src:
        src = src.replace(insert_marker, HELPER_FUNC + insert_marker, 1)
        with open(cuda_py, "w") as f:
            f.write(src)
        applied.append("#9a: _is_blackwell_class helper function")
    else:
        skipped.append("#9a: Could not find CudaPlatformBase class")
else:
    skipped.append("#9a: _is_blackwell_class already exists")

# Add the classmethod to CudaPlatform / CudaPlatformBase
with open(cuda_py) as f:
    src = f.read()

if "def is_blackwell_class" not in src:
    # Find is_device_capability_family method and add is_blackwell_class after it
    # Look for the pattern of an existing classmethod
    insert_after = "    def is_device_capability_family"
    if insert_after in src:
        # Find the end of is_device_capability_family method
        idx = src.index(insert_after)
        # Find the next method definition after it
        next_def = src.find("\n    @", idx + len(insert_after))
        if next_def == -1:
            next_def = src.find("\n    def ", idx + len(insert_after))

        CLASSMETHOD = '''
    @classmethod
    def is_blackwell_class(cls, device_id: int = 0) -> bool:
        """Check if device is a Blackwell-class GPU.

        Blackwell architecture includes:
        - SM100/SM101: B100, B200 (major=10)
        - SM120/SM121: GB10 DGX Spark (major=12)

        Note: SM11x may be used by future Blackwell variants.
        """
        return _is_blackwell_class(cls.get_device_capability(device_id))

'''
        if next_def > 0:
            src = src[:next_def] + CLASSMETHOD + src[next_def:]
            with open(cuda_py, "w") as f:
                f.write(src)
            applied.append("#9b: is_blackwell_class classmethod")
        else:
            skipped.append("#9b: Could not find insertion point for classmethod")
    else:
        skipped.append("#9b: is_device_capability_family not found")
else:
    skipped.append("#9b: is_blackwell_class classmethod already exists")

# Now apply is_blackwell_class() replacements in various files

# --- batch_invariant.py ---
print("  -> batch_invariant.py")
bi_py = os.path.join(VLLM_PKG, "model_executor", "layers", "batch_invariant.py")
patch_file(bi_py,
    "        current_platform.is_device_capability_family(100)\n        or current_platform.is_device_capability(80)",
    "        current_platform.is_blackwell_class()\n        or current_platform.is_device_capability(80)",
    "#9c: batch_invariant.py is_blackwell_class")

# --- batched_deep_gemm_moe.py ---
print("  -> batched_deep_gemm_moe.py")
bdg_py = os.path.join(VLLM_PKG, "model_executor", "layers", "fused_moe", "batched_deep_gemm_moe.py")
patch_file(bdg_py,
    "            and current_platform.is_device_capability_family(100)\n        )",
    "            and current_platform.is_blackwell_class()\n        )",
    "#9d: batched_deep_gemm_moe supports_packed_ue8m0")

# --- mxfp4.py ---
print("  -> mxfp4.py")
mxfp4_py = os.path.join(VLLM_PKG, "model_executor", "layers", "quantization", "mxfp4.py")
# Multiple replacements needed - read, modify, write
with open(mxfp4_py) as f:
    mxfp4_src = f.read()
mxfp4_orig = mxfp4_src

# Replace capability checks that should include SM12x
# 1. MXFP4 MXFP8 CUTLASS backend check
mxfp4_src = mxfp4_src.replace(
    "            current_platform.is_device_capability_family(100)\n            and has_flashinfer()\n            and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
    "            current_platform.is_blackwell_class()\n            and has_flashinfer()\n            and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
    1)
# 2. Fallback MXFP4 BF16 backend
mxfp4_src = mxfp4_src.replace(
    '        elif current_platform.is_device_capability_family(100) and has_flashinfer():\n            logger.info_once(\n                "Using FlashInfer MXFP4 BF16 backend for SM100',
    '        elif current_platform.is_blackwell_class() and has_flashinfer():\n            logger.info_once(\n                "Using FlashInfer MXFP4 BF16 backend for Blackwell-class GPU',
    1)

if mxfp4_src != mxfp4_orig:
    with open(mxfp4_py, "w") as f:
        f.write(mxfp4_src)
    applied.append("#9e: mxfp4.py is_blackwell_class (2 replacements)")
else:
    skipped.append("#9e: mxfp4.py (no changes needed)")

# --- mla_attention.py ---
print("  -> mla_attention.py")
mla_py = os.path.join(VLLM_PKG, "model_executor", "layers", "attention", "mla_attention.py")
with open(mla_py) as f:
    mla_src = f.read()
mla_orig = mla_src

# use_cudnn_prefill: is_device_capability_family(100) -> is_blackwell_class()
mla_src = mla_src.replace(
    "        and current_platform.is_device_capability_family(100)\n        and has_nvidia_artifactory()",
    "        and current_platform.is_blackwell_class()\n        and has_nvidia_artifactory()",
    1)

if mla_src != mla_orig:
    with open(mla_py, "w") as f:
        f.write(mla_src)
    applied.append("#9f: mla_attention.py cudnn_prefill is_blackwell_class")
else:
    skipped.append("#9f: mla_attention.py (no changes needed)")

# --- flashmla.py: restrict FlashMLA Sparse to SM90/SM100 only (NOT SM12x) ---
print("  -> flashmla.py")
flashmla_py = os.path.join(VLLM_PKG, "v1", "attention", "ops", "flashmla.py")
if os.path.exists(flashmla_py):
    with open(flashmla_py) as f:
        fmla_src = f.read()
    fmla_orig = fmla_src
    # The current code may use is_blackwell_class() or is_device_capability_family(100)
    # We need to ensure SM12x is excluded - FlashMLA Sparse only supports SM90/SM100
    fmla_src = fmla_src.replace(
        "        current_platform.is_device_capability_family(90)\n        or current_platform.is_device_capability_family(100)\n    ):\n        return (\n            False,\n            \"FlashMLA Sparse is only supported on Hopper and Blackwell devices.\"",
        '        current_platform.is_device_capability_family(90)\n        or current_platform.is_device_capability_family(100)\n    ):\n        return (\n            False,\n            "FlashMLA Sparse is only supported on SM90 (Hopper) "\n            "and SM100 (Blackwell B200/GB200)."',
        1)
    if fmla_src != fmla_orig:
        with open(flashmla_py, "w") as f:
            f.write(fmla_src)
        applied.append("#9g: flashmla.py error message update")
    else:
        skipped.append("#9g: flashmla.py (no changes or already correct)")
else:
    skipped.append("#9g: flashmla.py not found")


# ============================================================
# Patch #2: FP8 Block-Scale MoE on SM121
# ============================================================
print("\n[2/5] Patch #2: FP8 Block-Scale MoE on SM121")
fimoe_py = os.path.join(VLLM_PKG, "model_executor", "layers", "fused_moe", "flashinfer_cutlass_moe.py")

patch_file(fimoe_py,
    "                and p.is_device_capability(90)\n            )\n            # nvfp4",
    "                and (p.is_device_capability(90) or p.is_device_capability_family(120))\n            )\n            # nvfp4",
    "#2: FP8 block-scale MoE on SM121")


# ============================================================
# Patch #3: Split has_flashinfer_nvfp4 from has_flashinfer_cutlass_fused_moe
# ============================================================
print("\n[3/5] Patch #3: Split has_flashinfer_nvfp4")
fi_py = os.path.join(VLLM_PKG, "utils", "flashinfer.py")

with open(fi_py) as f:
    fi_src = f.read()
fi_orig = fi_src

# 3a. Modify has_flashinfer_cutlass_fused_moe to only check core entry point
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
    utilities (fp4_quantize, nvfp4_block_scale_interleave) are checked
    separately via has_flashinfer_nvfp4() and gated by
    _supports_quant_scheme(). This allows FP8 CUTLASS MoE to work on
    architectures like SM121 (GB10) that have cutlass_fused_moe but
    may lack FP4 utilities.
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
    """Return `True` if FlashInfer NVFP4 quantization utilities are available.

    Checks for fp4_quantize and nvfp4_block_scale_interleave which are
    required for NVFP4 quantization paths but not for FP8 CUTLASS MoE.
    """
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
    applied.append("#3a: Split has_flashinfer_nvfp4 from cutlass_fused_moe check")
else:
    skipped.append("#3a: has_flashinfer_cutlass_fused_moe (already modified or not found)")

# 3b. Add has_flashinfer_nvfp4 to __all__ exports
if '"has_flashinfer_cutlass_fused_moe",' in fi_src and '"has_flashinfer_nvfp4",' not in fi_src:
    fi_src = fi_src.replace(
        '"has_flashinfer_cutlass_fused_moe",',
        '"has_flashinfer_cutlass_fused_moe",\n    "has_flashinfer_nvfp4",',
        1)
    applied.append("#3b: Added has_flashinfer_nvfp4 to exports")

# 3c. Add functools import if not present
if "import functools" not in fi_src:
    fi_src = "import functools\n" + fi_src
    applied.append("#3c: Added functools import")

if fi_src != fi_orig:
    with open(fi_py, "w") as f:
        f.write(fi_src)

# 3d. Gate nvfp4 quant scheme in flashinfer_cutlass_moe.py
with open(fimoe_py) as f:
    fimoe_src = f.read()

# Add import for has_flashinfer_nvfp4
if "has_flashinfer_nvfp4" not in fimoe_src:
    fimoe_src = fimoe_src.replace(
        "from vllm.utils.flashinfer import (\n    flashinfer_cutlass_fused_moe,\n    has_flashinfer_cutlass_fused_moe,\n)",
        "from vllm.utils.flashinfer import (\n    flashinfer_cutlass_fused_moe,\n    has_flashinfer_cutlass_fused_moe,\n    has_flashinfer_nvfp4,\n)",
        1)
    # Add nvfp4 gate to the nvfp4 quant scheme check
    fimoe_src = fimoe_src.replace(
        "                and p.has_device_capability(100)\n            )\n        )",
        "                and p.has_device_capability(100)\n                and (scheme != (kNvfp4Static, kNvfp4Dynamic) or has_flashinfer_nvfp4())\n            )\n        )",
        1)
    with open(fimoe_py, "w") as f:
        f.write(fimoe_src)
    applied.append("#3d: Gate nvfp4 scheme on has_flashinfer_nvfp4()")
else:
    skipped.append("#3d: has_flashinfer_nvfp4 import already present")


# ============================================================
# Patch #6: Auto-configure TRITON_PTXAS_PATH
# ============================================================
print("\n[4/5] Patch #6: Auto-configure TRITON_PTXAS_PATH")
triton_importing_py = os.path.join(VLLM_PKG, "triton_utils", "importing.py")

with open(triton_importing_py) as f:
    ti_src = f.read()

if "_configure_triton_ptxas_for_new_gpus" not in ti_src:
    # Add shutil and subprocess imports
    if "import shutil" not in ti_src:
        ti_src = ti_src.replace("import os\n", "import os\nimport shutil\nimport subprocess\n", 1)

    # Add the auto-configure function before HAS_TRITON
    AUTO_CONFIG_FUNC = '''

def _configure_triton_ptxas_for_new_gpus():
    """
    Configure TRITON_PTXAS_PATH for GPUs that may not be supported by
    Triton's bundled ptxas (e.g., DGX Spark sm_121a).
    """
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
        arch = target.arch

        if arch >= 110:
            try:
                result = subprocess.run(
                    [system_ptxas, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                    major, minor = divmod(arch, 10)
                    logger.info(
                        "Detected GPU with compute capability %d.%d (arch=%d). "
                        "Configuring TRITON_PTXAS_PATH=%s",
                        major, minor, arch, system_ptxas,
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                logger.debug("Cannot use system ptxas: %s", e)

    except Exception as e:
        logger.debug("Failed to auto-configure TRITON_PTXAS_PATH: %s", e)


_configure_triton_ptxas_for_new_gpus()

'''
    ti_src = ti_src.replace("\nHAS_TRITON", AUTO_CONFIG_FUNC + "HAS_TRITON", 1)

    with open(triton_importing_py, "w") as f:
        f.write(ti_src)
    applied.append("#6: Auto-configure TRITON_PTXAS_PATH for SM121")
else:
    skipped.append("#6: _configure_triton_ptxas_for_new_gpus already exists")


# ============================================================
# Patch #7: Verify CUTLASS SM121 support
# ============================================================
print("\n[5/5] Patch #7: Verify CUTLASS SM121 kernel availability")
try:
    import torch
    from vllm._custom_ops import cutlass_scaled_mm
    a = torch.randn(4, 64, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn(64, 64, device='cuda').to(torch.float8_e4m3fn).t().contiguous().t()
    scale_a = torch.ones(4, 1, dtype=torch.float32, device='cuda')
    scale_b = torch.ones(1, 64, dtype=torch.float32, device='cuda')
    r = cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)
    applied.append(f"#7: CUTLASS scaled_mm already works on SM121 (output: {r.shape})")
except Exception as e:
    skipped.append(f"#7: CUTLASS scaled_mm NOT available: {e}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PATCH SUMMARY")
print("=" * 60)
print(f"\nApplied ({len(applied)}):")
for a in applied:
    print(f"  ✓ {a}")
print(f"\nSkipped ({len(skipped)}):")
for s in skipped:
    print(f"  - {s}")
print()
