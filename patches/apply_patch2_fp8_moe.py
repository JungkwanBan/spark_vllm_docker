#!/usr/bin/env python3
"""
Apply Patch #2: Enable FP8 Block-Scale MoE on SM121 (GB10).
Requires FlashInfer >= 0.6.4 which supports FP8 block scaling on Blackwell.

Run inside container:
  docker exec <container> python3 /tmp/apply_patch2_fp8_moe.py
"""
import os
import glob

VLLM_PKG = None
for p in glob.glob("/usr/local/lib/python3.*/dist-packages/vllm"):
    VLLM_PKG = p
    break

if not VLLM_PKG:
    print("ERROR: vllm package not found")
    exit(1)

print(f"vLLM package: {VLLM_PKG}")

# Verify FlashInfer version
try:
    import flashinfer
    fi_ver = flashinfer.__version__
    print(f"FlashInfer version: {fi_ver}")
    if fi_ver < "0.6.4":
        print(f"ERROR: FlashInfer >= 0.6.4 required, got {fi_ver}")
        exit(1)
except ImportError:
    print("ERROR: FlashInfer not installed")
    exit(1)

# Check for FP8 block-scale MoE support
try:
    from flashinfer.fused_moe import cutlass_fused_moe
    print("OK: flashinfer.fused_moe.cutlass_fused_moe available")
except (ImportError, AttributeError) as e:
    print(f"WARNING: cutlass_fused_moe not available: {e}")

fimoe_py = os.path.join(VLLM_PKG, "model_executor", "layers", "fused_moe", "flashinfer_cutlass_moe.py")

with open(fimoe_py) as f:
    src = f.read()

# The patch: allow SM121 (capability 12.x) to use FP8 block-scale MoE
# Original: only SM90 (Hopper)
# Patched: SM90 or SM12x (Blackwell-class including GB10)
OLD = "                and p.is_device_capability(90)\n            )\n            # nvfp4"
NEW = "                and (p.is_device_capability(90) or p.is_device_capability_family(120))\n            )\n            # nvfp4"

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    with open(fimoe_py, "w") as f:
        f.write(src)
    print("APPLIED: FP8 block-scale MoE enabled for SM121")
elif NEW in src:
    print("SKIPPED: Already applied")
else:
    # Try alternative anchor patterns
    # After round2 patches, the code might look different due to is_blackwell_class changes
    print("WARNING: Standard anchor not found. Checking current file state...")

    # Look for the capability check pattern
    import re
    cap_pattern = re.search(r'and p\.is_device_capability\(90\)', src)
    if cap_pattern:
        print(f"Found capability check at position {cap_pattern.start()}")
        # Show context
        start = max(0, cap_pattern.start() - 200)
        end = min(len(src), cap_pattern.end() + 200)
        print(f"Context:\n{src[start:end]}")
    else:
        # Maybe is_blackwell_class was used instead
        if "is_blackwell_class" in src:
            print("is_blackwell_class already in use - patch may already be equivalent")
        else:
            print("ERROR: Cannot find capability check pattern")
            exit(1)

print("\nDone.")
