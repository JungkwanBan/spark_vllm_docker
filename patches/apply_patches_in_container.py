#!/usr/bin/env python3
"""
Apply patches #1 (AOT cache fix), #4 (nogds force), #5 (MoE tuning)
directly to vLLM installation inside container.
Run with: docker exec <container> python3 /tmp/apply_patches_in_container.py
"""
import os
import json
import glob

VLLM_PKG = None
for p in glob.glob("/usr/local/lib/python3.*/dist-packages/vllm"):
    VLLM_PKG = p
    break

if not VLLM_PKG:
    print("ERROR: vllm package not found")
    exit(1)

print(f"vLLM package: {VLLM_PKG}")

# ============================================================
# Patch #1: AOT compile cache fix (caching.py)
# ============================================================
caching_py = os.path.join(VLLM_PKG, "compilation", "caching.py")
print(f"\n[1/3] AOT cache fix: {caching_py}")

with open(caching_py) as f:
    src = f.read()

OLD_AOT = """\
        for node in state["graph_module"].graph.nodes:
            node.meta.pop("source_fn_stack", None)
            node.meta.pop("nn_module_stack", None)
        for name, submod in state["graph_module"].named_children():
            if hasattr(submod, "graph"):
                for node in submod.graph.nodes:
                    node.meta.pop("source_fn_stack", None)
                    node.meta.pop("nn_module_stack", None)"""

NEW_AOT = """\
        def _strip_unpicklable_node_meta(graph_module: torch.fx.GraphModule) -> None:
            def _has_node_ref(obj: Any, depth: int = 0) -> bool:
                if depth > 8:
                    return False
                if isinstance(obj, torch.fx.Node):
                    return True
                if isinstance(obj, dict):
                    return any(_has_node_ref(v, depth + 1) for v in obj.values())
                if isinstance(obj, (list, tuple)):
                    return any(_has_node_ref(v, depth + 1) for v in obj)
                return False

            for node in graph_module.graph.nodes:
                keys_to_remove = [
                    k for k, v in node.meta.items()
                    if _has_node_ref(v)
                ]
                for k in keys_to_remove:
                    del node.meta[k]
            for _name, submod in graph_module.named_children():
                if hasattr(submod, "graph"):
                    for node in submod.graph.nodes:
                        keys_to_remove = [
                            k for k, v in node.meta.items()
                            if _has_node_ref(v)
                        ]
                        for k in keys_to_remove:
                            del node.meta[k]

        _strip_unpicklable_node_meta(state["graph_module"])"""

if OLD_AOT in src:
    src = src.replace(OLD_AOT, NEW_AOT, 1)
    # Also add the fallback Node handler in reducer_override
    OLD_REDUCER = "            return graph_reducer_override(self, obj)"
    NEW_REDUCER = """\
            if isinstance(obj, torch.fx.Node):
                return type(None), ()
            return graph_reducer_override(self, obj)"""
    src = src.replace(OLD_REDUCER, NEW_REDUCER, 1)
    with open(caching_py, "w") as f:
        f.write(src)
    print("  APPLIED: AOT cache fix")
else:
    print("  SKIPPED: Already patched or anchor not found")

# ============================================================
# Patch #4: Force nogds=True (weight_utils.py)
# ============================================================
weight_utils_py = os.path.join(VLLM_PKG, "model_executor", "model_loader", "weight_utils.py")
print(f"\n[2/3] nogds force: {weight_utils_py}")

with open(weight_utils_py) as f:
    src = f.read()

OLD_NOGDS = "    nogds = pg.size() > 1"
NEW_NOGDS = "    nogds = True  # GB10 does not support GDS"

if OLD_NOGDS in src:
    src = src.replace(OLD_NOGDS, NEW_NOGDS, 1)
    with open(weight_utils_py, "w") as f:
        f.write(src)
    print("  APPLIED: nogds=True forced")
else:
    print("  SKIPPED: Already patched or anchor not found")

# ============================================================
# Patch #5: GB10 MoE tuning configs
# ============================================================
configs_dir = os.path.join(VLLM_PKG, "model_executor", "layers", "fused_moe", "configs")
print(f"\n[3/3] MoE tuning configs: {configs_dir}")

MOE_CONFIG = {
    "triton_version": "3.5.0",
    "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    "2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    "4": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    "8": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    "16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    "24": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    "32": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    "48": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 4},
    "64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 4},
    "96": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
    "128": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 4},
    "256": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 3},
    "512": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 3},
    "1024": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
    "1536": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4},
    "2048": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4},
    "3072": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4},
    "4096": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 4},
}

for expert_count in [256, 512]:
    fname = f"E={expert_count},N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json"
    fpath = os.path.join(configs_dir, fname)
    with open(fpath, "w") as f:
        json.dump(MOE_CONFIG, f, indent=4)
    print(f"  INSTALLED: {fname}")

print("\nAll patches applied successfully.")
