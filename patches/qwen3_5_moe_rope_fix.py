"""
Fix: ignore_keys_at_rope_validation list -> set in qwen3_5_moe.py

transformers 5.x uses the | (union) operator on ignore_keys_at_rope_validation,
which requires a set, not a list. This patch converts the list literal to a set.

Tracks: https://github.com/vllm-project/vllm/issues/XXXXX
"""
import glob
import sys

patterns = [
    "/usr/local/lib/python3.*/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py",
]

target = None
for pat in patterns:
    matches = glob.glob(pat)
    if matches:
        target = matches[0]
        break

if not target:
    print("qwen3_5_moe.py not found, skipping rope fix.")
    sys.exit(0)

with open(target) as f:
    src = f.read()

old = (
    '        kwargs["ignore_keys_at_rope_validation"] = [\n'
    '            "mrope_section",\n'
    '            "mrope_interleaved",\n'
    '        ]\n'
)
new = (
    '        kwargs["ignore_keys_at_rope_validation"] = {\n'
    '            "mrope_section",\n'
    '            "mrope_interleaved",\n'
    '        }\n'
)

if old not in src:
    print("Already patched or anchor not found, skipping.")
else:
    src = src.replace(old, new, 1)
    with open(target, "w") as f:
        f.write(src)
    print(f"Fixed: ignore_keys_at_rope_validation list -> set in {target}")
