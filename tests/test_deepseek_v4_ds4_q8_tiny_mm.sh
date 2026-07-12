#!/bin/sh
set -eu

model=$1
tmp=$(mktemp -d /tmp/floyd-ds4-q8-tiny-mm.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

prompt='请写一个至少五百字的中文科幻故事，直接开始正文。'

run_case() {
    name=$1
    enabled=$2
    env SNAP="$model" CHAT=1 DRAFT=4 \
        PROMPT="$prompt" CTX=256 NGEN=5 DEEPSEEK_V4_CHAT_TRACE=1 \
        DS4_LOCK_FILE="$tmp/$name.lock" \
        DS4_METAL_Q8_TINY_MM="$enabled" \
        DS4_METAL_Q8_TINY_MM_TRACE=1 \
        DS4_METAL_GRAPH_DUMP_PREFIX="$tmp/$name" \
        DS4_METAL_GRAPH_DUMP_NAME=Qcur \
        DS4_METAL_GRAPH_DUMP_LAYER=0 \
        ./floyd >"$tmp/$name.out" 2>"$tmp/$name.trace"
    grep '^DEEPSEEK_V4_TOKEN ' "$tmp/$name.trace" | \
        sed 's/.* /token /' >"$tmp/$name.ids"
}

run_case baseline 0
run_case tiny_mm 1

diff -u "$tmp/baseline.ids" "$tmp/tiny_mm.ids"
grep -q '^ds4: Metal Q8 tiny MM calls=[1-9][0-9]*$' "$tmp/tiny_mm.trace"

python3 - "$tmp" <<'PY'
from array import array
from glob import glob
import os
import sys

root = sys.argv[1]


def first_qcur(prefix):
    paths = sorted(glob(f"{root}/{prefix}_verify_Qcur-0_pos*.bin"))
    if not paths:
        raise SystemExit(f"missing {prefix} verifier Qcur dump")
    values = array("f")
    with open(paths[0], "rb") as source:
        values.fromfile(source, os.path.getsize(paths[0]) // values.itemsize)
    return values


baseline = first_qcur("baseline")
candidate = first_qcur("tiny_mm")
if len(candidate) != len(baseline):
    raise SystemExit("tiny MPP Qcur shape differs from baseline")
maximum = max(abs(actual - expected)
              for actual, expected in zip(candidate, baseline))
print(f"DeepSeek V4 DS4 Q8 tiny MM Qcur parity: max_abs={maximum:.9g}")
if maximum >= 5e-3:
    raise SystemExit("tiny MM Qcur error exceeds 5e-3")
PY
