#!/bin/sh
set -eu

model=$1
tmp=$(mktemp -d /tmp/floyd-ds4-batch-parity.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

prompt='请写一个至少五百字的中文科幻故事，直接开始正文。'
names='Qcur,KVcur,kqv_out,hc_attn_post'

env SNAP="$model" CHAT=1 DSPARK_SPEC=0 \
    PROMPT="$prompt" CTX=256 NGEN=4 DEEPSEEK_V4_CHAT_TRACE=1 \
    DS4_LOCK_FILE="$tmp/greedy.lock" \
    DS4_METAL_GRAPH_DUMP_PREFIX="$tmp/greedy" \
    DS4_METAL_GRAPH_DUMP_NAME="$names" \
    DS4_METAL_GRAPH_DUMP_LAYER=0 \
    ./floyd >"$tmp/greedy.out" 2>"$tmp/greedy.trace"

env SNAP="$model" CHAT=1 DRAFT=3 \
    PROMPT="$prompt" CTX=256 NGEN=4 DEEPSEEK_V4_CHAT_TRACE=1 \
    DS4_LOCK_FILE="$tmp/spec.lock" \
    DS4_METAL_GRAPH_DUMP_PREFIX="$tmp/spec" \
    DS4_METAL_GRAPH_DUMP_NAME="$names" \
    DS4_METAL_GRAPH_DUMP_LAYER=0 \
    ./floyd >"$tmp/spec.out" 2>"$tmp/spec.trace"

grep '^DEEPSEEK_V4_TOKEN ' "$tmp/greedy.trace" >"$tmp/greedy.ids"
grep '^DEEPSEEK_V4_TOKEN ' "$tmp/spec.trace" >"$tmp/spec.ids"
diff -u "$tmp/greedy.ids" "$tmp/spec.ids"

prompt_tokens=$(sed -n \
    's/^DEEPSEEK_V4_PERF .*prompt_tokens=\([0-9][0-9]*\).*/\1/p' \
    "$tmp/greedy.trace")
test -n "$prompt_tokens"
verify_pos=$((prompt_tokens + 1))

python3 - "$tmp" "$verify_pos" <<'PY'
from array import array
import os
import sys

root = sys.argv[1]
position = int(sys.argv[2])


def read_f32(path):
    values = array("f")
    with open(path, "rb") as source:
        values.fromfile(source, os.path.getsize(path) // values.itemsize)
    return values


for name in ("Qcur", "KVcur", "kqv_out", "hc_attn_post"):
    greedy = read_f32(f"{root}/greedy_{name}-0_pos{position}.bin")
    batch = read_f32(f"{root}/spec_verify_{name}-0_pos{position}.bin")
    if len(batch) < len(greedy):
        raise SystemExit(f"{name}: verifier row is truncated")
    maximum = max(abs(actual - expected)
                  for actual, expected in zip(batch[:len(greedy)], greedy))
    print(f"DeepSeek V4 DS4 batch parity {name}: max_abs={maximum:.9g}")
    if maximum != 0.0:
        raise SystemExit(f"{name}: batch/decode reduction order differs")
PY
