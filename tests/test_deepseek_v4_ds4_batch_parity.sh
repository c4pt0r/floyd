#!/bin/sh
set -eu

model=$1
tmp=$(mktemp -d /tmp/floyd-ds4-batch-parity.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

prompt='请写一个至少五百字的中文科幻故事，直接开始正文。'
names='Qcur,KVcur,kqv_out,hc_attn_post,hc_ffn_post'

env DS4_LOCK_FILE="$tmp/greedy.lock" \
    DS4_METAL_GRAPH_DUMP_PREFIX="$tmp/greedy" \
    DS4_METAL_GRAPH_DUMP_NAME="$names" \
    DS4_METAL_GRAPH_DUMP_LAYER=all \
    ./floyd run --model "$model" --prompt "$prompt" --ctx 256 \
    --ngen 5 --draft 1 --trace >"$tmp/greedy.out" 2>"$tmp/greedy.trace"

env FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD=0 \
    DS4_LOCK_FILE="$tmp/spec.lock" \
    DS4_METAL_GRAPH_DUMP_PREFIX="$tmp/spec" \
    DS4_METAL_GRAPH_DUMP_NAME="$names" \
    DS4_METAL_GRAPH_DUMP_LAYER=all \
    ./floyd run --model "$model" --prompt "$prompt" --ctx 256 \
    --ngen 5 --draft 4 --trace >"$tmp/spec.out" 2>"$tmp/spec.trace"

grep '^DEEPSEEK_V4_TOKEN ' "$tmp/greedy.trace" >"$tmp/greedy.ids"
grep '^DEEPSEEK_V4_TOKEN ' "$tmp/spec.trace" >"$tmp/spec.ids"
diff -u "$tmp/greedy.ids" "$tmp/spec.ids"

prompt_tokens=$(sed -n \
    's/^DEEPSEEK_V4_PERF .*prompt_tokens=\([0-9][0-9]*\).*/\1/p' \
    "$tmp/greedy.trace")
test -n "$prompt_tokens"
verify_pos=$((prompt_tokens + 1))

python3 - "$tmp" "$verify_pos" 4 <<'PY'
from array import array
import os
import sys

root = sys.argv[1]
position = int(sys.argv[2])
rows = int(sys.argv[3])


def read_f32(path):
    values = array("f")
    with open(path, "rb") as source:
        values.fromfile(source, os.path.getsize(path) // values.itemsize)
    return values


for name in ("Qcur", "KVcur", "kqv_out", "hc_attn_post", "hc_ffn_post"):
    batch = read_f32(f"{root}/spec_verify_{name}-0_pos{position}.bin")
    offset = 0
    for row in range(rows):
        greedy = read_f32(f"{root}/greedy_{name}-0_pos{position + row}.bin")
        actual = batch[offset:offset + len(greedy)]
        if len(actual) != len(greedy):
            raise SystemExit(f"{name}: verifier row {row} is truncated")
        maximum = max(abs(value - expected)
                      for value, expected in zip(actual, greedy))
        print(f"DeepSeek V4 DS4 batch parity {name} row {row}: "
              f"max_abs={maximum:.9g}")
        if maximum != 0.0:
            raise SystemExit(f"{name} row {row}: batch/decode reduction order differs")
        offset += len(greedy)

for layer in range(43):
    batch = read_f32(f"{root}/spec_verify_hc_ffn_post-{layer}_pos{position}.bin")
    offset = 0
    for row in range(rows):
        greedy = read_f32(
            f"{root}/greedy_hc_ffn_post-{layer}_pos{position + row}.bin")
        actual = batch[offset:offset + len(greedy)]
        maximum = max(abs(value - expected)
                      for value, expected in zip(actual, greedy))
        print(f"DeepSeek V4 DS4 layer parity {layer} row {row}: "
              f"max_abs={maximum:.9g}")
        if maximum != 0.0:
            raise SystemExit(f"layer {layer} row {row}: batch/decode block state differs")
        offset += len(greedy)
PY
