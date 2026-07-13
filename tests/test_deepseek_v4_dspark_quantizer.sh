#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
DS4_DIR=${DS4_DIR:-"$ROOT/.deps/ds4"}
QUANTIZER="$DS4_DIR/gguf-tools/deepseek4-quantize"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/floyd-dspark-quantizer.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

test -n "${DSPARK:-}" || { echo "set DSPARK=/path/to/checkpoint" >&2; exit 2; }
test -n "${DSPARK_MTP:-}" || { echo "set DSPARK_MTP=/path/to/stage0.gguf" >&2; exit 2; }

PYTHONPATH="$ROOT:$ROOT/.deps/llama.cpp/gguf-py" "$PYTHON" \
    "$ROOT/tools/deepseek_v4_dspark_manifest.py" "$DSPARK" \
    --support-gguf "$DSPARK_MTP" --template-out "$TMP/template.gguf" \
    > "$TMP/manifest.json"

make -C "$DS4_DIR/gguf-tools" deepseek4-quantize >/dev/null
"$QUANTIZER" --hf "$DSPARK" --template "$TMP/template.gguf" \
    --copy-unchanged --dry-run >/dev/null
map_tensor() {
    "$QUANTIZER" --hf "$DSPARK" --template "$TMP/template.gguf" \
        --map-tensor "$1"
}

map_imatrix() {
    "$QUANTIZER" --hf "$DSPARK" --template "$TMP/template.gguf" \
        --map-imatrix "$1"
}

test "$(map_tensor mtp.0.main_norm.weight)" = \
    "mtp.0.main_norm.weight -> mtp.0.main_norm.weight"
test "$(map_tensor mtp.1.attn_q_a.weight)" = \
    "mtp.1.attn_q_a.weight -> mtp.1.attn.wq_a.weight"
test "$(map_tensor mtp.2.ffn_down_exps.weight)" = \
    "mtp.2.ffn_down_exps.weight -> mtp.2.ffn.experts.0.w2.weight"
test "$(map_tensor mtp.2.markov_head.markov_w2.weight)" = \
    "mtp.2.markov_head.markov_w2.weight -> mtp.2.markov_head.markov_w2.weight"

test "$(map_imatrix mtp.0.ffn_gate_exps.weight)" = \
    "mtp.0.ffn_gate_exps.weight -> blk.40.ffn_gate_exps.weight"
test "$(map_imatrix mtp.1.ffn_up_exps.weight)" = \
    "mtp.1.ffn_up_exps.weight -> blk.41.ffn_up_exps.weight"
test "$(map_imatrix mtp.2.ffn_down_exps.weight)" = \
    "mtp.2.ffn_down_exps.weight -> blk.42.ffn_down_exps.weight"

confidence_probe=$(
    "$QUANTIZER" --hf "$DSPARK" --template "$TMP/template.gguf" \
        --compare-tensor mtp.2.confidence_head.proj.weight 2>/dev/null
)
printf '%s\n' "$confidence_probe" | grep -q '^generated_bytes: 17408$'
printf '%s\n' "$confidence_probe" | grep -q '^byte_compare: FAIL '

echo "DeepSeek V4 DSpark quantizer mapping: ok"
