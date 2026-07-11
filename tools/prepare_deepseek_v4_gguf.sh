#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
LLAMA_CPP_REPO=${LLAMA_CPP_REPO:-https://github.com/cchuter/llama.cpp.git}
LLAMA_CPP_REV=${LLAMA_CPP_REV:-19b63dc368dfef6db6783e5ba3143927b7ed1c96}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"$ROOT/.deps/llama.cpp"}
PYTHON=${PYTHON:-python3}
DEEPSEEK_V4_EXPERT_WORKERS=${DEEPSEEK_V4_EXPERT_WORKERS:-8}
LLAMA_PATCHES="
$ROOT/patches/llama.cpp/deepseek-v4-native-mxfp4-converter.patch
$ROOT/patches/llama.cpp/deepseek-v4-context-reserve.patch
"

usage() {
    echo "usage: $0 <DeepSeek-V4-checkpoint> [output-directory]" >&2
    exit 2
}

test "$#" -ge 1 && test "$#" -le 2 || usage
MODEL=$1
OUTPUT=${2:-"$MODEL-GGUF"}

test -f "$MODEL/config.json" || {
    echo "checkpoint is missing config.json: $MODEL" >&2
    exit 2
}
test -f "$MODEL/model.safetensors.index.json" || {
    echo "checkpoint is missing model.safetensors.index.json: $MODEL" >&2
    exit 2
}

"$PYTHON" - "$MODEL/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
if config.get("model_type") != "deepseek_v4":
    raise SystemExit("checkpoint config must declare model_type deepseek_v4")
PY

mkdir -p "$OUTPUT"
MODEL_REAL=$(CDPATH= cd -- "$MODEL" && pwd -P)
OUTPUT_REAL=$(CDPATH= cd -- "$OUTPUT" && pwd -P)
case "$OUTPUT_REAL/" in
    "$MODEL_REAL/"*)
        echo "output directory must not be inside the checkpoint: $OUTPUT_REAL" >&2
        exit 2
        ;;
esac

if test ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py"; then
    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
    git clone --filter=blob:none "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

if test "${LLAMA_CPP_SKIP_REV_CHECK:-0}" != 1; then
    current=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || true)
    if test "$current" != "$LLAMA_CPP_REV"; then
        git -C "$LLAMA_CPP_DIR" fetch --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_REV"
        git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_REV"
    fi
    current=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
    test "$current" = "$LLAMA_CPP_REV" || {
        echo "llama.cpp revision mismatch: expected $LLAMA_CPP_REV, got $current" >&2
        exit 2
    }
    for patch in $LLAMA_PATCHES; do
        if git -C "$LLAMA_CPP_DIR" apply --reverse --check "$patch" 2>/dev/null; then
            :
        elif git -C "$LLAMA_CPP_DIR" apply --check "$patch"; then
            git -C "$LLAMA_CPP_DIR" apply "$patch"
        else
            echo "llama.cpp patch does not apply: $patch" >&2
            exit 2
        fi
    done
fi

set -- "$MODEL_REAL" \
    --outfile "$OUTPUT_REAL/model-{ftype}.gguf" \
    --outtype auto \
    --deepseek4-expert-outtypes mxfp4 \
    --deepseek4-expert-workers "$DEEPSEEK_V4_EXPERT_WORKERS" \
    --split-max-size 48G
"$PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$@"

first_shard=
for candidate in "$OUTPUT_REAL"/model-*-00001-of-*.gguf "$OUTPUT_REAL"/model-*.gguf; do
    if test -f "$candidate"; then
        first_shard=$candidate
        break
    fi
done
test -n "$first_shard" || {
    echo "conversion completed without a discoverable GGUF: $OUTPUT_REAL" >&2
    exit 1
}

printf '%s\n' "$LLAMA_CPP_REV" > "$OUTPUT_REAL/llama.cpp-revision.txt"
printf 'DeepSeek V4 GGUF: %s\n' "$first_shard"
