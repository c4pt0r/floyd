#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
DS4_REPO=${DS4_REPO:-https://github.com/antirez/ds4.git}
DS4_REV=${DS4_REV:-80ebbc396aee40eedc1d829222f3362d10fa4c6c}
DS4_DIR=${DS4_DIR:-"$ROOT/.deps/ds4"}
DS4_QUANT=${DS4_QUANT:-q2-imatrix}

test "$#" -ge 1 && test "$#" -le 2 || {
    echo "usage: $0 <DeepSeek-V4-checkpoint> [output-directory]" >&2
    exit 2
}
MODEL=$1
OUTPUT=${2:-"$MODEL-DS4"}

test -f "$MODEL/config.json" || {
    echo "checkpoint is missing config.json: $MODEL" >&2
    exit 2
}
python3 - "$MODEL/config.json" <<'PY'
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

if test ! -f "$DS4_DIR/download_model.sh"; then
    mkdir -p "$(dirname "$DS4_DIR")"
    git clone --filter=blob:none "$DS4_REPO" "$DS4_DIR"
fi
if test "${DS4_SKIP_REV_CHECK:-0}" != 1; then
    current=$(git -C "$DS4_DIR" rev-parse HEAD 2>/dev/null || true)
    if test "$current" != "$DS4_REV"; then
        git -C "$DS4_DIR" fetch --depth 1 "$DS4_REPO" "$DS4_REV"
        git -C "$DS4_DIR" checkout --detach "$DS4_REV"
    fi
fi

DS4_GGUF_DIR="$OUTPUT_REAL" "$DS4_DIR/download_model.sh" "$DS4_QUANT"
if test "${DS4_WITH_MTP:-0}" = 1; then
    DS4_GGUF_DIR="$OUTPUT_REAL" "$DS4_DIR/download_model.sh" mtp
fi

printf '%s\n' "$DS4_REV" > "$OUTPUT_REAL/ds4-revision.txt"
echo "DeepSeek V4 DS4 GGUF: $OUTPUT_REAL"
