#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PREPARE="$ROOT/tools/prepare_deepseek_v4_gguf.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/floyd-dsv4-prepare.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

MODEL="$TMP/model"
OUTPUT="$TMP/output"
LLAMA="$TMP/llama.cpp"
mkdir -p "$MODEL" "$LLAMA"

printf '%s\n' '{"model_type":"deepseek_v4"}' > "$MODEL/config.json"
printf '%s\n' '{}' > "$MODEL/model.safetensors.index.json"

cat > "$LLAMA/convert_hf_to_gguf.py" <<'PY'
import os
import pathlib
import sys

args = sys.argv[1:]
pathlib.Path(os.environ["FAKE_ARGS"]).write_text("\n".join(args) + "\n")
out = pathlib.Path(args[args.index("--outfile") + 1].replace("{ftype}", "mxfp4_moe"))
out.parent.mkdir(parents=True, exist_ok=True)
out.with_name(out.stem + "-00001-of-00004.gguf").write_bytes(b"GGUF")
PY

before=$(find "$MODEL" -type f -print | sort)
FAKE_ARGS="$TMP/args" \
LLAMA_CPP_DIR="$LLAMA" \
LLAMA_CPP_REV="test-revision" \
LLAMA_CPP_SKIP_REV_CHECK=1 \
PYTHON=python3 \
    "$PREPARE" "$MODEL" "$OUTPUT"
after=$(find "$MODEL" -type f -print | sort)

test "$before" = "$after"
test -f "$OUTPUT/model-mxfp4_moe-00001-of-00004.gguf"
test "$(cat "$OUTPUT/llama.cpp-revision.txt")" = "test-revision"
grep -Fx -- "$MODEL" "$TMP/args"
grep -Fx -- "--outtype" "$TMP/args"
grep -Fx -- "auto" "$TMP/args"
grep -Fx -- "--split-max-size" "$TMP/args"
grep -Fx -- "48G" "$TMP/args"

mkdir -p "$MODEL/nested-output"
if LLAMA_CPP_DIR="$LLAMA" LLAMA_CPP_SKIP_REV_CHECK=1 \
    "$PREPARE" "$MODEL" "$MODEL/nested-output" >"$TMP/stdout" 2>"$TMP/stderr"; then
    echo "expected nested output path to fail" >&2
    exit 1
fi
grep -F "output directory must not be inside the checkpoint" "$TMP/stderr"

BAD="$TMP/not-deepseek"
mkdir -p "$BAD"
printf '%s\n' '{"model_type":"other"}' > "$BAD/config.json"
if LLAMA_CPP_DIR="$LLAMA" LLAMA_CPP_SKIP_REV_CHECK=1 \
    "$PREPARE" "$BAD" "$TMP/bad-output" >"$TMP/stdout" 2>"$TMP/stderr"; then
    echo "expected non-DeepSeek checkpoint to fail" >&2
    exit 1
fi
grep -F "model_type deepseek_v4" "$TMP/stderr"

echo "DeepSeek V4 GGUF preparation contract: ok"
