#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PREPARE="$ROOT/tools/prepare_deepseek_v4_ds4.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/floyd-dsv4-ds4-prepare.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

MODEL="$TMP/model"
OUTPUT="$TMP/output"
DS4="$TMP/ds4"
mkdir -p "$MODEL" "$DS4"
printf '%s\n' '{"model_type":"deepseek_v4"}' > "$MODEL/config.json"

cat > "$DS4/download_model.sh" <<'SH'
#!/bin/sh
set -eu
printf '%s\n' "$1" > "$FAKE_TARGET"
mkdir -p "$DS4_GGUF_DIR"
touch "$DS4_GGUF_DIR/DeepSeek-V4-Flash-test-imatrix.gguf"
SH
chmod +x "$DS4/download_model.sh"

FAKE_TARGET="$TMP/target" \
DS4_DIR="$DS4" \
DS4_REV="test-ds4-revision" \
DS4_SKIP_REV_CHECK=1 \
    "$PREPARE" "$MODEL" "$OUTPUT"

test "$(cat "$TMP/target")" = "q2-imatrix"
test "$(cat "$OUTPUT/ds4-revision.txt")" = "test-ds4-revision"
test -f "$OUTPUT/DeepSeek-V4-Flash-test-imatrix.gguf"

echo "DeepSeek V4 DS4 preparation contract: ok"
