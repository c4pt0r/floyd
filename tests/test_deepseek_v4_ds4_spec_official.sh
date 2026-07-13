#!/bin/sh
set -eu

model=$1
tokens=${NGEN:-32}
draft=${DRAFT:-2}
margin=${FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN:-3.0}
mtp=${FLOYD_DEEPSEEK_V4_DS4_MTP:-"$model-DS4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"}
tmp=$(mktemp -d -t floyd-deepseek-v4-ds4-spec.XXXXXX)
trap 'rm -rf "$tmp"' EXIT INT TERM

if test ! -f "$mtp" || test -e "$mtp.part" || test -e "$mtp.aria2"; then
    echo "complete DS4 MTP GGUF not found: $mtp" >&2
    exit 2
fi

prompt=${PROMPT_TEXT:-写一个至少100字的中文故事。}
run_chat() {
    prefix=$1
    draft_flag=$2
    shift
    shift
    set +e
    env "$@" ./floyd run --model "$model" --prompt "$prompt" \
        --ctx 512 --ngen "$tokens" --draft "$draft_flag" --trace \
        >"$prefix.stdout" 2>"$prefix.stderr"
    rc=$?
    set -e
    if test "$rc" -ne 0; then
        cat "$prefix.stderr" "$prefix.stdout"
        return "$rc"
    fi
}

run_chat "$tmp/base" 1 -u DSPARK_SPEC -u DRAFT \
    -u FLOYD_DEEPSEEK_V4_DS4_MTP -u FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN
run_chat "$tmp/spec" "$draft" DSPARK_SPEC=1 \
    FLOYD_DEEPSEEK_V4_DS4_MTP="$mtp" \
    FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN="$margin"

sed -n 's/^DEEPSEEK_V4_TOKEN //p' "$tmp/base.stderr" >"$tmp/base.tokens"
sed -n 's/^DEEPSEEK_V4_TOKEN //p' "$tmp/spec.stderr" >"$tmp/spec.tokens"
test "$(wc -l <"$tmp/base.tokens" | tr -d ' ')" -eq "$tokens"
test "$(wc -l <"$tmp/spec.tokens" | tr -d ' ')" -eq "$tokens"
if ! cmp "$tmp/base.tokens" "$tmp/spec.tokens"; then
    diff -u "$tmp/base.tokens" "$tmp/spec.tokens" || true
    exit 1
fi

base_perf=$(grep 'DEEPSEEK_V4_PERF ' "$tmp/base.stderr" | tail -1)
spec_perf=$(grep 'DEEPSEEK_V4_PERF ' "$tmp/spec.stderr" | tail -1)
spec_rounds=$(printf '%s\n' "$spec_perf" |
    sed -n 's/.*spec_rounds=\([0-9][0-9]*\).*/\1/p')
spec_tokens=$(printf '%s\n' "$spec_perf" |
    sed -n 's/.*spec_tokens=\([0-9][0-9]*\).*/\1/p')
test -n "$spec_rounds"
test -n "$spec_tokens"
test "$spec_rounds" -gt 0

printf '%s\n' "$base_perf"
printf '%s\n' "$spec_perf"
awk -v accepted="$spec_tokens" -v rounds="$spec_rounds" \
    'BEGIN { printf "DeepSeek V4 DS4 spec: accepted_per_round=%.6f\n", accepted / rounds }'
printf 'DeepSeek V4 DS4 spec token parity: %s/%s draft=%s margin=%s\n' \
    "$tokens" "$tokens" "$draft" "$margin"
