#!/bin/sh
set -eu

model=$1
min_tps=${MIN_TPS:-30}
tokens=${NGEN:-16}
output=$(mktemp -t floyd-deepseek-v4-ds4-official.XXXXXX)
trap 'rm -f "$output"' EXIT INT TERM

PROMPT=${PROMPT_TEXT:-hello} ./floyd --model "$model" --ctx 512 --ngen "$tokens" \
    >"$output" 2>&1
cat "$output"

grep -q 'DEEPSEEK_V4_BACKEND backend=metal-ds4' "$output"
perf=$(grep 'DEEPSEEK_V4_PERF ' "$output" | tail -1)
generated=$(printf '%s\n' "$perf" | sed -n 's/.*generated_tokens=\([0-9][0-9]*\).*/\1/p')
decode_tps=$(printf '%s\n' "$perf" | sed -n 's/.*decode_tps=\([0-9.][0-9.]*\).*/\1/p')

test -n "$generated"
test -n "$decode_tps"
test "$generated" -eq "$tokens"
awk -v actual="$decode_tps" -v minimum="$min_tps" \
    'BEGIN { exit !(actual >= minimum) }'
printf 'DeepSeek V4 DS4 official: generated=%s decode_tps=%s minimum=%s\n' \
    "$generated" "$decode_tps" "$min_tps"
