#!/bin/sh
set -eu

model=$1
tokens=${2:-32}
out=$(mktemp /tmp/floyd-ds4-40tps-out.XXXXXX)
trace=$(mktemp /tmp/floyd-ds4-40tps-trace.XXXXXX)
lock=$(mktemp /tmp/floyd-ds4-40tps-lock.XXXXXX)
trap 'rm -f "$out" "$trace" "$lock"' EXIT

env SNAP="$model" CHAT=1 \
    PROMPT='请写一个至少五百字的中文科幻故事，直接开始正文。' \
    CTX=256 NGEN="$tokens" DRAFT=3 DEEPSEEK_V4_CHAT_TRACE=1 \
    DS4_LOCK_FILE="$lock" ./floyd >"$out" 2>"$trace"

perf=$(grep '^DEEPSEEK_V4_PERF ' "$trace")
count=$(grep -c '^DEEPSEEK_V4_TOKEN ' "$trace")
tps=$(printf '%s\n' "$perf" | sed -n 's/.* decode_tps=\([0-9.]*\).*/\1/p')
replay=$(printf '%s\n' "$perf" | sed -n 's/.* replay_ms=\([0-9.]*\).*/\1/p')
printf '%s\n' "$perf"
printf 'tokens=%s decode_tps=%s replay_ms=%s\n' "$count" "$tps" "$replay"

test "$count" -eq "$tokens"
awk -v value="$replay" 'BEGIN { exit !(value == 0.0) }'
awk -v value="$tps" 'BEGIN { exit !(value >= 40.0) }'
