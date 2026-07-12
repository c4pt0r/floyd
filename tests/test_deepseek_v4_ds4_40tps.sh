#!/bin/sh
set -eu

model=$1
tokens=${2:-32}
out=$(mktemp /tmp/floyd-ds4-40tps-out.XXXXXX)
trace=$(mktemp /tmp/floyd-ds4-40tps-trace.XXXXXX)
greedy_out=$(mktemp /tmp/floyd-ds4-40tps-greedy-out.XXXXXX)
greedy_trace=$(mktemp /tmp/floyd-ds4-40tps-greedy-trace.XXXXXX)
spec_ids=$(mktemp /tmp/floyd-ds4-40tps-spec-ids.XXXXXX)
greedy_ids=$(mktemp /tmp/floyd-ds4-40tps-greedy-ids.XXXXXX)
lock=$(mktemp /tmp/floyd-ds4-40tps-lock.XXXXXX)
trap 'rm -f "$out" "$trace" "$greedy_out" "$greedy_trace" "$spec_ids" "$greedy_ids" "$lock"' EXIT

env SNAP="$model" CHAT=1 \
    PROMPT='请写一个至少五百字的中文科幻故事，直接开始正文。' \
    CTX=256 NGEN="$tokens" DRAFT=3 DEEPSEEK_V4_CHAT_TRACE=1 \
    DS4_LOCK_FILE="$lock" ./floyd >"$out" 2>"$trace"

env SNAP="$model" CHAT=1 DSPARK_SPEC=0 \
    PROMPT='请写一个至少五百字的中文科幻故事，直接开始正文。' \
    CTX=256 NGEN="$tokens" DEEPSEEK_V4_CHAT_TRACE=1 \
    DS4_LOCK_FILE="$lock" ./floyd >"$greedy_out" 2>"$greedy_trace"

perf=$(grep '^DEEPSEEK_V4_PERF ' "$trace")
count=$(grep -c '^DEEPSEEK_V4_TOKEN ' "$trace")
tps=$(printf '%s\n' "$perf" | sed -n 's/.* decode_tps=\([0-9.]*\).*/\1/p')
replay=$(printf '%s\n' "$perf" | sed -n 's/.* replay_ms=\([0-9.]*\).*/\1/p')
direct=$(printf '%s\n' "$perf" | sed -n 's/.* direct_accepted=\([0-9][0-9]*\).*/\1/p')
prefix1=$(printf '%s\n' "$perf" | sed -n 's/.* prefix1_accepted=\([0-9][0-9]*\).*/\1/p')
printf '%s\n' "$perf"
printf 'tokens=%s decode_tps=%s replay_ms=%s direct_accepted=%s prefix1_accepted=%s\n' \
    "$count" "$tps" "$replay" "$direct" "$prefix1"

test "$count" -eq "$tokens"
test -n "$direct"
test "$direct" -gt 0
test -n "$prefix1"
test "$prefix1" -gt 0
test "$replay" = "0.000"
test "$(grep -c '^DEEPSEEK_V4_TOKEN ' "$greedy_trace")" -eq "$tokens"
grep '^DEEPSEEK_V4_TOKEN ' "$trace" | sed 's/.* /token /' >"$spec_ids"
grep '^DEEPSEEK_V4_TOKEN ' "$greedy_trace" | sed 's/.* /token /' >"$greedy_ids"
diff -u "$greedy_ids" "$spec_ids"
awk -v value="$tps" 'BEGIN { exit !(value >= 40.0) }'
