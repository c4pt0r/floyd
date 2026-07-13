#!/bin/sh
set -eu

model=$1
tokens=${NGEN:-32}
tmp=$(mktemp -d /tmp/floyd-ds4-draft2-tail.XXXXXX)
trap 'rm -rf "$tmp"' EXIT INT TERM

prompt='请写一个至少五百字的中文科幻故事，直接开始正文。'

run_chat() {
    mode=$1
    draft=$2
    shift
    shift
    set +e
    env DS4_LOCK_FILE="$tmp/$mode.lock" "$@" \
        ./floyd run --model "$model" --prompt "$prompt" --ctx 256 \
        --ngen "$tokens" --draft "$draft" --trace \
        >"$tmp/$mode.out" 2>"$tmp/$mode.trace"
    rc=$?
    set -e
    if test "$rc" -ne 0; then
        cat "$tmp/$mode.trace" "$tmp/$mode.out"
        return "$rc"
    fi
}

run_chat spec 2 FLOYD_DEEPSEEK_V4_DS4_CONFIDENCE_THRESHOLD=0
run_chat greedy 1

sed -n 's/^DEEPSEEK_V4_TOKEN /token /p' "$tmp/spec.trace" >"$tmp/spec.ids"
sed -n 's/^DEEPSEEK_V4_TOKEN /token /p' "$tmp/greedy.trace" >"$tmp/greedy.ids"
test "$(wc -l <"$tmp/spec.ids" | tr -d ' ')" -eq "$tokens"
test "$(wc -l <"$tmp/greedy.ids" | tr -d ' ')" -eq "$tokens"
diff -u "$tmp/greedy.ids" "$tmp/spec.ids"

perf=$(grep '^DEEPSEEK_V4_PERF ' "$tmp/spec.trace")
replay=$(printf '%s\n' "$perf" | sed -n 's/.* replay_ms=\([0-9.]*\).*/\1/p')
test "$replay" = "0.000"
printf '%s\n' "$perf"
printf 'DeepSeek V4 DS4 DRAFT2 tail parity: %s/%s replay_ms=%s\n' \
    "$tokens" "$tokens" "$replay"
