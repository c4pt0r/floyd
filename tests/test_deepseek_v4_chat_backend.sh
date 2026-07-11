#!/bin/sh
set -eu

model=$1
expected=$2
output=$(mktemp -t deepseek-v4-chat-backend.XXXXXX)
pid=
cleanup() {
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -f "$output"
}
trap cleanup EXIT INT TERM

PROMPT=hello NGEN=1 ./deepseek_v4_chat "$model" 64 1 >"$output" 2>&1 &
pid=$!
attempt=0
while [ "$attempt" -lt 50 ]; do
    if grep -q "DEEPSEEK_V4_BACKEND backend=$expected" "$output"; then
        cat "$output"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    attempt=$((attempt + 1))
    sleep 0.1
done
cat "$output"
echo "missing DEEPSEEK_V4_BACKEND backend=$expected" >&2
exit 1
