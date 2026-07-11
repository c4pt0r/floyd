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
    rm -f "$output" "$output.missing"
}
trap cleanup EXIT INT TERM

./floyd --model "$model" --ctx 64 --ngen 1 >"$output" 2>&1 &
pid=$!
attempt=0
while [ "$attempt" -lt 50 ]; do
    if grep -q "DEEPSEEK_V4_BACKEND backend=$expected" "$output"; then
        if pgrep -P "$pid" -f '[Pp]ython' >/dev/null 2>&1; then
            cat "$output"
            echo "DeepSeek V4 chat spawned a Python child process" >&2
            exit 1
        fi
        cat "$output"
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if ! grep -q "DEEPSEEK_V4_BACKEND backend=$expected" "$output"; then
    cat "$output"
    echo "missing DEEPSEEK_V4_BACKEND backend=$expected" >&2
    exit 1
fi
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
pid=

missing="$model/missing.gguf"
set +e
FLOYD_DEEPSEEK_V4_GGUF="$missing" ./floyd --model "$model" --ctx 64 --ngen 1 \
    </dev/null >"$output.missing" 2>&1
status=$?
set -e
cat "$output.missing"
test "$status" -eq 2
grep -q 'FLOYD_DEEPSEEK_V4_GGUF' "$output.missing"

echo "DeepSeek V4 chat resident backend contract: ok"
