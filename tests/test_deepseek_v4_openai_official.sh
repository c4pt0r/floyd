#!/bin/sh
set -eu

: "${DSPARK:?set DSPARK=/path/to/DeepSeek-V4-Flash-DSpark}"

port=$(python3 - <<'PY'
import socket

with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
)
models=$(mktemp -t floyd-deepseek-v4-openai-models.XXXXXX)
first=$(mktemp -t floyd-deepseek-v4-openai-first.XXXXXX)
second=$(mktemp -t floyd-deepseek-v4-openai-second.XXXXXX)
diagnostics=$(mktemp -t floyd-deepseek-v4-openai-stderr.XXXXXX)
server_pid=

cleanup() {
    if [ -n "$server_pid" ]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$models" "$first" "$second" "$diagnostics"
}
trap cleanup EXIT INT TERM

./floyd serve --model "$DSPARK" --host 127.0.0.1 --port "$port" \
    --served-model-name deepseek-v4 --ctx 512 --ngen 1 --draft 3 \
    --prefix-cache-mb 256 >/dev/null 2>"$diagnostics" &
server_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 600 ]; do
    if curl --silent --show-error --fail \
        "http://127.0.0.1:$port/v1/models" >"$models" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        cat "$diagnostics" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
[ "$ready" -eq 1 ] || {
    cat "$diagnostics" >&2
    echo "timed out waiting for DeepSeek V4 OpenAI server" >&2
    exit 1
}

curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say alpha."}]}' \
    "http://127.0.0.1:$port/v1/chat/completions" >"$first"
curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say beta."}]}' \
    "http://127.0.0.1:$port/v1/chat/completions" >"$second"

python3 - "$models" "$first" "$second" <<'PY'
import json
import sys

models, first, second = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
assert models["object"] == "list"
assert [item["id"] for item in models["data"]] == ["deepseek-v4"]
for response in (first, second):
    assert response["object"] == "chat.completion"
    assert response["model"] == "deepseek-v4"
    assert len(response["choices"]) == 1
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["usage"]["completion_tokens"] == 1
assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
print(
    "DeepSeek V4 OpenAI official: cached_tokens=%d"
    % second["usage"]["prompt_tokens_details"]["cached_tokens"]
)
PY

grep -q \
    "DEEPSEEK_V4_SERVE transport=http backend=metal-ds4 listen=127.0.0.1:$port model=deepseek-v4 prefix_cache_mb=256 auth=off" \
    "$diagnostics"
