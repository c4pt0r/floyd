#!/bin/sh
set -eu

: "${MOONLIGHT_MODEL:?set MOONLIGHT_MODEL=/path/to/moonlight_i8}"

port=$(python3 - <<'PY'
import socket

with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
)
models=$(mktemp -t floyd-moonlight-openai-models.XXXXXX)
completion=$(mktemp -t floyd-moonlight-openai-completion.XXXXXX)
stream=$(mktemp -t floyd-moonlight-openai-stream.XXXXXX)
diagnostics=$(mktemp -t floyd-moonlight-openai-stderr.XXXXXX)
server_pid=

cleanup() {
    if [ -n "$server_pid" ]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$models" "$completion" "$stream" "$diagnostics"
}
trap cleanup EXIT INT TERM

./floyd serve --model "$MOONLIGHT_MODEL" --host 127.0.0.1 --port "$port" \
    --served-model-name moonlight-test --ctx 512 --ngen 1 \
    >/dev/null 2>"$diagnostics" &
server_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 1200 ]; do
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
    echo "timed out waiting for Moonlight OpenAI server" >&2
    exit 1
}

curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    --data '{"model":"moonlight-test","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Say alpha."}],"max_tokens":1,"temperature":0}' \
    "http://127.0.0.1:$port/v1/chat/completions" >"$completion"
curl --silent --show-error --fail --no-buffer \
    -H 'Content-Type: application/json' \
    --data '{"model":"moonlight-test","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Say beta."}],"max_tokens":1,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
    "http://127.0.0.1:$port/v1/chat/completions" >"$stream"

python3 - "$models" "$completion" "$stream" <<'PY'
import json
import sys

models = json.load(open(sys.argv[1], encoding="utf-8"))
completion = json.load(open(sys.argv[2], encoding="utf-8"))
stream_lines = open(sys.argv[3], encoding="utf-8").read().splitlines()

assert models["object"] == "list"
assert [item["id"] for item in models["data"]] == ["moonlight-test"]

assert completion["object"] == "chat.completion"
assert completion["model"] == "moonlight-test"
assert len(completion["choices"]) == 1
assert completion["choices"][0]["message"]["role"] == "assistant"
assert isinstance(completion["choices"][0]["message"]["content"], str)
assert completion["choices"][0]["finish_reason"] == "length"
usage = completion["usage"]
assert usage["prompt_tokens"] > 0
assert usage["completion_tokens"] == 1
assert usage["total_tokens"] == usage["prompt_tokens"] + 1
assert usage["prompt_tokens_details"]["cached_tokens"] == 0
assert completion["floyd"]["cache_entries"] == 0
assert completion["floyd"]["cache_bytes"] == 0

events = [
    line.removeprefix("data: ")
    for line in stream_lines
    if line.startswith("data: ")
]
assert events[-1] == "[DONE]"
chunks = [json.loads(event) for event in events[:-1]]
assert chunks[0]["object"] == "chat.completion.chunk"
assert chunks[0]["model"] == "moonlight-test"
assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
assert any(chunk["choices"] and "content" in chunk["choices"][0]["delta"]
           for chunk in chunks)
terminal = next(
    chunk for chunk in chunks
    if chunk["choices"] and chunk["choices"][0]["finish_reason"]
)
assert terminal["choices"][0]["finish_reason"] == "length"
usage_chunk = next(chunk for chunk in chunks if not chunk["choices"])
stream_usage = usage_chunk["usage"]
assert stream_usage["prompt_tokens"] > 0
assert stream_usage["completion_tokens"] == 1
assert stream_usage["total_tokens"] == stream_usage["prompt_tokens"] + 1
assert stream_usage["prompt_tokens_details"]["cached_tokens"] == 0
assert usage_chunk["floyd"]["cache_entries"] == 0
assert usage_chunk["floyd"]["cache_bytes"] == 0
print(
    "Moonlight OpenAI official: completion_tokens=%d cached_tokens=%d sse_done=yes"
    % (usage["completion_tokens"], usage["prompt_tokens_details"]["cached_tokens"])
)
PY

grep -q 'MOONLIGHT_BACKEND backend=metal-moonlight' "$diagnostics"
grep -Fqx \
    "MOONLIGHT_SERVE transport=http backend=metal-moonlight listen=127.0.0.1:$port model=moonlight-test auth=off" \
    "$diagnostics"
