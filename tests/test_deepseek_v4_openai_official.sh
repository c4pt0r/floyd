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
stream=$(mktemp -t floyd-deepseek-v4-openai-stream.XXXXXX)
sampled=$(mktemp -t floyd-deepseek-v4-openai-sampled.XXXXXX)
sampled_stream=$(mktemp -t floyd-deepseek-v4-openai-sampled-stream.XXXXXX)
diagnostics=$(mktemp -t floyd-deepseek-v4-openai-stderr.XXXXXX)
server_pid=

cleanup() {
    if [ -n "$server_pid" ]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$models" "$first" "$second" "$stream" "$sampled" \
        "$sampled_stream" "$diagnostics"
}
trap cleanup EXIT INT TERM

./floyd serve --model "$DSPARK" --host 127.0.0.1 --port "$port" \
    --served-model-name deepseek-v4 --ctx 512 --ngen 1 --draft 3 \
    --prefix-cache-mb 256 >/dev/null 2>"$diagnostics" &
server_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 600 ]; do
    if models_status=$(curl --silent --show-error --fail \
        --output "$models" --write-out '%{http_code}' \
        "http://127.0.0.1:$port/v1/models" 2>/dev/null); then
        if [ "$models_status" = 200 ]; then
            ready=1
            break
        fi
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

first_status=$(curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say alpha."}]}' \
    --output "$first" --write-out '%{http_code}' \
    "http://127.0.0.1:$port/v1/chat/completions")
second_status=$(curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say beta."}]}' \
    --output "$second" --write-out '%{http_code}' \
    "http://127.0.0.1:$port/v1/chat/completions")
stream_status=$(curl --silent --show-error --fail --no-buffer \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say gamma."}],"max_tokens":1,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
    --output "$stream" --write-out '%{http_code}' \
    "http://127.0.0.1:$port/v1/chat/completions")
sampled_status=$(curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say delta."}],"max_tokens":1,"temperature":0.7,"top_p":0.9}' \
    --output "$sampled" --write-out '%{http_code}' \
    "http://127.0.0.1:$port/v1/chat/completions")
sampled_stream_status=$(curl --silent --show-error --fail --no-buffer \
    -H 'Content-Type: application/json' \
    --data '{"model":"deepseek-v4","messages":[{"role":"system","content":"Answer with one short word."},{"role":"user","content":"Say epsilon."}],"max_tokens":1,"temperature":0.7,"top_p":0.9,"stream":true,"stream_options":{"include_usage":true}}' \
    --output "$sampled_stream" --write-out '%{http_code}' \
    "http://127.0.0.1:$port/v1/chat/completions")
[ "$first_status" = 200 ]
[ "$second_status" = 200 ]
[ "$stream_status" = 200 ]
[ "$sampled_status" = 200 ]
[ "$sampled_stream_status" = 200 ]

python3 - "$models" "$first" "$second" "$stream" "$sampled" \
    "$sampled_stream" <<'PY'
import json
import sys

models, first, second = [
    json.load(open(path, encoding="utf-8")) for path in sys.argv[1:4]
]
sampled = json.load(open(sys.argv[5], encoding="utf-8"))


def assert_usage(usage):
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + 1
    assert usage["prompt_tokens_details"]["cached_tokens"] >= 0


def assert_completion(response):
    assert response["object"] == "chat.completion"
    assert response["model"] == "deepseek-v4"
    assert len(response["choices"]) == 1
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(response["choices"][0]["message"]["content"], str)
    assert response["choices"][0]["finish_reason"] == "length"
    assert_usage(response["usage"])
    assert response["floyd"]["prompt_ms"] >= 0
    assert response["floyd"]["decode_ms"] > 0


def assert_stream(path):
    events = [
        line.removeprefix("data: ")
        for line in open(path, encoding="utf-8").read().splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(
        chunk["choices"] and "content" in chunk["choices"][0]["delta"]
        for chunk in chunks
    )
    terminal = chunks[-2]
    assert terminal["choices"][0]["finish_reason"] == "length"
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == []
    assert_usage(usage_chunk["usage"])
    assert usage_chunk["floyd"]["prompt_ms"] >= 0
    assert usage_chunk["floyd"]["decode_ms"] > 0
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "deepseek-v4"
    return usage_chunk


def print_result(name, usage, floyd):
    print(
        "DeepSeek V4 OpenAI %s: prompt=%d completion=%d total=%d cached=%d "
        "prompt_ms=%.3f decode_ms=%.3f"
        % (
            name,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            usage["prompt_tokens_details"]["cached_tokens"],
            floyd["prompt_ms"],
            floyd["decode_ms"],
        )
    )


assert models["object"] == "list"
assert [item["id"] for item in models["data"]] == ["deepseek-v4"]
for response in (first, second):
    assert_completion(response)
assert_completion(sampled)
assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
stream_usage = assert_stream(sys.argv[4])
sampled_stream_usage = assert_stream(sys.argv[6])
assert stream_usage["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
assert sampled_stream_usage["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
print_result("first", first["usage"], first["floyd"])
print_result("second", second["usage"], second["floyd"])
print_result("sse", stream_usage["usage"], stream_usage["floyd"])
print_result("sampled", sampled["usage"], sampled["floyd"])
print_result(
    "sampled_sse",
    sampled_stream_usage["usage"],
    sampled_stream_usage["floyd"],
)
print(
    "DeepSeek V4 OpenAI official: json_cached=%d "
    "sse_cached=%d sampled_json_tokens=%d sampled_sse_tokens=%d "
    "sampled_sse_cached=%d default_draft=3 effective_sample_draft=1 "
    "sampled_prompt_ms=%.3f sampled_decode_ms=%.3f"
    % (
        second["usage"]["prompt_tokens_details"]["cached_tokens"],
        stream_usage["usage"]["prompt_tokens_details"]["cached_tokens"],
        sampled["usage"]["completion_tokens"],
        sampled_stream_usage["usage"]["completion_tokens"],
        sampled_stream_usage["usage"]["prompt_tokens_details"]["cached_tokens"],
        sampled["floyd"]["prompt_ms"],
        sampled["floyd"]["decode_ms"],
    )
)
PY
printf 'DeepSeek V4 OpenAI HTTP statuses: models=%s first=%s second=%s sse=%s sampled=%s sampled_sse=%s\n' \
    "$models_status" "$first_status" "$second_status" "$stream_status" \
    "$sampled_status" "$sampled_stream_status"

grep -q \
    "DEEPSEEK_V4_SERVE transport=http backend=metal-ds4 listen=127.0.0.1:$port model=deepseek-v4 prefix_cache_mb=256 auth=off" \
    "$diagnostics"
