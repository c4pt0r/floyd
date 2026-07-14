# OpenAI-Compatible Server Design

## Goal

Add a native HTTP server to `floyd serve` for OpenAI-compatible text chat. The
server must keep the selected Metal model resident, support DeepSeek V4 DS4 and
Moonlight, and preserve the existing `serve --stdio` JSONL interface.

The first release covers `GET /v1/models` and `POST /v1/chat/completions`,
including Server-Sent Events streaming. Tool calling, multimodal messages,
embeddings, the Responses API, structured output, log probabilities, and
multiple choices are outside this milestone and must return explicit errors
when requested through Chat Completions.

## CLI And Process Model

HTTP mode starts with:

```sh
./floyd serve --model /path/to/model --host 127.0.0.1 --port 8080 \
  --api-key secret --served-model-name deepseek-v4 --prefix-cache-mb 256
```

`--host` defaults to `127.0.0.1`, `--port` to `8080`, and
`--served-model-name` to the model directory basename. `--api-key` is optional;
when present, every `/v1/*` request must provide the same value as an
`Authorization: Bearer` token. HTTP mode and `--stdio` are mutually exclusive.

The server owns one resident model and executes one generation at a time. This
matches the current mutable KV/session contract and avoids unsafe concurrent
access. DeepSeek V4 reuses the byte-budgeted prefix snapshot LRU. Moonlight
resets its session for each independent request and reports zero cached tokens.

## Components

`openai_http.{c,h}` owns sockets, bounded HTTP/1.1 parsing, routing,
authentication, OpenAI request/response encoding, and SSE. It depends on a
model-neutral generation callback and does not include Metal or model headers.

The existing DeepSeek serve bridge is generalized around a token callback so
both JSONL buffering and HTTP streaming use the same request execution path. A
Moonlight adapter exposes equivalent request-scoped generation while retaining
its current chat and run behavior.

HTTP accepts requests with `Content-Length`, limits headers to 32 KiB and bodies
to 8 MiB, and closes the connection after each response. Unsupported transfer
encodings and malformed requests receive JSON errors. Client disconnect stops
the active token callback. `SIGINT` and `SIGTERM` stop accepting new work and
release the model after the active request finishes.

## OpenAI Contract

`GET /v1/models` returns an OpenAI `list` containing the single loaded model.
The request `model` field on Chat Completions must equal that advertised ID.

Chat Completions accepts string-content `system`, `user`, and `assistant`
messages plus `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`,
`stream`, and `stream_options.include_usage`. `n` is accepted only when equal
to one. Unsupported fields that change generation semantics are rejected
rather than ignored.

Non-streaming responses use `chat.completion`, one assistant message,
`finish_reason`, and standard prompt/completion/total usage. Streaming responses
emit a role chunk, content delta chunks, a terminal choice chunk, an optional
usage chunk, and `data: [DONE]`. DS4 cache usage is exposed through
`usage.prompt_tokens_details.cached_tokens`. Timing and cache diagnostics may
also appear under an ignorable `floyd` extension object.

Errors use the OpenAI shape:

```json
{"error":{"message":"...","type":"invalid_request_error","param":"...","code":"..."}}
```

Unknown routes return 404, invalid authentication returns 401, unsupported
methods return 405, oversized bodies return 413, and malformed or unsupported
request options return 400. Authorization values and prompt content are never
logged.

## Verification

Model-free tests cover HTTP parsing, limits, authentication, routing, OpenAI
request validation, JSON escaping, errors, SSE ordering, usage, and client
disconnects. A socket-level mock server smoke covers model listing and both
completion modes. CLI tests cover defaults, invalid values, transport
exclusivity, and help output.

An OpenAI Python SDK smoke verifies non-streaming and streaming calls. Opt-in
real-model tests run one-token requests against DeepSeek V4 and Moonlight; the
DeepSeek test sends a shared-prefix second request and requires a real cache
hit. Existing stdio, prefix-cache, C, Metal parity, and real-model regression
gates remain green.
