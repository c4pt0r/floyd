# OpenAI-Compatible Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native, resident-model OpenAI-compatible HTTP server for DeepSeek V4 DS4 and Moonlight while preserving `serve --stdio`.

**Architecture:** `openai_http.{c,h}` provides bounded HTTP/1.1, Bearer authentication, OpenAI JSON, and SSE over a model-neutral token callback. DeepSeek V4 and Moonlight retain their model-specific session ownership and implement the common generation callback; the CLI selects HTTP or stdio transport.

**Tech Stack:** C99, POSIX sockets, the repository's `json.h`, Server-Sent Events, Apple Metal runtimes, shell/C/Python tests, OpenAI Python SDK.

## Global Constraints

- Build remains macOS and Metal only; no CPU inference fallback or Python runtime process.
- Add no HTTP library or other production dependency.
- HTTP defaults to `127.0.0.1:8080`; `--stdio` and HTTP are mutually exclusive.
- `--api-key` is optional; when set, require the exact Bearer token for `/v1/*`.
- Serve one resident model and serialize all generation.
- Limit headers to 32 KiB and request bodies to 8 MiB.
- Preserve the current JSONL request and response schema for `serve --stdio`.
- DS4 keeps its byte-budgeted prefix LRU; Moonlight reports zero cached tokens.
- Do not commit models, fixtures, generated headers, binaries, logs, or `AGENTS.md`.

---

### Task 1: OpenAI Chat Protocol

**Files:**
- Create: `openai_http.h`
- Create: `openai_http.c`
- Create: `tests/test_openai_http_protocol.c`
- Modify: `Makefile`

**Interfaces:**
- Produces: `OpenAIChatRequest`, `OpenAIGenerationResult`, `OpenAITokenSink`, `OpenAIGenerateHandler`, `openai_chat_request_parse()`, `openai_chat_request_free()`, `openai_format_models_json()`, `openai_format_completion_json()`, and `openai_format_error_json()`.
- Consumes: `json_parse_full()` and `json_get()` from `json.h`.

- [ ] **Step 1: Write the failing protocol test**

Cover a valid request with `max_completion_tokens`, streaming usage, and three
string messages; verify defaults; verify exact model matching; and reject
`tools`, `tool_choice`, array content, `n=2`, `logprobs`, `response_format`, both
token-limit fields together, and invalid numeric ranges. Assert the formatted
completion contains standard choices and:

```c
CHECK(strstr(json, "\"object\":\"chat.completion\"") != NULL);
CHECK(strstr(json, "\"finish_reason\":\"length\"") != NULL);
CHECK(strstr(json, "\"prompt_tokens_details\":{\"cached_tokens\":7}") != NULL);
```

- [ ] **Step 2: Run the protocol test and verify RED**

Run: `make tests/test_openai_http_protocol`

Expected: build fails because `openai_http.h` and its functions do not exist.

- [ ] **Step 3: Define the model-neutral contract**

Add these public shapes to `openai_http.h`:

```c
typedef struct { char *role; char *content; } OpenAIMessage;
typedef struct {
    char *model;
    OpenAIMessage *messages;
    size_t message_count;
    int max_tokens;
    float temperature;
    float top_p;
    int stream;
    int include_usage;
} OpenAIChatRequest;

typedef struct {
    int prompt_tokens;
    int cached_tokens;
    int completion_tokens;
    double prompt_ms;
    double decode_ms;
    uint64_t cache_entries;
    uint64_t cache_bytes;
    const char *finish_reason;
} OpenAIGenerationResult;

typedef int (*OpenAITokenSink)(int token, const char *piece,
                               size_t piece_size, void *user_data);
typedef int (*OpenAIGenerateHandler)(
    void *user_data, const OpenAIChatRequest *request,
    OpenAITokenSink sink, void *sink_data,
    OpenAIGenerationResult *result, char *error, size_t error_size);
```

Expose parsers and allocated JSON formatters with `char **json` and
`size_t *json_size` output parameters. All allocated requests and JSON strings
are released by the caller.

- [ ] **Step 4: Implement strict parsing and JSON formatting**

Use `json_parse_full()` and structured field inspection. Treat missing
`temperature` as `0.0`, missing `top_p` as `1.0`, and missing token limit as
`-1`. Require `model` and a non-empty `messages` array. Encode strings with one
shared JSON escaping helper. Set standard `prompt_tokens`,
`completion_tokens`, `total_tokens`, and cached-token details.

- [ ] **Step 5: Run the protocol test and existing JSONL tests**

Run: `make tests/test_openai_http_protocol && ./tests/test_openai_http_protocol`

Expected: `OpenAI HTTP protocol tests: ok`.

Run: `make test-c`

Expected: every existing C test remains green.

- [ ] **Step 6: Commit the protocol**

```sh
git add openai_http.c openai_http.h tests/test_openai_http_protocol.c Makefile
git commit -m "feat: add OpenAI chat protocol"
```

---

### Task 2: Bounded HTTP/1.1 And SSE Transport

**Files:**
- Modify: `openai_http.h`
- Modify: `openai_http.c`
- Create: `tests/test_openai_http_socket.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `OpenAIGenerateHandler` from Task 1.
- Produces: `OpenAIHttpConfig`, `openai_http_handle_connection()`, and `openai_http_serve()`.

- [ ] **Step 1: Write failing socket-pair tests**

Use `socketpair(AF_UNIX, SOCK_STREAM, 0, pair)` and a fake handler that emits
`"Hel"` then `"lo"`. Test:

```c
static int fake_generate(void *ctx, const OpenAIChatRequest *request,
                         OpenAITokenSink sink, void *sink_data,
                         OpenAIGenerationResult *result,
                         char *error, size_t error_size) {
    CHECK(sink(1, "Hel", 3, sink_data));
    CHECK(sink(2, "lo", 2, sink_data));
    result->prompt_tokens = 5;
    result->cached_tokens = 2;
    result->completion_tokens = 2;
    result->finish_reason = "stop";
    return 1;
}
```

Assert `/v1/models`, non-streaming completion, SSE role/content/final/usage
ordering, `[DONE]`, correct `Content-Type`, valid and invalid Bearer tokens,
404/405, missing or conflicting `Content-Length`, unsupported transfer
encoding, 32 KiB header rejection, and 8 MiB body rejection.

- [ ] **Step 2: Run the socket test and verify RED**

Run: `make tests/test_openai_http_socket`

Expected: build fails for missing connection/server APIs.

- [ ] **Step 3: Add transport configuration and connection API**

```c
typedef struct {
    const char *host;
    int port;
    const char *api_key;
    const char *model_name;
} OpenAIHttpConfig;

int openai_http_handle_connection(
    int fd, const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data);
int openai_http_serve(
    const OpenAIHttpConfig *config,
    OpenAIGenerateHandler handler, void *user_data,
    char *error, size_t error_size);
```

- [ ] **Step 4: Implement bounded HTTP and streaming**

Read through `recv()` until `\r\n\r\n`, enforcing the header cap before
allocation. Parse request line and case-insensitive headers without mutating
body bytes. Read exactly `Content-Length`, cap before allocation, and close the
connection after one response. Ignore Authorization only when no API key is
configured.

For streaming, send HTTP headers, then the assistant-role chunk before calling
the model handler. The token sink JSON-escapes each content delta and writes one
`data: ...\n\n` event. Send the terminal choice, optional usage-only chunk, and
`data: [DONE]\n\n`. A failed `send()` makes the sink return zero so generation
stops.

Bind with `getaddrinfo()`, `SO_REUSEADDR`, `listen()`, and a sequential
`accept()` loop. Install `SIGINT`/`SIGTERM` handlers that set a stop flag and
close the listening socket; never log request bodies or Authorization.

- [ ] **Step 5: Run protocol, socket, and sanitizing checks**

Run: `make tests/test_openai_http_protocol tests/test_openai_http_socket`

Run: `./tests/test_openai_http_protocol && ./tests/test_openai_http_socket`

Expected: both tests print `ok`.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 6: Commit the transport**

```sh
git add openai_http.c openai_http.h tests/test_openai_http_socket.c Makefile
git commit -m "feat: add native OpenAI HTTP transport"
```

---

### Task 3: DeepSeek V4 HTTP Adapter And CLI

**Files:**
- Modify: `deepseek_v4_chat.c`
- Modify: `deepseek_v4_chat.h`
- Modify: `floyd.c`
- Modify: `Makefile`
- Modify: `tests/test_deepseek_v4_serve_cli.sh`
- Create: `tests/test_deepseek_v4_openai_official.sh`

**Interfaces:**
- Consumes: `OpenAIHttpConfig`, `OpenAIGenerateHandler`, and DS4 `deepseek_v4_ds4_generate_messages()`.
- Produces: HTTP fields on `DeepSeekV4ServeOptions` and a DS4 OpenAI generation adapter.

- [ ] **Step 1: Extend CLI tests first**

Require help to show `--host`, `--port`, `--api-key`, and
`--served-model-name`. Assert `serve --model DIR` selects HTTP defaults,
`--stdio --host` fails as a transport conflict, ports `0` and `65536` fail,
HTTP-only flags fail for `chat/run`, and `--stdio` still dispatches the existing
JSONL path unchanged.

- [ ] **Step 2: Run the CLI test and verify RED**

Run: `make floyd && sh tests/test_deepseek_v4_serve_cli.sh`

Expected: failure because `serve` still requires `--stdio` and the HTTP flags
are unknown.

- [ ] **Step 3: Add explicit transport options**

Extend `DeepSeekV4ServeOptions` with:

```c
int stdio;
const char *host;
int port;
const char *api_key;
const char *served_model_name;
```

Add the same values to `CliOptions`. Default to HTTP on
`127.0.0.1:8080`; retain `--stdio` as an explicit alternate transport. Compute
the default served name from the last non-empty component of `--model` without
modifying the input path.

- [ ] **Step 4: Factor DS4 request generation and add HTTP dispatch**

Extract one helper in `deepseek_v4_chat.c` that accepts roles/content,
per-request sampling, an `OpenAITokenSink`, and `OpenAIGenerationResult`. The
existing JSONL callback uses a buffer sink; the HTTP callback sends the sink
directly into `deepseek_v4_ds4_generate_messages()`. Default OpenAI
`max_tokens=-1` to the server's `--ngen`; use the server's `--draft` because
draft is not an OpenAI request field.

Log only:

```text
DEEPSEEK_V4_SERVE transport=http backend=metal-ds4 listen=127.0.0.1:8080 model=deepseek-v4 prefix_cache_mb=256 auth=off
```

Never print the API key.

- [ ] **Step 5: Add and run the real DS4 HTTP smoke**

The shell test starts `floyd serve` in the background on an unused localhost
port, waits for `/v1/models`, sends two one-token requests with a shared message
prefix, and checks OpenAI shape plus positive
`prompt_tokens_details.cached_tokens` on the second request. It terminates and
waits for the server in a trap.

Run:

```sh
DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark \
  sh tests/test_deepseek_v4_openai_official.sh
```

Expected: both completions succeed and the second reports a real DS4 cache hit.

Run: `make test-deepseek-v4-serve-official`

Expected: existing stdio cache smoke remains green.

- [ ] **Step 6: Commit the DS4 HTTP path**

```sh
git add deepseek_v4_chat.c deepseek_v4_chat.h floyd.c Makefile \
  tests/test_deepseek_v4_serve_cli.sh tests/test_deepseek_v4_openai_official.sh
git commit -m "feat: serve DeepSeek V4 over OpenAI HTTP"
```

---

### Task 4: Moonlight HTTP Adapter

**Files:**
- Modify: `moonlight_chat.c`
- Modify: `moonlight_chat.h`
- Modify: `floyd.c`
- Modify: `Makefile`
- Create: `tests/test_moonlight_openai_official.sh`

**Interfaces:**
- Consumes: `OpenAIGenerateHandler` and the existing resident `MoonlightModel`/`MoonlightSession`.
- Produces: `MoonlightServeOptions` and `moonlight_serve_run()`.

- [ ] **Step 1: Write the Moonlight real HTTP smoke first**

Start `floyd serve` with the q8 Moonlight model, request `/v1/models`, send one
non-streaming one-token completion, and send one streaming completion. Require
`backend=metal-moonlight`, exact advertised model ID, standard usage, SSE
`[DONE]`, and cached tokens equal to zero.

- [ ] **Step 2: Run the Moonlight smoke and verify RED**

Run:

```sh
MOONLIGHT_MODEL=/Users/dongxu/floyd/models/moonlight_i8 \
  sh tests/test_moonlight_openai_official.sh
```

Expected: `floyd` exits with the current message that serve requires DeepSeek
V4.

- [ ] **Step 3: Expose Moonlight serve options**

```c
typedef struct {
    const char *model_dir;
    int max_context;
    int max_new_tokens;
    const char *host;
    int port;
    const char *api_key;
    const char *served_model_name;
} MoonlightServeOptions;

int moonlight_serve_run(const MoonlightServeOptions *options);
```

- [ ] **Step 4: Refactor generation to a token sink**

Change `choose_token()` to accept request temperature/top-p. Change the internal
generation loop to invoke a callback instead of writing directly to stdout;
the existing chat/run path supplies a stdout callback and retains its output.

For HTTP, reset `MoonlightSession` and history at the start of every request,
render the supplied message list with `mtok_tmpl_msg()`, append
`mtok_tmpl_genprompt()`, and generate with request sampling. Fill standard
stats from `moonlight_session_stats()` deltas, set cached tokens and cache bytes
to zero, and set finish reason to `length` only when the request token limit is
reached without EOS.

- [ ] **Step 5: Run Moonlight HTTP and chat regressions**

Run the real smoke from Step 2; expected: model list, JSON completion, and SSE
completion pass with cache count zero.

Run:

```sh
sh tests/test_cli_default_chat.sh /Users/dongxu/floyd/models/moonlight_i8
```

Expected: existing interactive Metal chat test remains green.

Run: `make test-moonlight-metal MOONLIGHT_TINY=/Users/dongxu/floyd/fixture_tiny MOONLIGHT_ORACLE="$PWD/fixture_moonlight_metal"`

Expected: tiny Moonlight final logits and routes remain within their existing
thresholds.

- [ ] **Step 6: Commit the Moonlight HTTP path**

```sh
git add moonlight_chat.c moonlight_chat.h floyd.c Makefile \
  tests/test_moonlight_openai_official.sh
git commit -m "feat: serve Moonlight over OpenAI HTTP"
```

---

### Task 5: OpenAI SDK Smoke And User Quickstart

**Files:**
- Create: `tests/test_openai_sdk.py`
- Create: `tests/test_openai_sdk.sh`
- Modify: `README.md`
- Modify: `Makefile`

**Interfaces:**
- Consumes: a running server selected by `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- Produces: `make test-openai-sdk` and user-facing curl/Python examples.

- [ ] **Step 1: Write the SDK smoke**

Use the installed SDK without starting another runtime:

```python
from openai import OpenAI

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                api_key=os.environ.get("OPENAI_API_KEY", "unused"))
model = os.environ["OPENAI_MODEL"]
models = client.models.list()
assert [item.id for item in models.data] == [model]
reply = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Reply OK"}],
    max_tokens=2, temperature=0)
assert reply.choices[0].message.content is not None
chunks = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Reply OK"}],
    max_tokens=2, temperature=0, stream=True,
    stream_options={"include_usage": True})
assert any(chunk.usage is not None for chunk in chunks)
```

- [ ] **Step 2: Run against the DS4 server and verify any compatibility RED**

Start the real DS4 HTTP server with an API key, then run:

```sh
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=test-secret OPENAI_MODEL=deepseek-v4 \
  .venv/bin/python tests/test_openai_sdk.py
```

Expected before final compatibility fixes: any schema mismatch is reported by
the SDK as a parsing or API-status failure, not hidden by curl.

- [ ] **Step 3: Fix only observed SDK contract mismatches**

Keep the approved endpoint scope. Add missing standard fields required by the
SDK (`id`, `created`, `model`, `object`, `choices`, `usage`) to the existing
formatters; do not add Responses API or tool-call behavior.

- [ ] **Step 4: Update the README quickstart**

Place an HTTP quickstart directly after chat/run. Show unauthenticated local
serve, authenticated serve, curl, Python SDK, streaming, `--stdio`, DS4 cache
behavior, Moonlight's zero cross-request cache, memory guidance, and the
security warning that `0.0.0.0` should be paired with `--api-key`. Keep
performance discussion to existing benchmark instructions.

- [ ] **Step 5: Run the SDK and README-facing command checks**

Run the Python smoke against DS4 and Moonlight. Run `./floyd help` and verify
every documented flag appears. Run `make test-openai-sdk` with the three
required environment variables.

- [ ] **Step 6: Commit SDK compatibility and docs**

```sh
git add tests/test_openai_sdk.py tests/test_openai_sdk.sh README.md Makefile
git commit -m "docs: add OpenAI server quickstart"
```

---

### Task 6: Full Regression And Delivery Audit

**Files:**
- Modify only files implicated by a failing test.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified OpenAI compatibility evidence for both resident Metal runtimes.

- [ ] **Step 1: Run model-free and Metal regression gates**

```sh
make
make test-c
make metal-test
sh tests/test_metal_only_build.sh
sh tests/test_deepseek_v4_serve_cli.sh
```

Expected: all commands exit zero; Metal-only contract still finds no CPU
inference or fallback symbols.

- [ ] **Step 2: Run real HTTP and stdio gates**

```sh
DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark \
  sh tests/test_deepseek_v4_openai_official.sh
DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark \
  make test-deepseek-v4-serve-official
MOONLIGHT_MODEL=/Users/dongxu/floyd/models/moonlight_i8 \
  sh tests/test_moonlight_openai_official.sh
```

Expected: DS4 JSON/SSE and real prefix hit pass, JSONL remains unchanged, and
Moonlight JSON/SSE reports zero cached tokens.

- [ ] **Step 3: Run SDK gates for both models**

Use each real server's advertised model and optional key with
`tests/test_openai_sdk.py`. Require a model listing, one non-streaming response,
one streaming response, and a final usage chunk for each runtime.

- [ ] **Step 4: Audit repository and product binary**

```sh
git diff --check
git status --short --untracked-files=all
git ls-files AGENTS.md models kernels_metal.h '*.bin' '*.log'
nm floyd
otool -L floyd
```

Expected: only the pre-existing untracked `AGENTS.md` may appear; prohibited
tracked paths are absent; `nm` contains no CPU/fallback inference symbol; and
the binary links Metal, MetalKit, and Foundation.

- [ ] **Step 5: Commit any narrowly required regression fix**

If Step 1-4 required a source or test correction, stage only those reviewed
files and commit with a subject naming the corrected behavior. If no correction
was needed, create no empty commit.
