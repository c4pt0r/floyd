# Floyd Stdio Serve And Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent `floyd serve --stdio` JSONL interface with a memory-bounded DS4 prefix KV snapshot cache and CLI-first documentation.

**Architecture:** `deepseek_v4_serve.c` owns model-free JSONL parsing, validation, response encoding, and the sequential stdio loop. `deepseek_v4_prefix_cache.c` owns an exact-token, byte-budgeted LRU. The existing DS4 bridge renders full message prompts, restores the longest compatible snapshot, saves message-boundary snapshots, and generates the response.

**Tech Stack:** C99, existing `json.h`, DS4 public session snapshot APIs, shell smoke tests, Make.

## Global Constraints

- Preserve existing `chat`, `run`, `tf`, `gen`, and `help` behavior.
- No HTTP, concurrency, disk cache, or token streaming in this milestone.
- Default prefix cache budget is 256 MiB; `--prefix-cache-mb 0` disables it.
- Preserve 1 GiB of measurable available host memory and degrade allocation failures to misses.
- Do not commit models, generated fixtures, binaries, logs, `.deps`, or `AGENTS.md`.

---

### Task 1: Strict JSONL Protocol

**Files:**
- Modify: `json.h`
- Create: `deepseek_v4_serve.h`
- Create: `deepseek_v4_serve.c`
- Create: `tests/test_deepseek_v4_serve_protocol.c`
- Modify: `Makefile`

**Interfaces:**
- Produces `DeepSeekV4ServeRequest`, `deepseek_v4_serve_parse_request()`, `deepseek_v4_serve_request_free()`, `deepseek_v4_serve_write_response()`, and `deepseek_v4_serve_stdio()`.
- `deepseek_v4_serve_stdio()` accepts a callback so model-free tests can use a fake handler.

- [ ] Write tests that parse `prompt` and `messages` requests, preserve string/integer ids, validate numeric ranges, reject conflicting prompt forms, escape response text, and continue after one malformed line.

```c
CHECK(deepseek_v4_serve_parse_request(
    "{\"id\":\"a\",\"prompt\":\"hello\",\"max_tokens\":3}",
    &request, error, sizeof(error)));
CHECK(strcmp(request.id_json, "\"a\"") == 0);
CHECK(request.max_tokens == 3);
```

- [ ] Run `make tests/test_deepseek_v4_serve_protocol && ./tests/test_deepseek_v4_serve_protocol`; verify RED because the serve API does not exist.
- [ ] Extend `json.h` with full-input validation and recursive `json_free()`, then implement the smallest protocol parser/encoder and sequential callback loop.
- [ ] Re-run the protocol test and `make test-c`; verify GREEN.
- [ ] Commit with `git commit -m "feat: add DeepSeek V4 stdio protocol"`.

### Task 2: Byte-Budgeted Prefix LRU

**Files:**
- Create: `deepseek_v4_prefix_cache.h`
- Create: `deepseek_v4_prefix_cache.c`
- Create: `tests/test_deepseek_v4_prefix_cache.c`
- Modify: `Makefile`

**Interfaces:**
- Produces `deepseek_v4_prefix_cache_init/free/find_longest/prepare_insert/put_take/stats`.
- Keys contain a fingerprint string, a `uint64_t` configuration key, and copied token ids. Snapshot bytes transfer ownership on successful insertion.

- [ ] Write tests for longest exact prefix, fingerprint/config isolation, recency updates, byte-budget eviction, memory-reserve eviction, disabled cache, and oversize skip.

```c
CHECK(deepseek_v4_prefix_cache_prepare_insert(&cache, 80, UINT64_MAX));
CHECK(deepseek_v4_prefix_cache_put_take(
    &cache, "model-a", 7, prefix, 3, snapshot, 80));
CHECK(deepseek_v4_prefix_cache_find_longest(
    &cache, "model-a", 7, extended, 4, &hit));
CHECK(hit.prefix_tokens == 3);
```

- [ ] Run the new test and verify RED because the cache API does not exist.
- [ ] Implement a doubly linked LRU whose accounting includes entry, fingerprint, token array, and snapshot allocations. Skip entries larger than the hard budget; when available memory is known, evict until the insertion also leaves 1 GiB free.
- [ ] Run the cache test, protocol test, and `make test-c`; verify GREEN.
- [ ] Commit with `git commit -m "feat: add bounded prefix snapshot cache"`.

### Task 3: DS4 Request And Snapshot Integration

**Files:**
- Modify: `deepseek_v4_ds4.h`
- Modify: `deepseek_v4_ds4.c`
- Modify: `deepseek_v4_chat.c`
- Create: `patches/ds4/deepseek-v4-request-draft.patch` only if request-level draft needs a public DS4 setter
- Modify: `Makefile`
- Modify: `tests/test_deepseek_v4_ds4.c`
- Create: `tests/test_deepseek_v4_serve_stdio.c`

**Interfaces:**
- Adds `DeepSeekV4Message`, `DeepSeekV4Ds4RequestConfig`, cache fields in `DeepSeekV4Ds4Stats`, and a full-message request generator.
- Adds a cached-open variant while preserving the existing chat open/generate functions as wrappers.

- [ ] Extend bridge contract tests for default cache budget conversion, invalid sampling/speculation combinations, and stable configuration-key separation. Add a fake stdio handler that runs two shared-prefix requests through the real LRU without model weights.
- [ ] Run the bridge and stdio tests; verify RED for the missing request/cache APIs.
- [ ] Render tokens incrementally at message boundaries. On each request, restore the longest matching snapshot; on a miss, prefill and save the prefix before the final user message; save the full prompt before generation. Call `ds4_session_payload_bytes()` before eviction/allocation and treat snapshot failures as misses.
- [ ] Use greedy/speculative generation for `temperature == 0`; use `ds4_session_sample()` plus ordinary eval for sampling. Reject sampling with `draft > 1`. Keep request processing sequential so any request-level draft setter is restored after the request.
- [ ] Run `make METAL=1 test-deepseek-v4-ds4 test-deepseek-v4-ds4-q8-pair`, the new stdio test, and related V4 cache/attention tests.
- [ ] Commit bridge integration with `git commit -m "feat: reuse DS4 prefixes across requests"`.

### Task 4: CLI Flags And Stdio Entrypoint

**Files:**
- Modify: `floyd.c`
- Modify: `deepseek_v4_chat.h`
- Modify: `deepseek_v4_chat.c`
- Modify: `Makefile`
- Create: `tests/test_deepseek_v4_serve_cli.sh`
- Modify: `tests/test_deepseek_v4_chat_dispatch.sh`
- Modify: `tests/test_cli_default_chat.sh`

**Interfaces:**
- Adds `floyd serve --stdio --model DIR [--ctx N] [--ngen N] [--draft N] [--prefix-cache-mb N]`.
- Adds user flags `--ds4-model FILE`, `--ds4-support FILE`, and `--trace`; legacy environment variables remain compatible.
- Makes `floyd run --model <DeepSeek-V4> --prompt TEXT` use the built-in V4 runtime.

- [ ] Write shell tests for help text, required `--stdio`, invalid cache budgets, unknown/malformed requests, two JSON responses from one process, and unchanged chat/run dispatch.
- [ ] Run `make METAL=1 test-deepseek-v4-serve-cli`; verify RED because `serve` is unknown.
- [ ] Extend `cli_adapt()` with an explicit command enum and validated command-specific flags. Route V4 `run` and `serve` before the Moonlight loader; pass options directly rather than making new user features depend on environment variables.
- [ ] Run CLI tests plus `make test-cli-default-chat test-deepseek-v4-chat-dispatch`; verify GREEN.
- [ ] Commit with `git commit -m "feat: expose DeepSeek V4 stdio serve CLI"`.

### Task 5: README And End-To-End Verification

**Files:**
- Modify: `README.md`
- Create: `tests/test_deepseek_v4_serve_official.sh`
- Modify: `Makefile`

**Interfaces:**
- Documents download, conversion/preparation, Metal build, chat, run, and JSONL serve without making environment variables the primary interface.

- [ ] Add an opt-in official smoke that runs cache-disabled and cache-enabled requests, verifies identical greedy output, and requires `cache_hit:true` plus positive `cache_prefix_tokens` on the shared-prefix request.
- [ ] Rewrite the README quickstart around user commands. Keep one short performance note and point to `make METAL=1 test-deepseek-v4-ds4-40tps` for measurement.
- [ ] Run protocol/cache/stdio/CLI tests, `make test-c`, related V4 tests, Moonlight tiny TF/greedy, and the official DS4 smoke when model paths are available.
- [ ] Run `git diff --check`, `git status --short --untracked-files=all`, `git log --oneline`, and audit tracked files for models, fixtures, binaries, logs, and `AGENTS.md`.
- [ ] Commit with `git commit -m "docs: add Floyd agent serve quickstart"`.
