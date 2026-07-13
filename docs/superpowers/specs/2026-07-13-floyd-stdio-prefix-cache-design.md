# Floyd Stdio Serve And Prefix Cache Design

## Goal

Add a persistent `floyd serve --stdio` process for agent runtimes. It accepts
independent JSONL requests, keeps the DeepSeek V4 model resident, and reuses
compatible DS4 KV snapshots through a bounded in-memory prefix LRU. Existing
`chat`, `run`, `tf`, `gen`, and `help` behavior remains unchanged. HTTP,
concurrency, disk cache, and token streaming are outside this milestone.

## JSONL Contract

Stdin contains one JSON object per line and stdout emits exactly one final JSON
object for that request. Diagnostics remain on stderr so stdout is safe for
machine parsing. Requests are processed sequentially.

A request requires `id` (string or integer) and exactly one of `prompt` or
`messages`. `messages` is an array of `{ "role", "content" }` objects using
`system`, `user`, or `assistant`; `system` is also accepted as a convenience
with `prompt`. Optional fields are `max_tokens`, `temperature`, `top_p`, and
`draft`. Unknown fields are ignored for forward compatibility, while malformed
types, invalid ranges, conflicting prompt forms, and context overflow return a
per-request error without terminating the server.

Success output contains `id`, `text`, `usage`, `stats`, `cache_hit`,
`cache_prefix_tokens`, and `error: null`. `usage` reports prompt, cached,
uncached, and completion tokens. `stats` reports prompt/decode milliseconds and
the existing speculative counters. Errors echo `id` when available and contain
an object with stable `code` and human-readable `message`. Streaming events are
deferred.

Greedy requests use `temperature: 0` and may use DSpark speculation. Sampling
uses the existing DS4 sampler with `top_p`; sampling and speculative draft are
mutually exclusive in this milestone and produce a clear configuration error
when combined. A request-level `draft` may disable speculation or select a
valid supported draft length for the sequential server.

## Prefix LRU

The Floyd-owned cache stores exact token prefixes plus DS4 in-memory snapshots.
Before generation, the runner finds the longest cached prefix, restores it,
and asks `ds4_session_sync()` to evaluate only the suffix. It stores useful
message-boundary anchors, especially the system/history prefix before the final
user message, as well as the complete pre-generation prompt. This allows two
independent agent requests with a shared system/tools/history prefix to reuse
KV even when their final user messages diverge.

Keys include canonical base-model and DSpark-support fingerprints (path, size,
mtime), DS4 payload ABI, context size, exact token ids, and the relevant draft,
temperature, and top-p configuration. Full token equality is checked after a
hash match. `max_tokens` is not part of the key because it cannot alter the
pre-generation KV state.

`--prefix-cache-mb N` sets a hard byte budget; `0` disables the cache. The
default is a conservative 256 MiB and allocates nothing until a snapshot is
saved. Before allocation, the runner calls `ds4_session_payload_bytes()`. An
entry larger than the budget is skipped. Otherwise least-recently-used entries
are evicted until the new snapshot fits. The effective insertion budget also
preserves 1 GiB of currently available host memory when that measurement is
available; allocation or snapshot failure degrades to a cache miss rather than
terminating inference. Cache counters and current bytes are exposed in stats.

## CLI And Documentation

The public CLI becomes the only documented user configuration surface. The
serve command accepts `--model`, `--stdio`, `--ctx`, `--ngen`, `--draft`,
`--prefix-cache-mb`, and explicit DS4 base/support model flags. Environment
variables remain compatibility/debug inputs but are removed from primary
examples. `floyd run --model ... --prompt ...` must correctly dispatch to the
DeepSeek V4 runtime.

README quickstart covers Hugging Face download, DS4 preparation,
`make METAL=1 floyd`, chat, one-shot run, and a two-request stdio example that
shows `cache_prefix_tokens`. Performance documentation stays brief and points
to the benchmark target instead of preserving experimental result history.

## Verification

TDD starts with model-free tests for JSON parsing, response escaping, request
error isolation, byte-budget LRU eviction, oversize skip, longest-prefix lookup,
and fingerprint/config separation. A mock snapshot backend drives a complete
multi-request stdio smoke. Existing CLI tests guard all commands. An opt-in
official DS4 smoke sends two requests with a shared prefix and requires the
second response to report a positive cache prefix without changing generated
tokens. No models, generated fixtures, binaries, logs, or `AGENTS.md` are
committed.
