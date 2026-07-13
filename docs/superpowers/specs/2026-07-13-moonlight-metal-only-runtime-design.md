# Moonlight Metal-Only Runtime Design

## Goal

Make Floyd a macOS/Apple-Silicon Metal-only inference engine supporting both
Moonlight-16B-A3B and DeepSeek V4 DS4. During migration, the existing Moonlight
CPU implementation remains only as a numerical oracle. After complete tiny and
real-model parity, remove every CPU inference, numerical reference, and fallback
path, along with the user-facing `tf` and `gen` commands.

This design supersedes the build assumptions in the earlier user-first README
design. Host-side CLI parsing, model metadata, tokenization, file I/O, command
submission, and cache bookkeeping necessarily remain ordinary host code; the
"Metal-only" boundary applies to model numerical execution.

## Target Architecture

The final `floyd` binary exposes only `chat`, `run`, `serve --stdio`, and
`help`. A small CLI dispatcher selects one of two resident Metal runtimes:

- Moonlight native Metal runtime owned by Floyd.
- DeepSeek V4 resident DS4 Metal runtime through the pinned DS4 integration.

The Moonlight runtime uses persistent Metal buffers and command queues. It must
not route small matrices or single-token decode back to host kernels. Its graph
covers token embeddings, RMSNorm, MLA projections and absorption, RoPE, causal
attention, compressed KV append/read, sigmoid/noaux expert routing, shared and
routed SwiGLU experts, residuals, final norm, and lm-head. Prefill and decode
share the same model/session state and differ only in batch shape.

## Migration Boundaries

### Phase 1: Oracle Capture And Runtime Skeleton

Keep the existing CPU runtime unchanged as a temporary executable oracle.
Capture tiny and real Moonlight intermediate activations, logits, token streams,
and multi-turn chat outputs. Introduce a separate Metal session API so CPU and
Metal can be compared without adding more conditionals to `floyd.c`.

### Phase 2: Metal Numerical Closure

Implement and validate in dependency order:

1. Persistent tensor/buffer ownership and model loading.
2. Embedding, RMSNorm, final norm, and lm-head.
3. MLA projections, RoPE, causal attention, and compressed KV state.
4. Router scoring/selection and shared/routed experts.
5. Full layer, prefill, single-token decode, and multi-turn session state.

Each stage starts with a failing test against captured real or tiny oracle data.
Tolerance changes require measured error analysis; token-only checks cannot
replace intermediate activation parity.

### Phase 3: Product Cutover

After tiny teacher-forcing is exact, tiny greedy is exact, real-model logits are
within the established quantized tolerance, and real greedy/multi-turn chat
match the CPU oracle, switch `chat`, `run`, and `serve` to the Moonlight Metal
session. Benchmark prefill and decode; final throughput must exceed the captured
CPU baseline on the same checkpoint and hardware.

### Phase 4: CPU Removal And Simplification

Delete the Moonlight CPU engine, CPU quantized matmul and attention kernels,
Metal-to-CPU fallbacks, CUDA code, `tf`/`gen`, runtime backend toggles, and their
tests/documentation. Replace the current large `floyd.c` with the thin CLI and
model dispatch. Remove unused conversion/oracle utilities after preserving only
the preparation path required by the Metal model format.

Add a pinned DS4 patch that removes its `DS4_BACKEND_CPU`, CPU KV/session,
CPU forward code, and CPU-vs-GPU diagnostic entry points. The patch must apply
from a clean pinned revision and the official DS4 chat/run/serve tests must pass
afterward.

## Build And User Interface

`make METAL=1 floyd` is the sole runtime build. Plain `make`, `make floyd`,
`METAL=0`, non-Darwin hosts, and unsupported hardware fail with a concise
English message. `--metal` and runtime CPU backend selection are removed because
Metal is no longer optional. All public help, validation errors, and chat
banners use English.

README begins with a copy-pasteable Metal build and built-in chat quickstart,
then covers one-shot run, stdio serve, memory constraints, Moonlight model
preparation, and DeepSeek V4 preparation. Architecture history and migration
evidence remain below the user path.

## Verification And Completion Gates

Migration evidence must include:

- Tiny Moonlight intermediate max-absolute errors for every numerical stage.
- Tiny teacher-forcing and greedy token parity.
- Real Moonlight prefill/logit parity, greedy parity, and at least two chat turns
  that prove KV continuity.
- No Moonlight Metal CPU fallback calls during prefill or decode.
- Moonlight Metal prefill and decode faster than the captured CPU baseline.
- DeepSeek V4 official chat, run, stdio prefix-cache, and speculative tests.
- A clean build proving Metal is mandatory.
- Source audit finding no CPU inference/reference/fallback, CUDA runtime,
  `tf`/`gen`, stale user documentation, models, fixtures, binaries, logs, or
  committed `AGENTS.md`.

The CPU oracle is removed only after all parity gates pass. Until then, its
presence is explicitly temporary and does not count as completion.
