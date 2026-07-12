# DeepSeek V4 40 Tok/s Design

## Goal And Acceptance

Raise the default resident DS4 Metal path from the measured 36.401 tok/s to at
least 40.000 emitted tok/s on the local M3 Ultra. The authoritative gate uses
the official Q2-imatrix checkpoint through the unified `floyd` chat CLI, greedy
decoding, a Chinese story prompt, and at least 32 emitted tokens. A 128-token
run must independently remain at or above 40 tok/s before completion.

Startup/page-warming time and prompt throughput are reported separately. They
must not be removed from logs or counted as decode throughput.

## Optimization Sequence

1. Download the official 3.5 GB DS4 MTP component outside the repository.
   Benchmark draft lengths 2, 3, and 4 against the non-speculative baseline.
   Record accepted extra tokens per speculative round and end-to-end emitted
   tok/s. Keep MTP only if it improves the 128-token result without changing the
   greedy token stream.
2. If the best MTP result is below 40 tok/s, enable DS4 decode-stage profiling
   for representative HCA and CSA layers. Select the largest measured stage,
   then make one narrowly scoped change in the pinned DS4 integration or Floyd
   bridge. Profiling mode is diagnostic only because it disables fast paths.
3. Re-run the 32- and 128-token gates after every candidate. Reject changes
   that merely move work outside the timer, reduce emitted-token count, or rely
   on a warm prior process without reporting that condition.

Likely implementation candidates, chosen only after profiling, are reducing
per-token CPU/GPU synchronization, retaining argmax/verification on Metal, and
coalescing command buffers around the dominant decode stage. Quantization will
remain Q2-imatrix; Q4 is a quality option rather than the performance target.

## Runtime Contract

`floyd --model <official-checkpoint>` remains the shared entry point for
Moonlight and DeepSeek V4. DeepSeek startup must print `backend=metal-ds4`, and
performance logs must retain generated-token, speculative-round, accepted-token,
and decode-tps fields. Explicit `FLOYD_DEEPSEEK_V4_GGUF` continues to select the
exact native-MXFP4 fallback.

MTP configuration will be explicit and validated. Missing or incompatible MTP
weights must fail clearly rather than silently reporting speculative mode.

## Correctness And Repository Safety

The exact official fallback must still produce token `19923` (`Hello`). Native
FP4/FP8 Metal parity must remain `max_abs=0` with no CPU fallback. Moonlight tiny
teacher-forcing and greedy checks must remain 32/32 and 20/20, and both model
families must retain the unified chat UX.

Model files, generated fixtures, binaries, profiles, logs, `.deps`, and
`AGENTS.md` remain untracked. Changes are committed in small TDD-backed steps;
the final branch is merged to `master` and pushed only after the 40 tok/s gates
and regressions pass.
