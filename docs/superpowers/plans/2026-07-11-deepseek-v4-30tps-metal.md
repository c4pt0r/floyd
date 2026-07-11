# DeepSeek V4 30 tok/s Metal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the streaming DeepSeek correctness path with a resident ggml Metal runtime and verified DSpark generation that reaches at least 30 emitted tok/s.

**Architecture:** A pinned, ignored llama.cpp checkout converts safetensors to GGUF and supplies the DeepSeek4 Metal graph. A focused C++ bridge preserves `floyd` CLI dispatch; the current C implementation remains the oracle. DSpark is added only after base parity and performance are measured.

**Tech Stack:** C11, C++17, Make, llama.cpp/ggml, Metal, GGUF, safetensors, Python oracle.

## Global Constraints

- Benchmark `/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark` on the local M3 Ultra.
- `floyd --model <directory>` remains the common Moonlight and DeepSeek chat entrypoint.
- DeepSeek chat defaults to Metal and never spawns Python or an inference subprocess.
- Follow RED/GREEN TDD and do not relax numerical thresholds to hide divergence.
- Do not commit models, GGUF shards, dependency checkouts, fixtures, binaries, logs, or `AGENTS.md`.

---

### Task 1: Reproducible GGUF Preparation And Baseline

**Files:**
- Create: `tools/prepare_deepseek_v4_gguf.sh`
- Create: `tests/test_prepare_deepseek_v4_gguf.sh`
- Modify: `.gitignore`
- Modify: `Makefile`

**Interfaces:**
- Produces: `make prepare-deepseek-v4-gguf DSPARK=<dir>`, an ignored split GGUF and a revision manifest pinned to the validated llama.cpp commit.

- [ ] Write a shell test using a temporary fake checkpoint and fake converter; require argument validation, pinned revision propagation, output discovery, and no writes inside the source checkpoint.
- [ ] Run `sh tests/test_prepare_deepseek_v4_gguf.sh`; verify RED because the command does not exist.
- [ ] Implement the preparation script with `set -eu`, configurable `LLAMA_CPP_DIR`, explicit output directory, and converter exit propagation.
- [ ] Run the shell test to GREEN, then convert the official checkpoint and run `llama-cli` Metal greedy generation for at least 32 decode tokens.
- [ ] Record only commands and measured summaries in the commit message; commit no generated data.

### Task 2: Linked ggml Runtime Bridge

**Files:**
- Create: `deepseek_v4_ggml.cpp`
- Create: `deepseek_v4_ggml.h`
- Create: `tests/test_deepseek_v4_ggml.c`
- Modify: `Makefile`

**Interfaces:**
- Produces: `deepseek_v4_ggml_available`, `deepseek_v4_ggml_chat_run`, and backend timing counters through a C ABI.
- Consumes: a prepared first GGUF shard discovered from the original model directory or `FLOYD_DEEPSEEK_V4_GGUF`.

- [ ] Write a failing C test for missing model, invalid GGUF, backend name, and stable error text without loading the official weights.
- [ ] Link the pinned static llama/ggml libraries and verify RED at the undefined bridge symbols.
- [ ] Implement one model/context lifetime, official chat-template tokenization, Metal offload, greedy decode, streaming decode text, and llama timing extraction.
- [ ] Run the unit test and an official one-token smoke to GREEN; compare the first greedy token with the existing oracle.
- [ ] Commit the bridge separately from CLI dispatch.

### Task 3: Default DeepSeek Dispatch

**Files:**
- Modify: `deepseek_v4_chat.c`
- Modify: `deepseek_v4_chat.h`
- Modify: `tests/test_deepseek_v4_chat_backend.sh`
- Modify: `README.md`

**Interfaces:**
- `deepseek_v4_chat_run` selects ggml Metal for real chat; `FLOYD_DEEPSEEK_V4_REFERENCE=1` is an explicit correctness/debug override.

- [ ] Update the backend test to require `backend=metal-ggml`, absence of Python child processes, and a clear preparation error when GGUF is missing; verify RED.
- [ ] Route production chat through the bridge and prohibit silent fallback to the streaming reference runtime.
- [ ] Run the backend test, official token parity, `make test-c all test-tok`, and Moonlight teacher-forcing/greedy smoke.
- [ ] Commit default dispatch and documentation.

### Task 4: Resident DSpark Verification

**Files:**
- Modify: `deepseek_v4_ggml.cpp`
- Modify: `tools/prepare_deepseek_v4_gguf.sh`
- Create: `tests/test_deepseek_v4_ggml_dspark.c`

**Interfaces:**
- Produces: verified five-token DSpark proposal blocks plus counters for proposed, accepted, and emitted tokens.

- [ ] Add a real-checkpoint failing test for the existing accepted and rejected DSpark oracle cases; require identical greedy output with DSpark disabled.
- [ ] Extend conversion to retain the three target-layer states and checkpoint MTP tensors in an ignored companion GGUF.
- [ ] Build proposals in the resident graph, verify them with one causal base batch, and commit only the accepted prefix plus the first base mismatch token.
- [ ] Run numerical parity and report acceptance rate and emitted tok/s; commit after GREEN.

### Task 5: 30 tok/s Acceptance And Regression

**Files:**
- Create: `tests/bench_deepseek_v4_chat.sh`
- Modify: `Makefile`

**Interfaces:**
- Produces: `make bench-deepseek-v4-chat DSPARK=<dir>` with machine-readable load, prefill, base decode, DSpark acceptance, and emitted throughput.

- [ ] Write a benchmark gate that rejects fewer than 32 emitted tokens, non-Metal execution, token mismatch, or throughput below `30.0` tok/s; verify it fails on the reference backend.
- [ ] Profile the resident graph and make one measured change at a time until the gate passes; keep buffer lifetime and command batching inside ggml.
- [ ] Run all DeepSeek fixture/oracle/native-quant/manifest tests plus Moonlight teacher-forcing and greedy regression.
- [ ] Run `git diff --check`, `git status --short --untracked-files=all`, and inspect staged file sizes before the final commit and push.
