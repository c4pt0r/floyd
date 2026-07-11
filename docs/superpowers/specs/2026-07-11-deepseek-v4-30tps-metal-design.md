# DeepSeek V4 30 tok/s Metal Design

## Goal

Run the official `DeepSeek-V4-Flash-DSpark` checkpoint through the existing
`floyd --model <directory>` chat entrypoint at at least 30 emitted tokens per
second on the local M3 Ultra. Measure decode over at least 32 emitted tokens
after model load and prefill. Preserve Moonlight behavior and verify DeepSeek
token output against the existing official checkpoint oracle.

## Architecture

Keep the current C DeepSeek runtime as a numerical reference, not the production
decode engine. Use a pinned ggml/llama.cpp revision for the resident Metal graph:
it already implements DeepSeek4 HCA, CSA, Lightning Indexer, compressed KV, and
MXFP4 MoE SIMD-group matvec. An offline preparation command repacks the ignored
safetensors checkpoint into ignored GGUF shards. A narrow C ABI bridge lets the
existing C CLI own dispatch while ggml owns model loading, tokenization, KV, and
base forward execution.

Base autoregressive decode is expected to approach the machine's memory-bandwidth
limit but may remain below 30 tok/s. The final stage imports the checkpoint's MTP
weights and implements verified DSpark block proposals in the same resident ggml
runtime. Only base-verified proposal prefixes are emitted, so disabling DSpark
must produce the same greedy token sequence.

## Runtime Contract

`make METAL=1 floyd` builds GPU support by default. DeepSeek dispatch must report
the selected backend and fail clearly when the prepared GGUF is absent; it must
not silently use the minute-per-token reference path. Model load happens once,
weights stay mmap-backed/resident, and decode does not spawn Python or another
inference process. Python is allowed only in offline conversion and oracle tests.

## Verification

Tests cover prepared-model discovery, C bridge errors, official prompt/token
parity, greedy output parity with DSpark on and off, and Moonlight regressions.
Benchmarks report load time, prompt tok/s, base decode tok/s, DSpark acceptance,
and emitted tok/s. The goal is complete only when the official checkpoint emits
at least 30 tok/s. Models, GGUF shards, dependency checkouts, fixtures, binaries,
logs, and `AGENTS.md` remain untracked.
