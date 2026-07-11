# DeepSeek V4 Chat Design

## Scope

Add a real interactive chat executable for the official
`DeepSeek-V4-Flash-DSpark` checkpoint. The completed path must tokenize the
official prompt format, prefill all 43 base layers, decode beyond position zero,
stream decoded text until EOS, preserve a multi-turn conversation, and use the
three DSpark stages only when their proposals are verified by the base model.
Tools, HTTP serving, sampling, and persistent on-disk KV are outside this first
chat milestone.

## Architecture

Keep the existing Moonlight `Model` and `run_serve()` unchanged. Add a focused
V4 runtime API that owns checkpoint shards, tokenizer, per-layer KV/compressor/
indexer state, token history, and reusable activation buffers. The runtime may
reuse the numerically verified primitives in `deepseek_v4_forward.h`, but chat code
must call a stable prefill/decode interface rather than fixture capture helpers.

`deepseek_v4_chat.c` provides the executable. It renders the official minimal chat-mode
format:

```text
<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>
```

Assistant tokens are appended through `<｜end▁of▁sentence｜>`; the next user turn
then appends another user/assistant prefix. `/clear` resets tokens and all model
state, while `/exit` terminates.

## Decode Contract

Prefill accepts any positive token count within configured context. Decode
accepts one token at the state's exact next position. Every attention layer
appends sliding-window KV; ratio-4 and ratio-128 layers update compressor
remainder state, compressed KV, and Lightning Indexer state at block boundaries.
The returned logits correspond to the final input token. Invalid positions,
capacity overflow, and incomplete state transitions fail explicitly.

Greedy base decode is the correctness baseline. DSpark proposes a five-token
block from the three speculative stages, then the base runner verifies the block
in one causal forward pass. Only the matching prefix is emitted; the first
mismatch emits the base token and discards the remaining proposal. Disabling
DSpark must produce identical token IDs.

## Verification

TDD gates use the official checkpoint and Python oracle. Tests cover tokenizer
and prompt IDs, prefill versus split prefill/decode activations, block-boundary
compressor/indexer updates, greedy token parity, DSpark accept/reject behavior,
and a real two-turn chat smoke. Existing V4 numerical tests, `make test-c`,
Moonlight teacher-forcing, and Moonlight greedy remain regression gates. Model
weights, generated fixtures, binaries, logs, and `AGENTS.md` remain untracked.
