# DeepSeek V4 Lightning Indexer Design

## Scope

Add a standalone F32 reference for the Lightning Indexer's scoring and causal top-k selection. The tiny CSA fixture is the correctness boundary. Query projection, query RoPE, indexer KV compression, KV cache, decode state, sparse attention gather, and quantization remain separate milestones.

## Oracle Contract

Run the native Transformers `DeepseekV4Indexer` on the existing deterministic 12-token CSA input. Hook its scorer and store:

- RoPE-applied queries `[tokens, heads, head_dim]`
- compressed indexer keys `[entries, head_dim]`
- raw `weights_proj` output `[tokens, heads]`
- unmasked scorer output `[tokens, entries]`
- absolute query positions and final causal top-k indices

The tiny configuration uses four heads, head dimension eight, ratio four, three compressed entries, and top-k two. This exercises queries with zero, one, two, and three causally available entries.

## C Interface

`v4_indexer.h` consumes these precomputed F32 boundaries. For each query/key/head it computes `ReLU(q dot k) / sqrt(head_dim)`, multiplies by the projected head weight divided by `sqrt(n_heads)`, and sums across heads. Causal selection permits entry `e` only when `e < (position + 1) / ratio`; unavailable result slots are `-1`.

Reject null pointers, non-positive dimensions, `top_k > n_entries`, negative positions, and unsupported empty inputs. Keep selection deterministic for distinct scores; exact tie ordering is unspecified, matching PyTorch's contract.

## Acceptance

Raw score `max_abs` must be below `3e-5`, and all tiny top-k slots must match exactly. Existing V4, C, build, and Moonlight parity gates remain green. Generated fixtures and binaries are not committed.
