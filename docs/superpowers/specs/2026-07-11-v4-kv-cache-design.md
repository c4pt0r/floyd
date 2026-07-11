# DeepSeek V4 KV Cache Design

## Scope

Add a caller-owned F32 KV cache for completed V4 shared-KV entries. Ring mode retains the latest sliding-window keys; append mode retains compressed HCA/CSA or indexer entries up to capacity. This milestone excludes compressor remainder buffers, CSA overlap Ca/Cb state, quantized storage, multi-batch state, and cache rollback.

## State and Contract

`V4KVCacheF32` stores key rows, absolute positions, capacity, head dimension, count, write cursor, and mode. Appends must have strictly increasing non-negative positions. Ring mode overwrites the oldest row at capacity; append mode rejects overflow. Chronological access hides physical ring layout, and copy-out provides the contiguous input expected by `v4_indexer_forward_f32`.

`v4_sliding_attention_cached_f32` projects and rotates each incoming key at its absolute position, appends it, attends only to the cache's chronological rows, and emits the normal grouped output. A multi-token call is prefill; repeated one-token calls are decode. Both use the same loop and require contiguous attention positions.

## Oracle and Acceptance

Extend the tiny oracle with the native sliding layer's RoPE-applied keys. The C fixture must verify:

- ring append order and retained absolute positions after window wrap;
- cached keys against native oracle keys;
- prefill and one-token decode outputs against the existing attention oracle;
- append-cache copy-out as the existing Lightning Indexer input, preserving raw scores and top-k.

Attention/key `max_abs` must be below `3e-5`; indexer top-k must remain exact. Existing compressor, indexer, V4, C, build, and Moonlight gates stay green.
