# V4 Real HCA Runner Design

## Scope

Extend the official-checkpoint F32 reference runner from layers 0-2 through the
first HCA layer (layer 3), then reuse the same caller-owned state for base layers
3-42. This path remains a correctness gate: packed FP4/FP8 weights are consumed
directly, but no performance-oriented refactor is included.

## Official Contract

Layer modes come from `compress_ratios`: layers 0-1 are sliding-only, even layers
2-42 use overlapping CSA-4 plus Lightning Indexer, and odd layers 3-41 use
non-overlapping HCA-128 without an indexer. All compressed layers use compressed
YaRN RoPE. During a 128-token layer-3 prefill, the first HCA entry is causal only
for token 127; earlier queries see only their sliding keys.

Layers 0-2 use hash routing. Layer 3 and later select top-6 experts using
`sqrt(softplus(logit)) + gate.bias`, while routing weights use the unbiased scores,
sum-normalized and scaled by 1.5.

## State Boundary

`V4RealLayerState` is caller-owned and records the layer mode, absolute position,
the 128-entry sliding KV ring, completed compressed KV entries, optional CSA
indexer KV entries, and compressor remainder/overlap projections. Prefill and
single-token decode update the same state. Attention receives selected absolute
entry IDs; absent entries are represented by `-1` and never dereferenced.

## Verification

The Python oracle uses the official checkpoint and captures layer-3 input,
projected HCA KV/gates, compressed KV, learned-router scores/indices/weights, and
full output. A 128-token case exercises the first real HCA boundary. C tests first
fail on the unsupported layer/state API, then require exact expert IDs and
`max_abs < 3e-4` for isolated layer-3 state and output. Existing layers 0-2 and all
V4/Moonlight gates remain unchanged.
