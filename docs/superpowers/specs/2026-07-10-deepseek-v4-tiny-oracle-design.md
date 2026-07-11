# DeepSeek V4 Tiny Oracle Design

## Purpose

The V4 port needs a deterministic, CPU-sized correctness oracle before it can safely consume the official quantized DSpark checkpoint. The fixture covers every base-forward mechanism in three layers: sliding attention, HCA, CSA with indexer, mHC residual streams, hash MoE, and square-root-softplus top-k MoE.

DSpark speculative decoding is intentionally excluded. The official repository describes DSpark as an attached acceleration module; base V4 forward parity is a prerequisite for verifying its drafts.

## Fixture

`tools/make_deepseek_v4_oracle.py` builds a three-layer `DeepseekV4ForCausalLM` with hidden size 64, two mHC streams, eight routed experts, top-2 routing, and a 128-token vocabulary. Fixed seed 1234 initializes weights. Hash-router tables contain deterministic token-dependent expert pairs rather than the all-zero initialization used by Transformers.

The default output directory is `fixture_tiny_deepseek_v4`, which is already covered by the repository's `fixture_tiny*` ignore rule. It contains `config.json`, `model.safetensors`, and `oracle.safetensors`. The separate JSON reference records prompt IDs, greedy full IDs, and per-position argmax predictions.

## Oracle Data

`oracle.safetensors` stores full F32 logits, every layer boundary hidden state, and learned-router logits. Hidden states retain their native `[batch, sequence, hc_mult, hidden]` shape until the final hyper-head collapses the mHC streams.

Greedy generation runs without KV cache so cache implementation cannot contaminate the architectural reference. Cache parity will be a separate milestone after full-sequence forward matches.

## Acceptance

The builder must run on CPU with Transformers 5.13, produce stable shapes and IDs across two runs, and yield a checkpoint that the repository's safetensors probe can index. No generated fixture or reference artifact is committed.
