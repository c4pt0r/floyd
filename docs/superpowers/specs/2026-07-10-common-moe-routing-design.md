# Common MoE Routing Core Design

## Scope

The first general-MoE milestone extracts routing policy from the Moonlight forward path without changing expert storage, cache behavior, attention, or model configuration. It provides reusable score, selection, and weight-finalization primitives for model families that use softmax, sigmoid, or square-root softplus expert affinity. The concrete final target is `deepseek-ai/DeepSeek-V4-Flash-DSpark` revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`.

DeepSeek V4 attention, mHC residual streams, hash-MoE lookup, official FP4/FP8 storage, tokenizer changes, and the DSpark speculative module remain separate milestones. The V4 target forward must pass independently before DSpark acceleration is enabled. This keeps the existing V3/Moonlight parity gate authoritative while establishing one component V4 and other MoE backends can reuse.

## Interface

`moe_route.h` is a header-only module matching the repository's existing support-library style. It exposes:

- `moe_route_select(...)`: transform router logits into affinity scores, apply an optional correction bias only for expert selection, choose a stable top-k, and return unnormalized selected affinities.
- `moe_route_finalize(...)`: optionally normalize selected affinities and apply the model's routed scaling factor after any caller-specific top-p reduction.
- `MoeScoreFn`: `MOE_SCORE_SOFTMAX`, `MOE_SCORE_SIGMOID`, and `MOE_SCORE_SQRT_SOFTPLUS`.

The caller owns scratch buffers so the hot path adds no hidden allocation. Equal selection scores retain the lower expert index, preserving the current engine's deterministic scan order.

## Compatibility

Moonlight continues to use sigmoid affinity, `e_score_correction_bias` for selection only, optional selected-weight normalization, and `routed_scaling_factor`. Existing `TOPK` and expert `TOPP` overrides remain in `floyd.c`; finalization runs only after those policies determine the effective expert count.

## Verification

Unit tests cover all three score functions, bias semantics, stable ties, normalization, and routed scaling. Integration is accepted only if `make test-c`, CPU compilation, tiny teacher-forcing `32/32`, and tiny greedy `20/20` all pass.
