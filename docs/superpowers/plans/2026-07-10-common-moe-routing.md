# Common MoE Routing Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a tested routing core that supports softmax, sigmoid, and square-root softplus MoE affinity while preserving Moonlight inference exactly.

**Architecture:** Add a header-only pure routing module with caller-owned scratch storage. Keep expert top-p, storage, loading, and execution in `floyd.c`, and integrate only the existing sigmoid selection/finalization block.

**Tech Stack:** C11-compatible C, Make, existing `CHECK`-style test binaries, Transformers-generated tiny parity fixture.

## Global Constraints

- Preserve the legacy `floyd` CLI and environment variables.
- Do not change expert cache, prefetch, quantization, attention, sampling, or tokenizer behavior.
- Correction bias affects selected expert IDs but never the mixture weights.
- New routing functions perform no heap allocation.
- Keep `32/32` teacher-forcing and `20/20` greedy tiny parity.

---

### Task 1: Pure Routing Module

**Files:**
- Create: `moe_route.h`
- Create: `tests/test_moe_route.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: router logits, optional selection bias, expert count, top-k, and `MoeScoreFn`.
- Produces: `moe_route_select(...)` and `moe_route_finalize(...)` with caller-owned output and scratch arrays.

- [ ] **Step 1: Add a failing test target**

Add `tests/test_moe_route` to `TEST_BINS`, compile `tests/test_moe_route.c` against `moe_route.h`, and remove the binary in `clean`. The test must include `../moe_route.h` and check sigmoid bias semantics, softmax probabilities, square-root softplus values, deterministic ties, and final normalization/scaling.

- [ ] **Step 2: Run the test and verify RED**

Run: `make tests/test_moe_route`

Expected: compilation fails because `moe_route.h` does not exist.

- [ ] **Step 3: Implement the minimal header**

Define:

```c
typedef enum {
    MOE_SCORE_SOFTMAX,
    MOE_SCORE_SIGMOID,
    MOE_SCORE_SQRT_SOFTPLUS
} MoeScoreFn;

static int moe_route_select(const float *logits, const float *selection_bias,
                            int n_experts, int top_k, MoeScoreFn score_fn,
                            int *indices, float *weights,
                            float *scores, float *selection_scores);

static void moe_route_finalize(float *weights, int count,
                               int normalize, float routed_scale);
```

Use max-subtracted softmax, stable softplus (`x > 20 ? x : log1pf(expf(x))`), strict `>` top-k comparisons, and no allocations.

- [ ] **Step 4: Run the focused and complete C tests**

Run: `make tests/test_moe_route && ./tests/test_moe_route && make test-c`

Expected: `moe routing tests: ok`, followed by all C tests passing.

- [ ] **Step 5: Commit the module**

```bash
git add Makefile moe_route.h tests/test_moe_route.c
git commit -m "moe 1: add reusable routing core"
```

### Task 2: Moonlight Integration

**Files:**
- Modify: `floyd.c`

**Interfaces:**
- Consumes: Task 1's `moe_route_select(...)` and `moe_route_finalize(...)`.
- Produces: unchanged `moe(...)` behavior using the shared sigmoid implementation.

- [ ] **Step 1: Add a reduced-set assertion before integration**

As part of Task 1's initial failing test file, include the Moonlight ordering case used by `floyd.c`: select with sigmoid plus bias, reduce the effective count, then call `moe_route_finalize`. Assert the retained weights are normalized and scaled only after reduction. This assertion is RED with the rest of Task 1 until `moe_route.h` exists, then becomes the characterization test for integration.

- [ ] **Step 2: Capture the pre-integration parity baseline**

Run:

```bash
SNAP=/Users/dongxu/floyd/fixture_tiny TF=1 REF=/Users/dongxu/floyd/ref_tiny.json ./floyd 8 16 16
SNAP=/Users/dongxu/floyd/fixture_tiny REF=/Users/dongxu/floyd/ref_tiny.json ./floyd 8 16 16
```

Expected: teacher-forcing reports `32/32` and greedy reports `20/20` before the refactor.

- [ ] **Step 3: Replace only Moonlight's local routing math**

Include `moe_route.h`. In `moe(...)`, retain router matmul, `TOPK`, expert `TOPP`, usage accounting, and batch union. Replace the sigmoid/bias/top-k loop with `moe_route_select(..., MOE_SCORE_SIGMOID, ...)`, then replace normalization/scaling loops with `moe_route_finalize(w, Ke, c->norm_topk, c->routed_scale)` after `Ke` is final.

- [ ] **Step 4: Run unit and parity verification**

Run:

```bash
make clean && make test-c && make
SNAP=/Users/dongxu/floyd/fixture_tiny TF=1 REF=/Users/dongxu/floyd/ref_tiny.json ./floyd 8 16 16
SNAP=/Users/dongxu/floyd/fixture_tiny REF=/Users/dongxu/floyd/ref_tiny.json ./floyd 8 16 16
```

Expected: all C tests pass, teacher-forcing reports `32/32`, and greedy reports `20/20`.

- [ ] **Step 5: Commit the integration**

```bash
git add floyd.c tests/test_moe_route.c
git commit -m "moe 2: route Moonlight through common core"
```

### Task 3: Milestone Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: verified behavior from Tasks 1 and 2.
- Produces: a short architecture note identifying the reusable routing layer and its current limits.

- [ ] **Step 1: Document supported routing policies**

Add a concise development note stating that the shared core supports softmax, sigmoid, and square-root softplus, while execution remains Moonlight-specific and V4 hash routing/attention are not yet implemented.

- [ ] **Step 2: Verify documentation and repository state**

Run: `git diff --check && make test-c`

Expected: no whitespace errors and all C tests pass.

- [ ] **Step 3: Commit the documentation**

```bash
git add README.md docs/superpowers/specs/2026-07-10-common-moe-routing-design.md docs/superpowers/plans/2026-07-10-common-moe-routing.md
git commit -m "docs: define common MoE routing milestone"
```
