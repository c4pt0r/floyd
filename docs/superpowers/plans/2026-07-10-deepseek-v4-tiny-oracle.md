# DeepSeek V4 Tiny Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a deterministic tiny V4 checkpoint and layer-level oracle covering all base V4 attention and MoE modes.

**Architecture:** Use the native Transformers DeepSeek V4 implementation as the oracle. Save model weights separately from F32 activation references so the future C backend can index both with existing safetensors tooling.

**Tech Stack:** Python, PyTorch, Transformers 5.13, safetensors, JSON.

## Global Constraints

- Run entirely on CPU.
- Use seed 1234 and explicit hash-router tables.
- Cover sliding, HCA, and CSA attention in one fixture.
- Cover hash MoE and learned sqrt-softplus MoE.
- Generate fixed-length greedy output without KV cache.
- Do not commit generated weights or reference data.

---

### Task 1: Tiny Model Builder

**Files:**
- Create: `tools/make_deepseek_v4_oracle.py`
- Create: `tests/test_make_deepseek_v4_oracle.py`
- Modify: `Makefile`

**Interfaces:**
- Produces: `build_tiny() -> tuple[DeepseekV4ForCausalLM, DeepseekV4Config]` and `PROMPT_IDS`.

- [ ] **Step 1: Add a failing import test**

Import `build_tiny`, assert the three attention and MoE layer modes, verify token-dependent hash expert IDs, and run an eight-token forward checking hidden-state and logits shapes.

- [ ] **Step 2: Run RED**

Run: `PYTHON=/Users/dongxu/floyd/.venv/bin/python make test-deepseek-v4-oracle`

Expected: import fails because `tools.make_deepseek_v4_oracle` is absent.

- [ ] **Step 3: Implement the minimal builder**

Instantiate `DeepseekV4Config` with hidden size 64, head dim 16, three layers, `hc_mult=2`, eight experts, and layer modes `[sliding, HCA, CSA]` / `[hash_moe, moe, moe]`. Set deterministic `tid2eid` pairs and learned-router correction biases.

- [ ] **Step 4: Run GREEN**

Run the same Make target. Expected: `v4 oracle builder tests: ok`.

### Task 2: Checkpoint and Oracle Writer

**Files:**
- Modify: `tools/make_deepseek_v4_oracle.py`
- Modify: `tests/test_make_deepseek_v4_oracle.py`

**Interfaces:**
- Produces: `write_tiny(out_dir, ref_path, ngen)` and the model/reference artifacts described in the design.

- [ ] **Step 1: Add a failing temporary-directory test**

Call `write_tiny` with two generated tokens. Assert required files, JSON lengths, `oracle.logits`, all four hidden states, and two learned-router tensors.

- [ ] **Step 2: Run RED**

Expected: import or attribute failure because `write_tiny` is absent.

- [ ] **Step 3: Implement greedy and artifact writing**

Generate tokens by repeatedly running full-sequence `use_cache=False` forward and taking argmax. Run one final forward with hidden/router output, save contiguous F32 reference tensors, and write JSON with `prompt_ids`, `full_ids`, and `tf_pred`.

- [ ] **Step 4: Run GREEN and generate the local fixture**

```bash
PYTHON=/Users/dongxu/floyd/.venv/bin/python make test-deepseek-v4-oracle
/Users/dongxu/floyd/.venv/bin/python tools/make_deepseek_v4_oracle.py tiny
make tools/probe_safetensors
tools/probe_safetensors fixture_tiny_deepseek_v4
```

Expected: tests pass and the probe indexes both generated safetensors files.

### Task 3: Regression and Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the fixture command and scope**

Add a short development command noting that the fixture is an architecture oracle, not a performance model.

- [ ] **Step 2: Run all gates**

Run Python oracle tests, `make test-c`, Moonlight teacher-forcing, and Moonlight greedy parity.

- [ ] **Step 3: Commit source and docs**

```bash
git add Makefile README.md tools/make_deepseek_v4_oracle.py tests/test_make_deepseek_v4_oracle.py docs/superpowers/specs/2026-07-10-v4-tiny-oracle-design.md docs/superpowers/plans/2026-07-10-v4-tiny-oracle.md
git commit -m "v4 oracle: add deterministic architecture fixture"
```
