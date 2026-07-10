# DSpark Weight Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index the official DSpark safetensors dtypes safely and provide an exact CPU reference decoder for its FP4 expert weights and E8M0 scales.

**Architecture:** Extend the existing header-only shard reader without changing legacy dtype codes. Add a separate header-only V4 quantization reference module so Moonlight's uniform int4 implementation remains untouched.

**Tech Stack:** C, safetensors headers, Make, official DeepSeek conversion table, existing `CHECK` test style.

## Global Constraints

- Preserve BF16/F16/F32/U8/I8 dtype codes and behavior.
- Never silently decode I64, FP8, or scale tensors as F16.
- Decode FP4 low nibble before high nibble, matching the official converter.
- Use one E8M0 scale per 32 logical FP4 values.
- Perform no heap allocation inside V4 scalar/row decode helpers.
- Preserve Moonlight teacher-forcing `32/32` and greedy `20/20`.

---

### Task 1: Raw DSpark Dtype Indexing

**Files:**
- Modify: `st.h`
- Modify: `tests/test_st.c`

**Interfaces:**
- Consumes: safetensors dtype strings.
- Produces: explicit `ST_DTYPE_*` constants and safe raw-only dtype recognition.

- [ ] **Step 1: Add failing dtype tests**

Assert that legacy strings retain codes 0 through 3 and that `I64`, `F8_E8M0`, and `F8_E4M3` map to distinct new constants.

- [ ] **Step 2: Run RED**

Run: `make tests/test_st && ./tests/test_st`

Expected: exit 1 with `dtype non gestito: I64`.

- [ ] **Step 3: Add explicit dtype constants**

Define constants with legacy values unchanged:

```c
enum {
    ST_DTYPE_BF16 = 0, ST_DTYPE_F16 = 1, ST_DTYPE_F32 = 2,
    ST_DTYPE_U8 = 3, ST_DTYPE_I64 = 4,
    ST_DTYPE_F8_E8M0 = 5, ST_DTYPE_F8_E4M3 = 6
};
```

Update `st_dtype_code`. Make both F32 conversion readers handle only BF16/F16/F32 explicitly and fail clearly for raw-only types.

- [ ] **Step 4: Run GREEN**

Run: `make tests/test_st && ./tests/test_st && make test-c`

Expected: all C tests pass.

- [ ] **Step 5: Commit**

```bash
git add st.h tests/test_st.c
git commit -m "v4 weights 1: index DSpark safetensors dtypes"
```

### Task 2: FP4 and E8M0 Reference Decoder

**Files:**
- Create: `v4_quant.h`
- Create: `tests/test_v4_quant.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: packed FP4 bytes, E8M0 scale bytes, logical element count.
- Produces: `v4_fp4_code_value`, `v4_e8m0_to_f32`, and `v4_fp4_dequant_row`.

- [ ] **Step 1: Add the failing test target**

Test all 16 official FP4 values, low/high nibble order, E8M0 codes 0/125/126/127/128/254/255, and a 64-element row whose scale changes at element 32.

- [ ] **Step 2: Run RED**

Run: `make tests/test_v4_quant`

Expected: Make fails because `v4_quant.h` is absent.

- [ ] **Step 3: Implement the reference decoder**

Use the exact table:

```c
{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
```

Decode scale byte 255 as `NAN`; otherwise return `ldexpf(1.0f, code - 127)`. Decode element `i` from the low nibble when `i` is even and multiply it by `scales[i / 32]`.

- [ ] **Step 4: Run GREEN**

Run: `make tests/test_v4_quant && ./tests/test_v4_quant && make test-c`

Expected: `v4 quant tests: ok` and all C tests pass.

- [ ] **Step 5: Commit**

```bash
git add Makefile v4_quant.h tests/test_v4_quant.c
git commit -m "v4 weights 2: add FP4 reference decoder"
```

### Task 3: Real Checkpoint Header Probe

**Files:**
- Create: `st_probe.h`
- Create: `tests/test_st_probe.c`
- Create: `tools/probe_safetensors.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: a local model directory accepted by `st_init`.
- Produces: total tensor count and per-dtype counts without reading tensor data.

- [ ] **Step 1: Add a failing probe-helper test**

Construct an in-memory `shards` value containing representative dtype codes and assert that `st_dtype_counts` and `st_dtype_name` return deterministic results. Add the test to `TEST_BINS` with a dependency on the absent `st_probe.h`.

- [ ] **Step 2: Run RED**

Run: `make tests/test_st_probe`

Expected: Make fails because `st_probe.h` is absent.

- [ ] **Step 3: Implement helper and probe tool**

Add allocation-free dtype counting and naming helpers in `st_probe.h`. Build `tools/probe_safetensors` against `st.h`, `st_probe.h`, `json.h`, and `compat.h`. The program accepts exactly one model directory and prints shard count, tensor count, and deterministic `dtype_name=count` lines.

- [ ] **Step 4: Run GREEN and probe the DSpark directory**

Run:

```bash
make tools/probe_safetensors
tools/probe_safetensors /Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark
```

Expected after download completion: 48 shards index successfully, total tensor count is 72,317, and output includes I64, F8_E8M0, F8_E4M3, and I8.

- [ ] **Step 5: Run regression gates**

Run `make test-c`, build `floyd`, then run the tiny teacher-forcing and greedy commands from the previous milestone.

- [ ] **Step 6: Commit**

```bash
git add Makefile st_probe.h tests/test_st_probe.c tools/probe_safetensors.c docs/superpowers/specs/2026-07-10-dspark-weight-primitives-design.md docs/superpowers/plans/2026-07-10-dspark-weight-primitives.md
git commit -m "v4 weights 3: probe DSpark checkpoint headers"
```
