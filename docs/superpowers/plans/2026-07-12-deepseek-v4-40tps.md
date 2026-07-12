# DeepSeek V4 40 Tok/s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise official DS4 Q2-imatrix chat decode to at least 40 emitted tok/s for both 32- and 128-token runs while preserving exact and Moonlight parity.

**Architecture:** First expose the DS4 speculative controls through a validated C configuration boundary and the existing `--draft` CLI. Then benchmark the official MTP component with token-exact base/spec traces. If MTP does not clear both performance gates, use DS4's layer-scoped decode profiler to choose a single Metal optimization and repeat the same gates.

**Tech Stack:** C99, Objective-C/Metal, pinned antirez/ds4, Make, POSIX shell, official DS4 GGUFs.

## Global Constraints

- Performance means emitted tokens divided by measured decode time in unified `floyd` chat.
- Both 32-token and 128-token Chinese-story runs must report at least 40.000 tok/s.
- Base and speculative greedy token ID streams must be identical.
- Do not commit models, fixtures, binaries, profiles, logs, `.deps`, or `AGENTS.md`.
- Preserve explicit native-MXFP4 fallback and Moonlight chat/parity behavior.

---

### Task 1: Validated MTP Controls

**Files:**
- Modify: `deepseek_v4_ds4.h`
- Modify: `deepseek_v4_ds4.c`
- Modify: `floyd.c`
- Modify: `tests/test_deepseek_v4_ds4.c`
- Modify: `tests/test_deepseek_v4_chat_dispatch.sh`

**Interfaces:**
- Produces: `DeepSeekV4Ds4SpecConfig { int draft_tokens; float margin; }`
- Produces: `deepseek_v4_ds4_spec_config_from_env(config, error, error_size)` returning 1 for valid configuration and 0 for invalid input.
- Consumes: `DRAFT` from `--draft N` and optional `FLOYD_DEEPSEEK_V4_DS4_MTP_MARGIN`; defaults are draft 2 and margin 3.0.

- [ ] **Step 1: Write failing configuration and CLI tests**

Add checks that defaults are `2/3.0`, `DRAFT=4` and margin `2.5` parse exactly,
draft outside `2..16` and non-positive/non-finite margin fail with a useful
message, and `floyd ... --draft 3` reaches the DeepSeek speculative dispatch.

- [ ] **Step 2: Verify RED**

Run:

```bash
make test-deepseek-v4-ds4
make METAL=1 test-deepseek-v4-chat-dispatch \
  DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark
```

Expected: configuration symbols are missing and `--draft 3` does not emit the
speculative-mode diagnostic.

- [ ] **Step 3: Implement the minimal configuration boundary**

Parse with `strtol`/`strtof`, reject trailing input and invalid ranges, pass the
values into `ds4_engine_options`, and make DeepSeek `use_spec` true when either
`DSPARK_SPEC` is enabled or CLI `DRAFT` is greater than one.

- [ ] **Step 4: Verify GREEN and regressions**

Run the two RED commands plus `make test-c`; expect all to pass.

- [ ] **Step 5: Commit**

```bash
git add deepseek_v4_ds4.h deepseek_v4_ds4.c floyd.c \
  tests/test_deepseek_v4_ds4.c tests/test_deepseek_v4_chat_dispatch.sh
git commit -m "feat: expose DS4 speculative controls"
```

### Task 2: Official MTP Correctness And Benchmark Matrix

**Files:**
- Create: `tests/test_deepseek_v4_ds4_spec_official.sh`
- Modify: `Makefile`
- External only: `/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark-DS4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`

**Interfaces:**
- Consumes: `FLOYD_DEEPSEEK_V4_DS4_MTP`, `DRAFT`, and optional margin.
- Produces: a shell gate that compares traced token IDs and reports base/spec
  throughput, speculative rounds, accepted extra tokens, and acceptance ratio.

- [ ] **Step 1: Write the failing official spec gate**

Run base and speculative chat with `DEEPSEEK_V4_CHAT_TRACE=1`, 32 generated
tokens, and the same Chinese prompt. Extract `DEEPSEEK_V4_TOKEN` lines, compare
them with `cmp`, require `spec_rounds > 0`, and print parsed performance fields.

- [ ] **Step 2: Verify RED without MTP weights**

```bash
make METAL=1 test-deepseek-v4-ds4-spec-official \
  DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark
```

Expected: fail with a missing MTP GGUF diagnostic.

- [ ] **Step 3: Download the official external MTP component**

```bash
DS4_GGUF_DIR=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark-DS4 \
  /Users/dongxu/floyd/.deps/ds4/download_model.sh mtp
```

Verify the final file exists and no `.aria2`/`.part` sidecar remains.

- [ ] **Step 4: Run the draft/margin matrix**

For drafts `2 3 4` and margins `2.0 3.0 4.0`, run the official gate at 32 and
128 tokens. Save raw output only under `/tmp`; summarize emitted tok/s and
accepted extras per round in the working notes.

- [ ] **Step 5: Commit the reusable gate**

```bash
git add Makefile tests/test_deepseek_v4_ds4_spec_official.sh
git commit -m "test: gate official DS4 speculative decode"
```

### Task 3: Forty Tok/s Decision Gate

**Files:**
- Modify: `tests/test_deepseek_v4_ds4_official.sh`
- Modify: `README.md`
- Potentially modify: `deepseek_v4_ds4.c` only if MTP clears both gates.

**Interfaces:**
- Produces: `MIN_TPS=40` acceptance for `NGEN=32` and `NGEN=128`.

- [ ] **Step 1: Run RED at 40 tok/s on the base path**

```bash
make METAL=1 test-deepseek-v4-ds4-official DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark NGEN=32 MIN_TPS=40
make METAL=1 test-deepseek-v4-ds4-official DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark NGEN=128 MIN_TPS=40
```

Expected baseline: both fail near the existing 36 tok/s result.

- [ ] **Step 2: Apply the best verified speculative configuration**

Only when the Task 2 matrix shows identical token IDs and both runs above 40,
make that draft/margin the DS4 defaults when an explicit MTP path is present.
Otherwise leave defaults unchanged and proceed directly to Task 4.

- [ ] **Step 3: Verify GREEN or record the profiling decision**

Run both 40 tok/s commands. If either fails, do not lower the threshold; execute
Task 4. If both pass, update README with measured commands and values.

- [ ] **Step 4: Commit only a proven default change**

```bash
git add deepseek_v4_ds4.c tests/test_deepseek_v4_ds4_official.sh README.md
git commit -m "perf: enable verified DS4 speculative decode"
```

### Task 4: Profile-Guided Decode Optimization

**Files:**
- Modify after evidence: pinned DS4 integration patch under `patches/ds4/`, or
  `deepseek_v4_ds4.c` when the measured bottleneck is bridge synchronization.
- Test: `tests/test_deepseek_v4_ds4_official.sh`

**Interfaces:**
- Consumes: layer-scoped DS4 profiler output for CSA layer 2 and HCA layer 3.
- Produces: one optimization tied to the largest measured decode-stage cost.

- [ ] **Step 1: Capture diagnostic profiles outside git**

```bash
for layer in 2 3; do
  printf '写一个至少100字的中文故事。\n:exit\n' | \
    env DS4_METAL_DECODE_STAGE_PROFILE=1 \
      DS4_METAL_DECODE_STAGE_PROFILE_LAYER=$layer \
    ./floyd --model /Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark \
      --ctx 512 --ngen 8 >"/tmp/ds4-stage-$layer.log" 2>&1
done
```

Rank stages by total milliseconds. Profiling output is diagnostic and cannot be
used as the throughput result.

- [ ] **Step 2: Add a failing performance or call-count test for the top stage**

The test must exercise the observed path and fail on the current implementation:
either the 40 tok/s official gate or a deterministic counter proving an avoidable
CPU/GPU synchronization. Do not implement an unprofiled candidate.

- [ ] **Step 3: Implement one minimal optimization**

Patch only the selected stage. Preserve DS4's official token verification and
all environment switches. Pin the exact external revision/patch in `Makefile`.

- [ ] **Step 4: Re-run 32/128 gates and commit**

Both must exceed 40 tok/s in two consecutive runs. Commit the test and selected
implementation as separate focused commits.

### Task 5: Final Numerical And Integration Audit

**Files:**
- Modify: `README.md` only for final reproducible commands and measurements.

- [ ] **Step 1: Run DeepSeek correctness gates**

```bash
make METAL=1 test-deepseek-v4-native-quant-metal DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark
FLOYD_DEEPSEEK_V4_GGUF=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark-GGUF-cchuter/model-bf16-00001-of-00004.gguf \
  make METAL=1 test-deepseek-v4-ggml-official DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark
```

Expected: FP4/FP8 `max_abs=0`, no CPU fallback, token `19923` / `Hello`.

- [ ] **Step 2: Run unified chat and Moonlight gates**

```bash
make test-c
make METAL=1 test-cli-default-chat MOONLIGHT=/Users/dongxu/floyd/models/moonlight_i8
./floyd tf --model /Users/dongxu/floyd/fixture_tiny --ref /Users/dongxu/floyd/ref_tiny.json --cap 8 --ebits 16 --dbits 16
./floyd gen --model /Users/dongxu/floyd/fixture_tiny --ref /Users/dongxu/floyd/ref_tiny.json --cap 8 --ebits 16 --dbits 16
```

Expected: shared chat UX, 32/32 TF, and 20/20 greedy.

- [ ] **Step 3: Audit and integrate**

Run `git diff --check`, full untracked status, tracked-large-file audit, and
confirm no forbidden artifacts. Fast-forward `master`, rerun the 40 tok/s gate
from the merged tree, push `origin/master`, and verify remote/local hashes.
