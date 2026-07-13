# Moonlight Metal-Only Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Floyd's Moonlight CPU inference engine with a persistent, native Metal runtime while preserving DeepSeek V4 DS4 Metal support and a small `chat`/`run`/`serve` product CLI.

**Architecture:** Keep the current CPU Moonlight code only as a temporary oracle while a separate opaque `MoonlightModel`/`MoonlightSession` API reaches numerical parity. Model metadata and safetensors indexing stay on the host; embeddings, norms, MLA, RoPE, compressed KV, routing, experts, and lm-head execute in Metal with persistent buffers and no inference fallback. After tiny and real parity plus a same-machine performance win, switch the CLI and delete the oracle, CPU/CUDA kernels, backend switches, and `tf`/`gen` commands.

**Tech Stack:** C99, Objective-C ARC, Metal Shading Language, Apple Metal/Foundation frameworks, safetensors via `st.h`, Python/PyTorch oracle tooling, shell/C regression tests.

## Global Constraints

- The final runtime supports macOS on Apple Silicon only and is built with `make METAL=1 floyd`.
- CPU inference/reference/fallback is temporary and must be deleted after parity; host CLI, tokenization, file I/O, and cache bookkeeping remain host code.
- The final public commands are `chat`, `run`, `serve --stdio`, and `help`; all public output is English.
- Moonlight Metal prefill and decode must both beat the captured CPU baseline on the same checkpoint and hardware.
- Do not commit checkpoints, generated fixtures, reference dumps, binaries, logs, `kernels_metal.h`, or `AGENTS.md`.
- Every numerical tolerance change requires a max-absolute/RMSE/cosine error report; token-only parity is not sufficient for intermediate stages.

## File Map

- `moonlight_model.h`, `moonlight_model.c`: Moonlight config parsing, tensor descriptors, safetensors ownership, and host-to-Metal upload metadata.
- `moonlight_metal.h`, `moonlight_metal.m`: opaque model/session API, persistent `MTLBuffer` ownership, command encoding, prefill/decode lifecycle, and counters.
- `moonlight_kernels.metal`: all Moonlight numerical kernels; no host numerical fallback.
- `moonlight_oracle.h`, `moonlight_oracle.c`: temporary activation writer used by the current CPU engine; deleted at cutover.
- `tools/make_moonlight_oracle.py`: deterministic tiny/real official-Python activation, logit, token, and multi-turn oracle generation; deleted or reduced to a non-product developer tool at cutover.
- `tests/test_moonlight_metal_primitives.c`: embedding, RMSNorm, residual, q8/q4 matmul, and lm-head parity.
- `tests/test_moonlight_metal_mla.c`: MLA projection, RoPE, causal attention, and compressed KV parity.
- `tests/test_moonlight_metal_moe.c`: router selection/weights and shared/routed expert parity.
- `tests/test_moonlight_metal_runtime.c`: full-layer, prefill, decode, and stats parity.
- `tests/test_moonlight_metal_official.sh`: opt-in real checkpoint teacher-forcing, greedy, multi-turn, and benchmark gate.
- `tests/test_metal_only_build.sh`: final negative/positive build and source audit.
- `floyd.c`, `Makefile`, `README.md`: final thin CLI, Metal-only build, and user quickstart.

---

### Task 1: Freeze Oracle Contracts And Baselines

**Files:**
- Create: `tools/make_moonlight_oracle.py`
- Create: `moonlight_oracle.h`
- Create: `moonlight_oracle.c`
- Modify: `floyd.c:620-1420`
- Modify: `.gitignore`
- Test: `tests/test_moonlight_oracle.py`

**Interfaces:**
- Produces: `int moonlight_oracle_enabled(void)` and `int moonlight_oracle_write_f32(const char *name, const float *data, size_t count)`.
- Produces: ignored `fixture_moonlight_metal/oracle.safetensors` and `fixture_moonlight_metal/ref.json` with keys `embed`, `layer.N.attn`, `layer.N.router_scores`, `layer.N.router_ids`, `layer.N.moe`, `layer.N.output`, `final_norm`, `logits`, `prompt_ids`, `greedy_ids`, and two-turn continuations.

- [ ] **Step 1: Add a failing oracle schema test**

```python
REQUIRED = {
    "embed", "layer.0.attn", "layer.1.router_scores",
    "layer.1.router_ids", "layer.1.moe", "layer.1.output",
    "final_norm", "logits",
}
assert REQUIRED <= set(load_file(args.oracle))
assert reference["prompt_ids"]
assert reference["greedy_ids"]
assert len(reference["turns"]) == 2
```

- [ ] **Step 2: Run the schema test and verify RED**

Run: `.venv/bin/python tests/test_moonlight_oracle.py -v`

Expected: FAIL because `tools.make_moonlight_oracle` and the activation fixture do not exist.

- [ ] **Step 3: Implement deterministic tiny and real capture**

Use PyTorch forward hooks on embeddings, each decoder layer, attention output, and MLP output; reproduce router logits/selection explicitly with sigmoid, correction bias, group masking, top-k normalization, and `routed_scaling_factor`. Write numeric arrays to safetensors and token/chat metadata to JSON. The C oracle writer is enabled only by `FLOYD_MOONLIGHT_ORACLE_DIR` and writes atomically through `name.tmp` followed by `rename`.

- [ ] **Step 4: Capture and verify baselines**

Run:

```bash
.venv/bin/python tools/make_moonlight_oracle.py tiny \
  --model fixture_tiny --output fixture_moonlight_metal
.venv/bin/python tools/make_moonlight_oracle.py real \
  --model models/moonlight_i8 --source models/Moonlight-16B-A3B-Instruct \
  --output fixture_moonlight_metal_real --max-tokens 16
make clean && make
./floyd tf --model fixture_tiny --ref ref_tiny.json
./floyd gen --model fixture_tiny --ref ref_tiny.json
```

Expected: schema test PASS; record CPU prefill/decode milliseconds, exact tiny TF/greedy hits, real greedy IDs, and two-turn continuation IDs in the commit message or `docs/parity-report.md`, not in a log file.

- [ ] **Step 5: Commit the oracle contract**

```bash
git add .gitignore tools/make_moonlight_oracle.py moonlight_oracle.h \
  moonlight_oracle.c tests/test_moonlight_oracle.py floyd.c
git commit -m "test: freeze Moonlight runtime oracles"
```

### Task 2: Add Persistent Metal Runtime Ownership

**Files:**
- Create: `moonlight_model.h`
- Create: `moonlight_model.c`
- Create: `moonlight_metal.h`
- Create: `moonlight_metal.m`
- Create: `moonlight_kernels.metal`
- Modify: `Makefile`
- Test: `tests/test_moonlight_metal_runtime.c`

**Interfaces:**
- Produces: opaque `MoonlightModel` and `MoonlightSession`.
- Produces:

```c
typedef struct {
    int context_size;
    int max_batch;
} MoonlightOptions;

typedef struct {
    uint64_t command_buffers;
    uint64_t prefill_tokens;
    uint64_t decode_tokens;
    uint64_t cpu_fallbacks;
    double prefill_ms;
    double decode_ms;
} MoonlightStats;

int moonlight_model_open(MoonlightModel **out, const char *path,
                         char *error, size_t error_size);
void moonlight_model_close(MoonlightModel *model);
int moonlight_session_create(MoonlightSession **out, MoonlightModel *model,
                             const MoonlightOptions *options,
                             char *error, size_t error_size);
void moonlight_session_destroy(MoonlightSession *session);
void moonlight_session_reset(MoonlightSession *session);
const char *moonlight_device_name(const MoonlightModel *model);
MoonlightStats moonlight_session_stats(const MoonlightSession *session);
```

- [ ] **Step 1: Add lifecycle RED tests**

```c
CHECK(moonlight_model_open(&model, fixture, error, sizeof(error)) == 1);
CHECK(strstr(moonlight_device_name(model), "Apple") != NULL);
CHECK(moonlight_session_create(&session, model, &options,
                               error, sizeof(error)) == 1);
CHECK(moonlight_session_stats(session).cpu_fallbacks == 0);
moonlight_session_reset(session);
```

- [ ] **Step 2: Run the lifecycle test and verify RED**

Run: `make METAL=1 tests/test_moonlight_metal_runtime`

Expected: compilation fails because `moonlight_metal.h` is absent.

- [ ] **Step 3: Implement ownership without numerical execution**

Create one `MTLDevice`, command queue, compiled library, and pipeline cache per model. Memory-map/index the checkpoint once, upload model-resident weights once, and allocate session activation/KV buffers once from `context_size` and `max_batch`. Reject invalid dimensions and allocation failures with English errors. Do not call `waitUntilCompleted` inside individual operators; synchronization belongs at public forward boundaries.

- [ ] **Step 4: Run lifecycle and existing Metal tests**

Run:

```bash
make METAL=1 tests/test_moonlight_metal_runtime
./tests/test_moonlight_metal_runtime fixture_tiny fixture_moonlight_metal/oracle.safetensors
make METAL=1 metal-test
```

Expected: lifecycle PASS, device begins with `Apple`, allocation counters stay stable across reset, `cpu_fallbacks=0`, existing q8/q4 Metal parity remains PASS.

- [ ] **Step 5: Commit the runtime skeleton**

```bash
git add moonlight_model.h moonlight_model.c moonlight_metal.h moonlight_metal.m \
  moonlight_kernels.metal tests/test_moonlight_metal_runtime.c Makefile
git commit -m "feat: add persistent Moonlight Metal session"
```

### Task 3: Close Embedding, Norm, Quantized Matmul, And Lm-Head

**Files:**
- Modify: `moonlight_metal.h`
- Modify: `moonlight_metal.m`
- Modify: `moonlight_kernels.metal`
- Test: `tests/test_moonlight_metal_primitives.c`

**Interfaces:**
- Produces test-only entry points under `#ifdef FLOYD_MOONLIGHT_TESTING`:

```c
int moonlight_test_embed(MoonlightSession *, const int *ids, int count, float *out);
int moonlight_test_rmsnorm(MoonlightSession *, const float *x,
                           const float *weight, int rows, int width, float *out);
int moonlight_test_matmul(MoonlightSession *, const char *tensor,
                          const float *x, int rows, float *out);
int moonlight_test_lm_head(MoonlightSession *, const float *x, int rows, float *out);
```

- [ ] **Step 1: Add primitive parity assertions**

```c
CHECK(max_abs(embed, oracle_embed, embed_count) < 3e-6f);
CHECK(max_abs(norm, oracle_norm, norm_count) < 3e-5f);
CHECK(max_abs(q8, oracle_q8, q8_count) < 2e-4f);
CHECK(max_abs(q4, oracle_q4, q4_count) < 2e-3f);
CHECK(argmax(logits, vocab) == oracle_token);
CHECK(moonlight_session_stats(session).cpu_fallbacks == 0);
```

- [ ] **Step 2: Run primitive test and verify RED**

Run: `make METAL=1 tests/test_moonlight_metal_primitives && ./tests/test_moonlight_metal_primitives fixture_tiny fixture_moonlight_metal/oracle.safetensors`

Expected: FAIL at the first missing/unimplemented primitive.

- [ ] **Step 3: Implement GPU primitives**

Add grid-stride embedding lookup, F32-accumulating RMSNorm, residual add, q8/q4 matrix multiplication using persistent weight buffers, SiLU-multiply, and final lm-head. Encode all operators for a stage into one command buffer and copy only explicitly requested test outputs back to shared memory.

- [ ] **Step 4: Verify numerical closure**

Run the primitive test for `fixture_tiny`, `fixture_tiny_i8`, and `models/moonlight_i8`. Expected: reported max-absolute errors remain within the assertions, argmax matches, and `cpu_fallbacks=0` for both prefill-shaped and single-row inputs.

- [ ] **Step 5: Commit primitives**

```bash
git add moonlight_metal.h moonlight_metal.m moonlight_kernels.metal \
  tests/test_moonlight_metal_primitives.c Makefile
git commit -m "feat: close Moonlight Metal primitives"
```

### Task 4: Close MLA, RoPE, Causal Attention, And Compressed KV

**Files:**
- Modify: `moonlight_metal.h`
- Modify: `moonlight_metal.m`
- Modify: `moonlight_kernels.metal`
- Test: `tests/test_moonlight_metal_mla.c`

**Interfaces:**
- Produces: `int moonlight_test_attention(MoonlightSession *, int layer, const float *x, int rows, int position, float *out)`.
- Produces persistent per-layer compressed cache containing normalized `kv_lora` latent values and rotated `qk_rope` keys for positions `[0, context_size)`.

- [ ] **Step 1: Add MLA/KV RED cases**

```c
CHECK(run_attention(session, layer, prompt, prompt_len, 0, prefill));
CHECK(run_attention(session2, layer, prompt, prompt_len - 1, 0, prefix));
CHECK(run_attention(session2, layer, prompt + prompt_len - 1, 1,
                    prompt_len - 1, decode));
CHECK(max_abs(prefill_last, decode, hidden) < 3e-4f);
CHECK(max_abs(prefill, oracle_attn, prompt_len * hidden) < 3e-4f);
CHECK(kv_length(session) == prompt_len);
CHECK(moonlight_session_stats(session).cpu_fallbacks == 0);
```

- [ ] **Step 2: Run MLA test and verify RED**

Run: `make METAL=1 tests/test_moonlight_metal_mla && ./tests/test_moonlight_metal_mla fixture_tiny fixture_moonlight_metal/oracle.safetensors`

Expected: FAIL because attention/KV entry points are not implemented.

- [ ] **Step 3: Implement MLA and cache state**

Implement q/kv low-rank projections, RMSNorm, q expansion, interleaved RoPE using checkpoint theta, compressed KV append, causal score reduction, stable softmax, value accumulation through `kv_b`, and output projection. Prefill may tile keys/queries; decode must use the same cache layout and must not allocate or copy the full cache per token.

- [ ] **Step 4: Verify tiny then real MLA**

Run:

```bash
./tests/test_moonlight_metal_mla fixture_tiny fixture_moonlight_metal/oracle.safetensors
./tests/test_moonlight_metal_mla models/moonlight_i8 fixture_moonlight_metal_real/oracle.safetensors
make METAL=1 metal-test
```

Expected: prefill/decode equivalence and oracle error under `3e-4`, exact causal boundary checks, stable KV length after reset/reuse, zero CPU fallbacks.

- [ ] **Step 5: Commit MLA/KV**

```bash
git add moonlight_metal.h moonlight_metal.m moonlight_kernels.metal \
  tests/test_moonlight_metal_mla.c Makefile
git commit -m "feat: add Moonlight Metal MLA and KV cache"
```

### Task 5: Close Router And MoE Experts

**Files:**
- Modify: `moonlight_metal.h`
- Modify: `moonlight_metal.m`
- Modify: `moonlight_kernels.metal`
- Test: `tests/test_moonlight_metal_moe.c`

**Interfaces:**
- Produces: `int moonlight_test_moe(MoonlightSession *, int layer, const float *x, int rows, int *route_ids, float *route_weights, float *out)`.
- Consumes checkpoint routing fields `n_group`, `topk_group`, `num_experts_per_tok`, `norm_topk_prob`, and `routed_scaling_factor`.

- [ ] **Step 1: Add router/expert RED checks**

```c
CHECK(memcmp(route_ids, oracle_ids, route_count * sizeof(*route_ids)) == 0);
CHECK(max_abs(route_weights, oracle_weights, route_count) < 3e-6f);
CHECK(max_abs(shared_out, oracle_shared, rows * hidden) < 3e-4f);
CHECK(max_abs(moe_out, oracle_moe, rows * hidden) < 8e-4f);
CHECK(moonlight_session_stats(session).cpu_fallbacks == 0);
```

- [ ] **Step 2: Run MoE test and verify RED**

Run: `make METAL=1 tests/test_moonlight_metal_moe && ./tests/test_moonlight_metal_moe fixture_tiny fixture_moonlight_metal/oracle.safetensors`

Expected: FAIL because Metal routing and experts are not implemented.

- [ ] **Step 3: Implement deterministic Metal routing and experts**

Compute router logits in F32, sigmoid scores, group selection, correction-bias top-k IDs with deterministic tie-breaking by expert ID, original-score weights, optional normalization, and routed scaling. Batch tokens by selected expert on the host only as index bookkeeping; execute gate/up/down and shared SwiGLU numerically on Metal. Keep model-resident experts in persistent buffers, and reject a checkpoint that exceeds the Metal allocation budget with an English memory estimate rather than falling back to CPU streaming.

- [ ] **Step 4: Verify tiny and real routing**

Run the MoE test against tiny and `models/moonlight_i8`. Expected: every route ID matches, route/max-abs thresholds pass, no model buffer allocation increases after warmup, and `cpu_fallbacks=0`.

- [ ] **Step 5: Commit MoE**

```bash
git add moonlight_metal.h moonlight_metal.m moonlight_kernels.metal \
  tests/test_moonlight_metal_moe.c Makefile
git commit -m "feat: close Moonlight Metal MoE"
```

### Task 6: Integrate Full Layers, Prefill, And Decode

**Files:**
- Modify: `moonlight_metal.h`
- Modify: `moonlight_metal.m`
- Modify: `moonlight_kernels.metal`
- Test: `tests/test_moonlight_metal_runtime.c`

**Interfaces:**
- Produces:

```c
int moonlight_session_prefill(MoonlightSession *, const int *ids, int count,
                              float *last_logits, char *error, size_t error_size);
int moonlight_session_decode(MoonlightSession *, int token, float *logits,
                             char *error, size_t error_size);
int moonlight_session_position(const MoonlightSession *);
```

- [ ] **Step 1: Add full-runtime RED cases**

```c
CHECK(moonlight_session_prefill(session, prompt, n, logits, error, sizeof(error)));
CHECK(argmax(logits, vocab) == ref_tf[n - 1]);
for (int i = 0; i < 20; ++i) {
    int token = argmax(logits, vocab);
    CHECK(token == ref_greedy[n + i]);
    CHECK(moonlight_session_decode(session, token, logits, error, sizeof(error)));
}
CHECK(moonlight_session_position(session) == n + 20);
CHECK(moonlight_session_stats(session).cpu_fallbacks == 0);
```

- [ ] **Step 2: Run runtime test and verify RED**

Run: `make METAL=1 tests/test_moonlight_metal_runtime && ./tests/test_moonlight_metal_runtime fixture_tiny fixture_moonlight_metal`

Expected: FAIL on the first full-layer/logit mismatch.

- [ ] **Step 3: Encode the complete graph**

Chain embedding, per-layer pre-norm MLA residual, post-norm dense/MoE residual, final norm, and lm-head. Reuse two activation buffers, encode one command buffer per layer group rather than per primitive, use shared-event or command-buffer completion only where expert index bookkeeping requires host visibility, and keep decode buffers at stable addresses.

- [ ] **Step 4: Verify full tiny closure**

Run:

```bash
./tests/test_moonlight_metal_runtime fixture_tiny fixture_moonlight_metal
./tests/test_moonlight_metal_runtime fixture_tiny_i8 fixture_moonlight_metal_i8
make METAL=1 test-c test-tok metal-test
```

Expected: exact 32/32 tiny teacher-forcing predictions, exact 20/20 tiny greedy tokens for the F32 fixture, quantified int8 parity, per-layer max-absolute values reported, and zero fallbacks.

- [ ] **Step 5: Commit the complete tiny runtime**

```bash
git add moonlight_metal.h moonlight_metal.m moonlight_kernels.metal \
  tests/test_moonlight_metal_runtime.c Makefile
git commit -m "feat: complete Moonlight Metal inference graph"
```

### Task 7: Real Checkpoint Chat Parity And Performance Gate

**Files:**
- Create: `tests/test_moonlight_metal_official.sh`
- Modify: `moonlight_metal.m`
- Modify: `Makefile`
- Modify: `docs/parity-report.md`

**Interfaces:**
- Consumes real checkpoint path from `MOONLIGHT` and ignored oracle path from `MOONLIGHT_ORACLE`.
- Produces one parseable stderr line:

```text
MOONLIGHT_METAL prefill_tokens=N prefill_ms=X prefill_tps=X decode_tokens=N decode_ms=X decode_tps=X command_buffers=N cpu_fallbacks=0
```

- [ ] **Step 1: Add official gate with deliberate failure**

The script must compare full greedy token ID streams, compare two-turn continuation IDs without resetting the session, require `cpu_fallbacks=0`, and require both Metal `prefill_tps` and `decode_tps` to exceed the frozen CPU baseline from Task 1.

- [ ] **Step 2: Run the official gate and preserve RED numbers**

Run:

```bash
MOONLIGHT=models/moonlight_i8 \
MOONLIGHT_ORACLE=fixture_moonlight_metal_real \
make METAL=1 test-moonlight-metal-official
```

Expected: initial failure identifies the first real activation, token, or performance gate and prints the measured stage timings.

- [ ] **Step 3: Optimize only the measured bottleneck**

Use Metal counter samples/signposts to distinguish upload, command submission, dense/attention, router synchronization, routed experts, lm-head, and readback. Apply the smallest change that removes the dominant cost: persistent private weights with staged upload, larger fused kernels, expert batching, fewer command-buffer boundaries, or argmax-on-GPU. Do not relax parity thresholds to obtain speed.

- [ ] **Step 4: Prove real parity and a speed win**

Run the official gate three times after one warmup. Expected: identical greedy IDs and two-turn IDs on all runs; zero fallback; median Metal prefill and decode both faster than the recorded CPU baseline. Record hardware, model/container, token counts, max-abs values, and medians concisely in `docs/parity-report.md`.

- [ ] **Step 5: Commit the real-model gate**

```bash
git add tests/test_moonlight_metal_official.sh moonlight_metal.m Makefile docs/parity-report.md
git commit -m "perf: validate Moonlight Metal chat runtime"
```

### Task 8: Switch Product CLI And Documentation

**Files:**
- Modify: `floyd.c`
- Modify: `deepseek_v4_chat.c`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `tests/test_cli_default_chat.sh`
- Modify: `tests/test_deepseek_v4_chat_dispatch.sh`
- Modify: `tests/test_deepseek_v4_serve_cli.sh`

**Interfaces:**
- `floyd chat --model DIR [--ctx N] [--ngen N]`
- `floyd run --model DIR --prompt TEXT [--ctx N] [--ngen N]`
- `floyd serve --stdio --model DIR [--ctx N] [--prefix-cache-mb N]`
- `floyd help`

- [ ] **Step 1: Update CLI tests to the final English contract**

Assert that help lists only `chat`, `run`, `serve`, and `help`; errors include `missing --model`, `unknown option`, and `Metal is required`; chat includes `:reset clears the conversation, :exit quits`; Moonlight startup prints `backend=metal`; removed `tf`, `gen`, `--metal`, and backend-selection flags exit 2.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
make METAL=1 floyd
tests/test_cli_default_chat.sh fixture_tiny
tests/test_deepseek_v4_chat_dispatch.sh fixture_tiny_deepseek_v4
tests/test_deepseek_v4_serve_cli.sh
```

Expected: failures show the old Italian help, legacy commands, and CPU Moonlight dispatch.

- [ ] **Step 3: Cut chat/run/serve to the Metal session**

Replace Moonlight calls to `Model`, `run_chat`, `run_text_moon`, and `run_serve` with `MoonlightModel`/`MoonlightSession`. Keep tokenization and JSONL protocol unchanged. Translate every public message to English and make command-specific validation reject irrelevant flags.

- [ ] **Step 4: Rewrite the README first screen**

Start with prerequisites, `make METAL=1 floyd`, Moonlight preparation/download, built-in chat, DeepSeek V4 preparation/chat, one-shot run, stdio serve, and memory guidance. Explain that model-resident Moonlight weights must fit unified memory; provide a lower-bit converted model command and a smaller `--ctx` example for constrained machines. Keep performance text to the benchmark command and a short expectation statement.

- [ ] **Step 5: Run user-path regressions and commit**

Run:

```bash
make clean && make METAL=1 floyd
make METAL=1 test-c test-tok metal-test
MOONLIGHT=models/moonlight_i8 MOONLIGHT_ORACLE=fixture_moonlight_metal_real \
  make METAL=1 test-moonlight-metal-official
DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark \
  make METAL=1 test-deepseek-v4-ds4-official test-deepseek-v4-serve-official
```

Expected: Moonlight and DeepSeek V4 user paths pass with English output and `backend=metal`.

```bash
git add floyd.c deepseek_v4_chat.c Makefile README.md tests/test_cli_default_chat.sh \
  tests/test_deepseek_v4_chat_dispatch.sh tests/test_deepseek_v4_serve_cli.sh
git commit -m "feat: make Metal chat the Floyd interface"
```

### Task 9: Delete CPU/CUDA Runtime And Enforce Metal-Only Builds

**Files:**
- Delete: `moonlight_oracle.h`
- Delete: `moonlight_oracle.c`
- Delete: `tools/make_moonlight_oracle.py`
- Delete: `tests/test_moonlight_oracle.py`
- Delete: legacy `tf`/`gen` oracle-only tools and tests identified by `rg -l 'floyd (tf|gen)|FLOYD_MOONLIGHT_ORACLE' tools tests`
- Delete: CPU numerical/runtime sections from `floyd.c`
- Delete: unused CUDA source/build integration discovered by `rg -l 'COLI_CUDA|cuda_'`
- Modify: `backend_metal.h`
- Modify: `backend_metal.m`
- Modify: `Makefile`
- Modify: `patches/ds4/` patch list and add one pinned CPU-removal patch
- Create: `tests/test_metal_only_build.sh`
- Modify: `README.md`

**Interfaces:**
- Final `floyd.c` contains CLI parsing, tokenization, command dispatch, and no model numerical kernels.
- Plain `make`, `make floyd`, `METAL=0`, non-Darwin, and non-arm64 builds fail before compilation with a concise English diagnostic.

- [ ] **Step 1: Add a failing source/build audit**

```bash
test "$(uname -s)" = Darwin
make clean
if make floyd >build.out 2>&1; then exit 1; fi
grep -q 'METAL=1 is required' build.out
make METAL=1 floyd
! nm ./floyd | grep -E 'matmul_i4|matmul_q_idot|DS4_BACKEND_CPU|coli_cuda'
! rg -n 'COLI_CUDA|fallback CPU|DS4_BACKEND_CPU|floyd (tf|gen)|--metal' \
  --glob '!docs/superpowers/**' --glob '!patches/ds4/**' .
```

- [ ] **Step 2: Run the audit and verify RED**

Run: `tests/test_metal_only_build.sh`

Expected: FAIL because CPU kernels, CUDA, optional Metal build logic, and legacy commands remain.

- [ ] **Step 3: Remove obsolete product code**

Delete the temporary oracle and all Moonlight CPU matmul, norm, attention, MoE, KV, forward, TF/gen, CUDA, backend-toggle, and fallback code. Remove transient q8/q4 APIs from `backend_metal.*` after all callers use `moonlight_metal.*`. Keep only model preparation tooling needed to create the supported Moonlight container.

- [ ] **Step 4: Remove DS4 CPU backend at the pinned revision**

Add a patch that deletes `DS4_BACKEND_CPU`, CPU session/KV payload allocation, CPU forward/decode dispatch, and CPU comparison entry points without changing the resident Metal API Floyd uses. Recreate `.deps/ds4` from `DS4_REV`, apply the complete patch series, and rebuild to prove the patch does not depend on local residue.

- [ ] **Step 5: Run the complete final gate**

Run:

```bash
tests/test_metal_only_build.sh
make clean && make METAL=1 floyd
make METAL=1 test-c test-tok metal-test
MOONLIGHT=models/moonlight_i8 MOONLIGHT_ORACLE=fixture_moonlight_metal_real \
  make METAL=1 test-moonlight-metal-official
DSPARK=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark \
DS4_MODEL=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark-DS4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
DS4_SUPPORT=/Users/dongxu/floyd/models/DeepSeek-V4-Flash-DSpark-DS4/DeepSeek-V4-Flash-DSpark-DSpark-3Stage-Q4K-imatrix.gguf \
  make METAL=1 test-deepseek-v4-ds4-official test-deepseek-v4-ds4-spec-official \
    test-deepseek-v4-serve-official
git diff --check
git status --short --untracked-files=all
git ls-files | rg '(^models/|^fixture_[^/]+/|\.bin$|\.log$|^AGENTS\.md$|kernels_metal\.h$)'
```

Expected: all tests PASS; the final tracked-file audit prints nothing; status contains no task changes and may contain only the pre-existing untracked `AGENTS.md`.

- [ ] **Step 6: Commit final removal**

```bash
git add floyd.c backend_metal.h backend_metal.m Makefile README.md patches/ds4 \
  tests/test_metal_only_build.sh
git add -u moonlight_oracle.h moonlight_oracle.c tools tests
git diff --cached --name-only | \
  rg '(^models/|^fixture_[^/]+/|\.bin$|\.log$|^AGENTS\.md$|kernels_metal\.h$)' && exit 1 || true
git commit -m "refactor: remove CPU inference runtimes"
```

Do not declare completion until the commit contains no model, fixture, binary, log, generated Metal header, or `AGENTS.md`.
