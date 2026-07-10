# floyd

**Same tiny engine, an even bigger model.** `floyd` runs **Moonlight-16B-A3B**
(moonshotai, `deepseek_v3` architecture — MLA attention, 64-expert
sigmoid/`noaux_tc` MoE router, 2 shared experts) in pure C, with the same
methodology as [colibrì](https://github.com/JustVugg/colibri): offline
BF16 → int8/int4 conversion, streamed experts with an LRU cache, and
token-exact parity validation against a `transformers` oracle. An optional
Metal backend adds Apple Silicon GPU acceleration on the batch (prefill/TF)
matmul path.

*Stesso motore, un modello diverso.*

## Relationship to colibrì

`floyd.c` is a fork of colibrì's `c/glm.c` — the same engine (streaming
expert LRU, MLA compressed KV-cache with weight absorption, int8/int4/int2
quantization kernels, batch-union MoE, RAM auto-budgeting), retargeted from
GLM-5.2 to Moonlight-16B-A3B. All credit for the original engine design and
methodology goes to colibrì: **https://github.com/JustVugg/colibri**.

The two architectures (both `deepseek_v3`-family) are close enough that the
adaptation is small and converges on two points:

1. **No q-LoRA.** Moonlight's `config.json` has `q_lora_rank: null`, so
   attention reads `self_attn.q_proj.weight` directly (`q = q_proj(x)`)
   instead of the `q_a_proj` → `q_a_layernorm` → `q_b_proj` low-rank path
   `glm.c` uses unconditionally for GLM-5.2.
2. **No MTP, no DSA.** Moonlight has `num_nextn_predict_layers=0` (no
   multi-token-prediction head) and no DSA-indexer weights. `glm.c`'s
   "enable only if the weights exist" detection already handles this
   cleanly — floyd just never sees those tensors.

Everything else — the router math (sigmoid scoring, `noaux_tc` top-k,
`norm_topk_prob`, `routed_scaling_factor`), the MLA math, the quantized
matmul kernels — carried over unchanged, because Moonlight's `config.json`
matches colibrì's `glm_moe_dsa` runtime parameterization exactly
(`n_group=1`, `topk_group=1`).

## MoE runtime evolution

`moe_route.h` is the first model-independent runtime component. It implements
softmax, sigmoid, and square-root-softplus expert affinity while keeping
selection bias separate from mixture weights. Moonlight now uses this shared
path without changing its routing results. Expert execution, attention, KV
cache, and weight loading remain architecture-specific; DeepSeek V4 hash-MoE,
mHC, CSA/HCA attention, FP4/FP8 storage, and DSpark decoding are not yet
implemented.

Generate the deterministic CPU-sized V4 architecture oracle with:

```bash
.venv/bin/python tools/make_v4_oracle.py tiny
```

This creates ignored `fixture_tiny_v4/` and `ref_v4_tiny.json` artifacts. The
fixture covers sliding/HCA/CSA attention, mHC, hash MoE, and learned
sqrt-softplus routing; it is a correctness target, not a performance model.

## What's *not* here (scope)

Ported deliberately narrow, matching the
[design doc](docs/superpowers/specs/2026-07-10-floyd-moonlight-parity-design.md)'s
non-goals:

- No C-side tokenizer or chat loop (`tok.h`/`tok_unicode.h` are carried over
  from colibrì but unused at runtime here — Python does tokenization for the
  oracle only, same boundary as colibrì).
- No MTP speculative decoding (Moonlight has no MTP head).
- No DSA sparse attention (Moonlight has no indexer weights).
- No CUDA, no HTTP server.
- Metal accelerates the **batch** matmul path only (prefill / teacher-forcing,
  sequence length ≥ 8 by default); single-token decode stays on CPU.

## Quick start

### 1. Python venv (conversion + oracle generation only — the engine itself has zero runtime dependencies)

```bash
cd floyd
python3 -m venv .venv
.venv/bin/pip install -q torch transformers safetensors numpy accelerate tiktoken blobfile
```

### 2. Tiny fixture — exact parity gate (seconds, no download)

```bash
.venv/bin/python tools/make_oracle.py tiny          # -> fixture_tiny/ + ref_tiny.json
make
SNAP=fixture_tiny TF=1 REF=ref_tiny.json ./floyd 8 16 16   # teacher-forcing: expect 32/32
SNAP=fixture_tiny REF=ref_tiny.json ./floyd 8 16 16        # greedy: expect 20/20
```

### 3. Real model (Moonlight-16B-A3B-Instruct, ~32 GB BF16 download)

```bash
.venv/bin/pip install -q -U "huggingface_hub[cli]"
.venv/bin/hf download moonshotai/Moonlight-16B-A3B-Instruct \
  --local-dir models/Moonlight-16B-A3B-Instruct

# oracle (CPU f32, ~64 GB RAM, can take tens of minutes for a 16B model)
.venv/bin/python tools/make_oracle.py real \
  --model models/Moonlight-16B-A3B-Instruct --ref ref_moonlight.json --ngen 20

# convert to int8 (and/or int4) containers
.venv/bin/python tools/convert_moonlight.py \
  --indir models/Moonlight-16B-A3B-Instruct --outdir models/moonlight_i8 --ebits 8 --dbits 8

# engine run: SNAP=<container> [TF=1] REF=<ref>.json ./floyd <expert-cache> <expert-bits> <dense-bits>
SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
```

The unquantized HF checkpoint directory also works directly as `SNAP=` (the
engine falls back to reading raw safetensors when no `.qs` container is
present) — useful as an f32 control run isolated from quantization noise.

### 4. Metal backend (optional, Apple Silicon only)

```bash
make clean && make METAL=1
make metal-test                       # kernel-level unit test vs CPU f64 reference
FLOYD_METAL=1 SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
```

`FLOYD_METAL=1` on a CPU-only build (`make` without `METAL=1`) hard-errors
(exit code 2) rather than silently ignoring the flag.

## Results

All numbers below are measured, not estimated — see
[`docs/parity-report.md`](docs/parity-report.md) for the full logs and
per-run detail. Machine: Apple M3 Ultra, 32 cores, 512 GB RAM, macOS 15.5.

| Checkpoint | Container | TF (teacher-forcing) | greedy | pos/s (TF) |
|---|---|---|---|---|
| tiny fixture (synthetic, real architecture) | f32 | **32/32** | **20/20** | ~250-266 |
| Moonlight-16B-A3B-Instruct | f32 (control, unquantized) | **52/52** · 30/30 (secondary ref) | — | 5.3 |
| Moonlight-16B-A3B-Instruct | int8 | 48/52 · 29/30 | 16/24* | 13.2 |
| Moonlight-16B-A3B-Instruct | int4 | 46/52 · 24/30 | 16/24* | 18.1 |

\* Both int8 and int4 greedy runs match the oracle for the first 16 of 24
tokens, then emit one spurious extra token at position 17; int8's tail then
re-syncs with the oracle's sequence one slot later (undercounted by the
engine's strict per-index scoring), while int4 diverges further from that
point. See `docs/parity-report.md` for the full token dumps.

The f32 control run is the load-bearing result: **52/52 and 30/30 argmax-exact
against the `transformers` f32 oracle on the real 16B checkpoint** — proof
that every int8/int4 TF gap above is quantization noise, not an engine bug.
A meaningful share of that noise is the CPU engine's own int8
activation-quantization matmul kernel (`IDOT`, on by default): on the tiny
int8 fixture, `IDOT=0` (exact f32 activations) alone raises TF from 24/32 to
29/32. See `docs/parity-report.md` for the full IDOT isolation.

### Metal A/B (`FLOYD_METAL=1` vs CPU-only, same container, same ref)

| Ref | CPU | Metal | Note |
|---|---|---|---|
| fixture (S=32, int8) | 24/32 | 27/32 | 7 positions flip (5 MISS→OK, 2 OK→MISS); all near-tie argmax (logit margin 0.004-0.19) |
| Moonlight int8, long ref (S=52) | 48/52 | 51/52 | 3 positions flip, all MISS→OK; margins 0.03-0.25 on a ~14-18 logit scale |
| Moonlight int8, short ref (S=30) | 29/30 | 29/30 | exact match, 0 flips |

Every flip was verified to sit on a narrow top1-vs-top2 logit margin. The
CPU side of these A/Bs still runs the int8 activation-quantization kernel
(`IDOT`, default on), while Metal consumes f32 activations directly — so the
CPU-vs-Metal delta here is dominated by that CPU-side activation
quantization, not by GPU-vs-CPU floating-point reduction-order
non-associativity (bounded at ~2e-5 max relative error by the
independently-validated kernel-level test, see `make metal-test`). Metal
landing systematically closer to the f32 oracle (27/32, 51/52 vs CPU's
24/32, 48/52) follows from that: it simply skips a lossy step the CPU path
takes. Not a kernel correctness bug either way.

**Honest performance note: Metal is currently 15-25% slower than CPU** on
these shapes (13.2 vs 17.3 pos/s on the long ref, 211 vs sub-260 pos/s on the
fixture). Root cause: activation buffers are transient per-call
(`newBufferWithBytes` + `waitUntilCompleted`, no dispatch batching across a
layer's many small matmuls), and streamed-expert weight buffers can never be
cached across calls (the same host pointer is recycled for a different
expert's bytes on every LRU miss, so caching by pointer would serve stale
weights) — so every Metal-eligible expert matmul re-uploads its weights from
scratch. No performance tuning was attempted; this is reported as measured,
per this project's culture of honest numbers. See `docs/parity-report.md`
for the full breakdown.

## Known limitations

- No C-side tokenizer/chat interaction loop — engine consumes token IDs via
  `ref.json`, produced by the Python oracle tooling.
- No MTP speculative decoding (Moonlight has no MTP head — `num_nextn_predict_layers=0`).
- No DSA sparse attention (Moonlight has no indexer weights).
- Metal backend accelerates only the batch matmul path (prefill / TF,
  `S >= 8` by default via `FM_MIN_S`); single-token decode runs on CPU
  regardless of `FLOYD_METAL`.
- Metal is measured slower than CPU on the shapes tested here — see above.
  It exists as a correctness-validated, honestly-reported first cut, not a
  performance win yet.
- Real-model oracle uses `transformers`' **native** `DeepseekV3` loader, not
  Moonlight's bundled remote code (`modeling_deepseek.py`), which fails to
  import against transformers 5.13 (a documented, expected fallback — not a
  bug — see `docs/parity-report.md`).
