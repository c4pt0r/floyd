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

## What's *not* here (scope)

Ported deliberately narrow, matching the
[design doc](docs/superpowers/specs/2026-07-10-floyd-moonlight-parity-design.md)'s
non-goals:

- No MTP speculative decoding (Moonlight has no MTP head).
- No DSA sparse attention (Moonlight has no indexer weights).
- No CUDA, no HTTP server.
- Metal accelerates the **batch** matmul path only (prefill / teacher-forcing,
  sequence length ≥ 8 by default); single-token decode stays on CPU.

The design doc above's own non-goals *did* originally include "no C-side
tokenizer or chat loop" (`tok.h`/`tok_unicode.h` were carried over from
colibrì unused, and Python did tokenization for the oracle only). A later
chat phase closed that gap: `tok_moon.h` is a from-scratch C implementation
of the moonshot tokenizer (raw-byte-rank BPE + moonshot pre-tokenizer,
independently gated at 52/52 cases vs the `tiktoken` oracle — see
[`docs/chat-report.md`](docs/chat-report.md)), and `./floyd chat`/`./floyd run`
now do the whole prompt → tokenize → generate → detokenize loop natively,
with no Python in the loop. See the Quickstart and CLI reference below.

## Quick start

`floyd` has a CLI: `chat` (interactive multi-turn), `run` (single-shot),
`tf`/`gen` (teacher-forcing / greedy parity checks vs a Python oracle). Every
command needs `--model DIR` pointing at a converted container or a raw HF
checkpoint directory.

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
./floyd tf  --model fixture_tiny --ref ref_tiny.json --cap 8 --ebits 16 --dbits 16   # teacher-forcing: expect 32/32
./floyd gen --model fixture_tiny --ref ref_tiny.json --cap 8 --ebits 16 --dbits 16   # greedy: expect 20/20
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

# parity check against the oracle
./floyd tf --model models/moonlight_i8 --ref ref_moonlight.json

# interactive multi-turn chat (moonshot chat_template, KV-cache persists across turns
# and across process restarts unless --no-kvsave)
./floyd chat --model models/moonlight_i8

# single-shot generation
./floyd run --model models/moonlight_i8 --prompt "What is the capital of France? Answer in one word."
```

The unquantized HF checkpoint directory also works directly as `--model` (the
engine falls back to reading raw safetensors when no `.qs` container is
present) — useful as an f32 control run isolated from quantization noise.

### 4. Metal backend (optional, Apple Silicon only)

```bash
PATH=/opt/homebrew/bin:$PATH make METAL=1 clean
PATH=/opt/homebrew/bin:$PATH make METAL=1
make metal-test                       # kernel-level unit test vs CPU f64 reference
./floyd tf --model models/moonlight_i8 --ref ref_moonlight.json --metal
```

The `PATH=/opt/homebrew/bin:$PATH` prefix matters: the Makefile locates
`libomp` via `$(shell brew --prefix libomp)` (see `Makefile`), and if `brew`
itself isn't on `PATH` (e.g. a minimal shell/CI environment) that `shell`
call silently returns empty — no error, just a `libomp non trovato` build
warning and a single-thread binary. Put Homebrew's `bin` on `PATH` before
building (`METAL=1` or not) to get the multi-threaded OpenMP build.

`--metal` (or legacy `FLOYD_METAL=1`) on a CPU-only build (`make` without
`METAL=1`) hard-errors (exit code 2) rather than silently ignoring the flag.

#### Metal is for batch, not chat decode

Metal accelerates the **batch** matmul path (prefill / teacher-forcing /
forward over `S >= FM_MIN_S` tokens at once, default `FM_MIN_S=8`) — chat
decode (`S=1`, one token at a time) stays on CPU by design and default, and
should. The generic Metal dispatch wrapper this project ships (transient
`x`/`y` buffers per call, `waitUntilCompleted` synchronous per-call) pays a
fixed per-dispatch overhead that batch shapes amortize and single-token
decode cannot. Measured on an Apple M3 Ultra: forcing `FM_MIN_S=1` (so
decode also routes to the GPU) collapsed chat throughput to **1.66 tok/s**,
against **5.93 tok/s on CPU** for the same run — a ~3.6x regression, not a
rounding difference. With the default `FM_MIN_S=8` (decode stays CPU),
long-generation throughput was ~7.16 tok/s vs 6.89 tok/s CPU-only: parity,
because Metal simply isn't in the decode path.

**Do not lower `FM_MIN_S` for chat.** The engine now warns loudly on stderr
at startup if you do (`FM_MIN_S<8` with `--metal`/`FLOYD_METAL=1`); this is
an experimental, currently-slower code path, not a hidden win waiting to be
unlocked. `FM_MIN_S` remains useful for tuning where the batch/prefill
threshold kicks in on shapes other than chat decode.

## CLI reference

```
$ ./floyd help
floyd — Moonlight-16B-A3B in pure C
uso: floyd <comando> [flags] | (legacy) SNAP=<dir> floyd <cap> <ebits> <dbits>

comandi:
  chat   conversazione interattiva     floyd chat --model DIR
  run    generazione singola           floyd run  --model DIR --prompt "..."
  tf     teacher-forcing vs oracolo    floyd tf   --model DIR --ref ref.json
  gen    greedy vs oracolo             floyd gen  --model DIR --ref ref.json
  help   questo testo

flags globali:  --model DIR (obbligatorio) | --cap N (64) | --ebits 4|8|16 (8)
  --dbits 4|8|16 (8) | --ram GB | --metal
chat/run:  --ngen N (chat:512, run:256) | --ctx N (4096) | --temp T (0.7)
  --top-p P (0.90) | --system "..." | --prompt "..." (solo run) | --draft N
chat:      --no-kvsave (disattiva persistenza KV su disco; solo chat, run non persiste mai)
tf/gen:    --ref FILE (obbligatorio)
flag duplicati: l'ultima occorrenza vince

variabili d'ambiente: interfaccia legacy/debug (IDOT, DSA, MTP, PILOT, STATS, ...)
```

(Italian in the tool's own output; English gloss: `chat` = interactive
conversation, `run` = single generation, `tf` = teacher-forcing vs oracle,
`gen` = greedy vs oracle. `--cap` is the per-layer expert LRU cache size,
`--ebits`/`--dbits` are expert/dense quantization bit-widths, `--ref` is the
oracle reference JSON, required by `tf`/`gen` and rejected (exit 2) for
`chat`/`run`, `--draft` is the n-gram speculative-decoding draft window,
`--prompt` is likewise rejected (exit 2) for anything other than `run`.
`--no-kvsave` disables the on-disk KV persistence `chat` does by default —
`run` is single-shot and never persists KV, with or without the flag. If a
flag is repeated, the last occurrence wins.)

`chat` starts a REPL (`:reset` clears context, `:exit` quits) that keeps its
KV-cache warm across turns **and** across process restarts — `<model-dir>/.coli_kv`
is loaded on startup and appended to after every turn, so re-running
`./floyd chat --model DIR` on the same conversation resumes instantly with no
re-prefill (see `docs/chat-report.md` for measured evidence: a 61-token
conversation resumed in 0.0s). Pass `--no-kvsave` to keep everything
in-memory only.

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
| Moonlight tokenizer (`tok_moon.h` vs `tiktoken`/oracle) | — | **52/52** cases · **42/42** chat-template ids | — | — |
| Moonlight chat E2E (`./floyd chat`, int8, real weights) | int8 | — | 2-turn memory check: correct "Paris" + correct recall of prior question | 6.7-7.4 tok/s |

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

## Legacy & debug environment variables

The original interface — before the `chat`/`run`/`tf`/`gen`/`help` CLI
existed — was entirely environment-variable driven:
`SNAP=<dir> [TF=1] [REF=<ref>.json] ./floyd <expert-cache> <expert-bits> <dense-bits>`.
The CLI is a thin adapter in front of this: every `--flag` just calls
`setenv()` before falling through to the same unmodified engine code. **Both
interfaces work and are tested** (`docs/parity-report.md` uses the legacy
env-var form throughout and was re-verified against the current binary as
part of this task — see `docs/chat-report.md`). The env-var form remains
useful as a compatibility/developer interface: scripting one-off debug
knobs without adding a CLI flag for each of them, and it's what
`docs/parity-report.md`'s existing logs document verbatim.

```bash
# legacy positional form, still works exactly as documented in docs/parity-report.md:
SNAP=fixture_tiny_i8 TF=1 REF=ref_tiny.json ./floyd 8 8 8
SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
FLOYD_METAL=1 SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
```

Below, the env vars split into two groups: a handful that are just the
legacy name for a documented CLI flag (`cli_adapt()` sets them via
`setenv()` under the hood), and the rest — debug/tuning knobs with **no**
CLI equivalent by design, for developing/profiling the engine rather than
normal use.

### Compat (CLI equivalent exists)

| Variable | Purpose | Default |
|---|---|---|
| `KVSAVE` | on-disk KV persistence; legacy name for `--no-kvsave` (`0` disables) | 1 |
| `NUCLEUS` | token-sampling nucleus (top-p) on the vocabulary logits; legacy name for `--top-p` | 0.90 |

### Developer/debug (env-only, no CLI flag)

| Variable | Purpose | Default |
|---|---|---|
| `IDOT` | int8 activation-quantization matmul kernel (0 = exact f32 activations) | 1 |
| `DSA` | DSA sparse attention (only if the checkpoint has indexer weights) | auto |
| `MTP` | multi-token-prediction speculative decoding (only if the checkpoint has an MTP head) | auto |
| `PILOT` | router-piloted expert prefetch | 0 |
| `PILOT_K` | pilot prefetch fan-out | 8 |
| `STATS` | `STATS=<file>` writes an expert-usage histogram at end of run | unset |
| `NOPACK` | disable weight packing | 0 |
| `DROP` | drop-on-evict cache policy variant (keep evicted expert pages in the OS page cache as free L2, vs. discarding them) | 0 |
| `PREFETCH` | re-enable cross-layer `WILLNEED` madvise prefetch (boolean; off by default because parallel real loads made it redundant, and under memory pressure the speculative readahead was getting re-evicted) | 0 |
| `TOPK` | MoE **expert-routing** knob: force `n` experts/token (clamped to the checkpoint's top-k — `n` above it has no further effect). Independent of `TOPP`; no CLI equivalent | 0 (off = use checkpoint's config top-k) |
| `TOPP` | MoE **expert-routing** knob: adaptive top-p — after ranking the routed experts by router weight, keep only as many as needed for cumulative weight to reach `p` (0..1) of the total, i.e. the disk-read reducer benchmarked by colibrì. Independent of `TOPK`, no CLI equivalent | 0 (off) |
| `MLOCK` | force/disable `mlock()` of resident weights | auto (on, macOS) |
| `SPEC` | n-gram speculative decoding | 1 |
| `REPIN` | re-pin hot experts every N emitted tokens | 0 (off) |
| `ABSORB` | MLA weight-absorption toggle | auto |
| `CHAT_TEMPLATE` | apply chat template in `run_serve`'s legacy protocol mode | 1 |
| `THINK` | emit `<think>` reasoning block instead of `<think></think>` (nothink) | 0 |
| `PIN` | `PIN=<statsfile>` pins the top experts by usage frequency permanently in RAM (evaluated before the RAM cap, so pinned experts count against the resident budget) | unset (off) |
| `PIN_GB` | RAM budget in GB for `PIN`'s pinned experts | 10.0 |
| `AUTOPIN` | auto-pin the historically most-used experts (from `<model-dir>/.coli_usage`) once usage history is confident enough (>=5000 recorded selections); `AUTOPIN=0` disables | 1 (on) |
| `CAP_RAISE` | auto-raise the per-layer expert LRU cache size when the RAM budget allows more than `--cap` requested, up to `n_experts` (fixes large-RAM machines running with an undersized cache); `CAP_RAISE=0` restores the old fixed-cap behavior | 1 (on) |
| `SERVE` | `SERVE=1` starts the persistent serve mode used by the `coli` CLI / HTTP gateway (model stays resident, answers requests) instead of the one-shot batch/validation paths | unset (off) |
| `SCORE` | `SCORE=<requests.txt>` runs benchmark scoring mode: log-likelihood per line instead of generation | unset |
| `REPLAY` | fixed-token decode benchmark: prefills all but the prompt's last oracle token, then replays the oracle continuation one token at a time, so CPU and CUDA see identical hidden-state inputs for a controlled perf comparison | unset |
| `DIRECT` | `DIRECT=1` uses `O_DIRECT` for expert weight slab reads instead of buffered I/O; measured worse on this project's dev host (VHDX on NVMe, DRAM-less) but expected better on real NVMe | 0 (buffered) |
| `LOOKA` | `LOOKA=1` measures (counters only, zero behavioral effect) how predictable MoE routing is ahead of time, to evaluate whether a router-piloted prefetch could hide disk latency | 0 (off) |
| `SEED` | fixes the sampling RNG seed for deterministic generation | unset (seeded from clock + pid) |
| `DSA_FORCE` | `DSA_FORCE=1` forces the DSA lightning-indexer selection to always run (test knob: makes the sparse top-min(k,T) selection behave like dense attention) | 0 (off) |
| `DSA_TOPK` | overrides the DSA indexer's `index_topk` config value (test/dev override) | unset (use checkpoint config) |
| `MTP_DEBUG` | `>=2` prints per-step MTP draft debug info (pre/post-block predictions, hidden-state norms); any truthy value also logs a draft-vs-verified-token hit/miss line during decode | unset (off) |
| `MTP_PRENORM` | RFC/experiment toggle for the MTP head's hidden-state normalization order (whether `h` skips the post-`model.norm` rmsnorm before its own `hnorm`) | unset (off — uses the vLLM-matching post-norm path) |
| `MTP_SWAP` | swaps the concatenation order of the MTP head's two input halves (embedding vs. hidden state) fed into `eh_proj` | unset (off — embedding first, hidden state second) |
| `FM_MIN_S` | Metal backend only (`make METAL=1`): minimum batch size `S` before a matmul is dispatched to the GPU; smaller batches run on CPU | 8 |
| `OMP_WAIT_POLICY` | not floyd-specific: set to `passive` automatically at startup (if not already set by the caller), so idle OpenMP threads sleep instead of spinning while the engine waits on disk I/O | `passive` (auto-set) |

See `usage()`/`floyd help` and the CLI reference above for the supported
flags most users need; the developer/debug table is for people who need the
lower-level knobs `tf`/`gen`/`chat`/`run` don't expose. `PIN` and
`CAP_RAISE` are the two most likely to matter in practice — see the
transcripts in `docs/chat-report.md`.

## Known limitations

- `tf`/`gen` (teacher-forcing / greedy parity checks) still consume token IDs
  via `ref.json`, produced by the Python oracle tooling — that's a parity
  fixture format, not a limitation of the engine's own tokenizer, which
  `chat`/`run` use directly (see CLI reference above).
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
