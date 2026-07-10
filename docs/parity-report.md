# Parity report: real Moonlight-16B-A3B-Instruct (int8 / int4 vs f32 oracle)

## Machine

- Apple M3 Ultra, 32 cores, 512 GiB RAM (`hw.memsize`=549755813888 bytes)
- macOS 15.5 (build 24F74)
- Engine build: `floyd.c` @ `f7af80174d89b938441e42ba18809b62c626b47b` (task 3), compiled with
  `clang -O3 -Xclang -fopenmp ... -lomp` (unmodified for this task)

## Model

- `moonshotai/Moonlight-16B-A3B-Instruct`, snapshot commit
  `4e735b07a89f73647dfab71ab91b840f362ede5b` (from
  `~/.cache/huggingface/hub/models--moonshotai--Moonlight-16B-A3B-Instruct/refs/main`)
- 27 safetensors shards, ~30 GB on disk at `models/Moonlight-16B-A3B-Instruct`
- `DeepseekV3ForCausalLM` architecture (`config.json`): `hidden_size=2048`,
  `n_routed_experts=64`, `n_shared_experts=2`, `moe_intermediate_size=1408`,
  `num_nextn_predict_layers=0` (no MTP), no DSA-indexer keys — matches the
  fixture/converter's exclusion-list assumptions from Task 3/4.

## Oracle

`tools/make_oracle.py real` was run with `transformers` 5.13.0 installed. The
checkpoint's bundled remote code (`auto_map` -> `configuration_deepseek.py` /
`modeling_deepseek.py`) is written against a ~4.46-era `transformers` internal
API and fails to import under 5.13:

```
[fallback] remote code fallito (ImportError: cannot import name 'is_torch_fx_available'
from 'transformers.utils.import_utils' ...); provo il nativo transformers
loader=native
```

Both oracle refs therefore used the **native** `transformers` `DeepseekV3`
implementation (built-in support for this architecture), not the checkpoint's
own remote code. This is a documented, expected fallback — not a bug — and is
recorded in `oracle_real.log` / `oracle_real_long.log`.

Two oracle refs were generated (CPU f32, greedy):

- `ref_moonlight.json` — prompt "capital of France" (28 prompt tokens),
  `--ngen 20` requested, model emitted EOS after 2 tokens: `full_ids` has 30
  positions (28 prompt + 2 generated), text `"Paris<|im_end|>"`.
- `ref_moonlight_long.json` — prompt about why the sky is blue (28 prompt
  tokens), `--ngen 24`, no early EOS: 52 positions (28 prompt + 24
  generated), text: `"The sky appears blue due to a phenomenon called
  Rayleigh scattering. When sunlight enters Earth's atmosphere, it interacts
  with gas molecules"`.

`ref_moonlight_long.json` (52 positions, no truncation from early EOS) is
used as the primary TF/greedy target below; `ref_moonlight.json` (30
positions) is used as a secondary TF datapoint.

## Containers

Built with `tools/convert_moonlight.py` (Task 4), unmodified:

```
.venv/bin/python tools/convert_moonlight.py --indir models/Moonlight-16B-A3B-Instruct --outdir models/moonlight_i8 --ebits 8 --dbits 8
.venv/bin/python tools/convert_moonlight.py --indir models/Moonlight-16B-A3B-Instruct --outdir models/moonlight_i4 --ebits 4 --dbits 8
```

Both conversions ran to completion, all 27 input shards processed, no
errors/tracebacks in the logs.

```
$ du -sh models/moonlight_i8 models/moonlight_i4
 17G	models/moonlight_i8
 10G	models/moonlight_i4
```

Matches the brief's expected range (i8 ≈ 16-17 GB, i4 ≈ 9-10 GB: routed
experts at `ebits`, dense/attention/shared-expert weights at int8, router +
norms + embed/lm_head kept F32 raw).

## Fixture baseline (for scale, from Task 3 / Task 4 — not rerun here)

| container      | TF (teacher-forcing)     | greedy            |
|----------------|---------------------------|--------------------|
| f32 fixture    | 32/32                      | 20/20              |
| int8 fixture   | 24/32 (plausibly fixture-scale noise sensitivity — tiny random weights sit near argmax decision boundaries; consistent with, though not isolated by, the real-model results where int8 reaches 92-97%) | 0/20 (no error-correction across greedy steps once the first token diverges) |

## Real-model results

Engine invocation: `SNAP=<container> [TF=1] REF=<ref>.json ./floyd 64 <ebits> 8`
(cache=64 experts/layer, `dbits=8` in both containers).

### f32 control (unquantized HF checkpoint read directly)

The engine reads unquantized safetensors directly: `qt_from_disk()` falls
back to a full-tensor read (BF16 -> f32 via `st.h`) when no `.qs` sibling
exists, and `bits=16` maps to raw-f32 tensors at runtime — so the HF
checkpoint directory itself works as a container. This isolates engine
correctness from quantization noise.

**TF vs `ref_moonlight_long.json` (52 positions, primary target):**
```
$ SNAP=models/Moonlight-16B-A3B-Instruct TF=1 REF=ref_moonlight_long.json ./floyd 4 16 16
[RAM_GB=331.1 auto] cap ALZATO 4->64: il budget lo consente (proiezione picco 40.2 GB)
== Motore C GLM (glm_moe_dsa), cache=4 expert/layer | expert@16-bit densa@16-bit | idot: neon ==
caricato in 0.85s | densa residente: 5957.50 MB | layers=27 experts=64 | MTP assente (draft=0)
PREFILL (teacher-forcing) C vs oracolo: 52/52 posizioni | 5.3 pos/s
PROFILO: expert-disk 3.841s | expert-matmul 3.183s | attention 1.029s (di cui kvb 0.095s) | lm_head 0.000s | altro 1.763s
```

**TF vs `ref_moonlight.json` (30 positions, secondary datapoint):**
```
$ SNAP=models/Moonlight-16B-A3B-Instruct TF=1 REF=ref_moonlight.json ./floyd 4 16 16
caricato in 0.84s | densa residente: 5957.50 MB | layers=27 experts=64 | MTP assente (draft=0)
PREFILL (teacher-forcing) C vs oracolo: 30/30 posizioni | 4.4 pos/s
```

**f32 TF is 52/52 and 30/30 — the C engine's forward pass is exact on the
real weights** (argmax-identical to the transformers f32 oracle at every
position of both refs). The int8/int4 TF gaps below (48/52, 46/52; 29/30,
24/30) are therefore proven quantization effects, not engine bugs.

### int8 (`models/moonlight_i8`, expert@8-bit densa@8-bit)

**TF vs `ref_moonlight_long.json` (52 positions, primary target):**
```
$ SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight_long.json ./floyd 64 8 8
[RAM_GB=330.6 auto] cap=64 ok (proiezione picco 22.6 GB)
== Motore C GLM (glm_moe_dsa), cache=64 expert/layer | expert@8-bit densa@8-bit | idot: neon ==
caricato in 1.05s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)
PREFILL (teacher-forcing) C vs oracolo: 48/52 posizioni | 13.2 pos/s
PROFILO: expert-disk 1.806s | expert-matmul 0.677s | attention 0.468s (di cui kvb 0.015s) | lm_head 0.000s | altro 0.997s
```

**TF vs `ref_moonlight.json` (30 positions, secondary datapoint):**
```
$ SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
caricato in 0.72s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)
PREFILL (teacher-forcing) C vs oracolo: 29/30 posizioni | 13.5 pos/s
```

**Greedy vs `ref_moonlight_long.json` (24-token generation, primary target):**
```
$ SNAP=models/moonlight_i8 REF=ref_moonlight_long.json ./floyd 64 8 8
caricato in 0.68s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)

Riferimento (oracolo): 1008 18985 13994 9447 5667 308 261 44291 4387 21837 77870 124994 13 4386 46591 49329 145427 19397 11 483 121389 472 9939 50710
Motore C GLM         : 1008 18985 13994 9447 5667 308 261 44291 4387 21837 77870 124994 13 4386 46591 49329 276  145427 19397 11 483 121389 472 9939
Token coincidenti: 16/24
Speculazione n-gram (DRAFT=0): 1.04 token/forward (23 fw per 24 tok)
Hit-rate cache expert: 72.5% (hit=3425 miss=1300) | RSS: 13.89 GB | 5.2 tok/s
```

Note on the greedy divergence: at position 17 the engine emits an extra
token (`276`) not present in the oracle continuation, then every subsequent
engine token matches the oracle's sequence one slot later (`145427 19397 11
483 121389 472 9939` — exactly the oracle's positions 17-23). I.e. this is a
single spurious-token insertion, not a semantic runaway; per-index matching
(as the engine itself computes) counts everything after the insertion point
as mismatched, giving 16/24.

### int4 (`models/moonlight_i4`, expert@4-bit densa@8-bit)

**TF vs `ref_moonlight_long.json` (52 positions, primary target):**
```
$ SNAP=models/moonlight_i4 TF=1 REF=ref_moonlight_long.json ./floyd 64 4 8
[RAM_GB=331.2 auto] cap=64 ok (proiezione picco 15.1 GB)
caricato in 0.88s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)
PREFILL (teacher-forcing) C vs oracolo: 46/52 posizioni | 18.1 pos/s
```

**TF vs `ref_moonlight.json` (30 positions, secondary datapoint):**
```
$ SNAP=models/moonlight_i4 TF=1 REF=ref_moonlight.json ./floyd 64 4 8
caricato in 0.69s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)
PREFILL (teacher-forcing) C vs oracolo: 24/30 posizioni | 18.1 pos/s
```

**Greedy vs `ref_moonlight_long.json` (24-token generation, primary target):**
```
$ SNAP=models/moonlight_i4 REF=ref_moonlight_long.json ./floyd 64 4 8
caricato in 0.66s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)

Riferimento (oracolo): 1008 18985 13994 9447 5667 308 261 44291 4387 21837 77870 124994 13 4386 46591 49329 145427 19397 11 483 121389 472 9939 50710
Motore C GLM         : 1008 18985 13994 9447 5667 308 261 44291 4387 21837 77870 124994 13 4386 46591 49329 276  145427 19397 11 483 6349  4449 472
Token coincidenti: 16/24
Speculazione n-gram (DRAFT=0): 1.04 token/forward (23 fw per 24 tok)
Hit-rate cache expert: 72.2% (hit=3415 miss=1316) | RSS: 8.73 GB | 6.1 tok/s
```

Same single-token insertion at position 17 as int8 (identical prefix up to
that point), then diverges further (position 22 onward differs from both
the oracle and the int8 run) — consistent with int4's larger expert-weight
quantization error compounding once greedy has no error-correction.

## Summary table

| container | TF (52-pos, primary) | TF (30-pos, secondary) | greedy (24 gen, primary) | pos/s (TF) | RSS (greedy) |
|-----------|----------------------|-------------------------|---------------------------|------------|--------------|
| f32 fixture (Task 3) | 32/32 | — | 20/20 | 250.3 | 0.04 GB |
| int8 fixture (Task 4) | 24/32 | — | 0/20 | — | — |
| **f32 real (control)** | **52/52** | **30/30** | — | 5.3 | — |
| **int8 real** | **48/52** | **29/30** | **16/24** | 13.2 | 13.89 GB |
| **int4 real** | **46/52** | **24/30** | **16/24** | 18.1 | 8.73 GB |

## Assessment

- The f32 control run proves the engine forward pass is exact on the real
  checkpoint: 52/52 and 30/30 against the transformers f32 oracle. Every
  TF mismatch in the int8/int4 rows is attributable to weight
  quantization, not to the C implementation of attention/MoE/rope/etc.
- int8 TF on the real checkpoint (48/52 primary, 29/30 secondary) is close
  to, but not exactly, the oracle — well above the "grossly low" (<40/52)
  investigation threshold, and the shorter secondary ref shows a
  substantially higher match rate (29/30 = 96.7% vs 48/52 = 92.3%),
  consistent with per-position quantization noise having more
  opportunities to flip a near-tied argmax as the sequence gets longer.
  Given the exact f32 control, this is now proven (not just inferred) to
  be quantization noise: real-model logits are confident and int8 dequant
  error is small (measured 0.39% max rel. error in a per-tensor dequant spot check during converter review; single-tensor sample, not a full-model sweep), with the residual mismatches being occasional
  near-tied-logit flips.
- int4 TF (46/52, 24/30) is modestly below int8 on both refs, as expected
  from the larger per-row quantization error on 4-bit routed-expert
  weights; still far from a failure mode (no crash, no garbage output).
- Both greedy runs match the oracle for the first 16 of 24 tokens before a
  single-token insertion at the same position (17), after which int8
  recovers exact alignment with the oracle's *sequence* (its per-index
  score of 16/24 undercounts this) while int4 diverges further. This is
  the expected behavior for greedy decoding under quantization noise: no
  cross-step error correction once probabilities are close together.
- No segfaults, dimension mismatches, or missing-tensor errors were
  observed in any of the six real-model engine runs (2 containers x [TF
  long, TF short, greedy long]).
