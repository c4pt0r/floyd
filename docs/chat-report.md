# Chat E2E report: real Moonlight-16B-A3B-Instruct (int8)

Machine: Apple M3 Ultra, 32 cores, ~346 GB RAM free at test time, macOS
15.5. Model: `models/moonlight_i8` (int8 container, converted from
`moonshotai/Moonlight-16B-A3B-Instruct`). Binary: `floyd` built from
`floyd.c` at the state right before this task's commit (clean
`make clean && make`, zero warnings — see regression section below).

## 1. Tokenizer parity gates (pre-condition for any of this to be
   meaningful — chat correctness depends on lossless C tokenization)

```
$ make test-tok
./tests/test_tok_moon models/Moonlight-16B-A3B-Instruct tok_cases.json
tokenizer parity: 52/52
template build: ok (42/42 ids)
```

- **52/52** hand-curated tokenizer cases (raw-byte BPE + moonshot
  pre-tokenizer) match the `tiktoken`/`transformers` Python oracle
  exactly (`tok_moon.h`, gate introduced in Chat-task 3).
- **42/42** chat-template token ids (system/user/assistant role markers,
  generation prompt, multi-turn framing) match the oracle's
  `apply_chat_template` output exactly.
- Per the project ledger (`.superpowers/sdd/progress.md`, Chat-task 3
  entry), an independent reviewer additionally ran **50 adversarial
  cases** (edge-case Unicode: combining marks, mixed CJK/Latin,
  emoji/ZWJ sequences, degenerate byte strings) against the same
  oracle — all 50 matched. That sweep is not part of the committed
  `tok_cases.json` fixture; it is recorded here as reviewer-verified
  evidence supplementing the 52/52 gate.

## 2. E2E multi-turn chat smoke (real i8 weights)

Command (exact, as specified in the task brief):

```
printf 'What is the capital of France? Answer in one word.\nWhat did I just ask you?\n:exit\n' \
  | ./floyd chat --model models/moonlight_i8 2>chat_smoke.err | tee chat_smoke.out
```

### stdout (verbatim)

```
== Motore C GLM (glm_moe_dsa), cache=64 expert/layer | expert@8-bit densa@8-bit | idot: neon ==
caricato in 0.70s | densa residente: 3411.24 MB | layers=27 experts=64 | MTP assente (draft=0)

› ◆ Paris
› ◆ You asked for the capital of France, and I provided the answer, which is Paris.
›
```

- Turn 1 answer: **"Paris"** — correct, single word as instructed.
- Turn 2 answer: **"You asked for the capital of France, and I provided
  the answer, which is Paris."** — this is the memory-check evidence:
  the model correctly recalls and paraphrases the *first* question
  ("capital of France") when asked "What did I just ask you?", proving
  the KV-cache/history carries real conversational context across
  turns, not just a stateless single-shot completion.
- No literal `<|im_end|>` (or any other special-token text) leaked into
  the output — verified with `grep -c im_end chat_smoke.out` → 0
  matches.

### stderr (verbatim, load/tokenizer banner + per-turn stats)

```
[MTP] assente (draft=0)
[USAGE] storia expert: 24180 selezioni (models/moonlight_i8/.coli_usage)
[PIN] hot-store: 1354 expert in RAM (11.7 GB) in 1s da models/moonlight_i8/.coli_usage
[PIN] mlock: 11.8 GB inchiodati in RAM fisica / wired in physical RAM (niente compressione/no compression) in 1s
[RAM_GB=345.9 auto] cap=64 ok (proiezione picco 34.3 GB)
[stop] 1 token di stop: 163586
floyd chat — :reset azzera, :exit esce
[prefill] layer 1/27 · 28 token
...
[prefill] layer 27/27 · 28 token

[1 tok, 6.74 tok/s | ctx 29/4096 | RSS 14.36 GB]
[prefill] layer 1/27 · 14 token
...
[prefill] layer 27/27 · 14 token

[18 tok, 7.39 tok/s | ctx 61/4096 | RSS 15.04 GB]
```

- Turn 1: 28-token prompt prefill (system + user + genprompt), 1
  generated token (EOS right after "Paris"), **6.74 tok/s**, ctx 29/4096.
- Turn 2: only **14 tokens** re-prefilled (the new user turn +
  genprompt), *not* the full 29-token history — confirming the KV
  cache carries the first turn forward instead of re-tokenizing/
  re-running the whole conversation. 18 tokens generated,
  **7.39 tok/s**, ctx grows to 61/4096.

## 3. KV persistence (`.coli_kv`) round trip

Brief's three-part check: (a) default kvsave writes+persists across a
process restart ("warm" resume, no re-prefill), (b) `--no-kvsave`
writes nothing.

**(a) Default kvsave — first run** (the smoke test above) ended with
`ctx 61/4096` and left `models/moonlight_i8/.coli_kv` on disk
(3,794,972 bytes after turn 2).

**(a) Default kvsave — process restart, same model dir:**

```
$ printf ':exit\n' | ./floyd chat --model models/moonlight_i8 2>chat_warm.err
```

stderr (verbatim):

```
[MTP] assente (draft=0)
[USAGE] storia expert: 33696 selezioni (models/moonlight_i8/.coli_usage)
[PIN] hot-store: 1437 expert in RAM (12.5 GB) in 1s da models/moonlight_i8/.coli_usage
[PIN] mlock: 12.5 GB inchiodati in RAM fisica / wired in physical RAM (niente compressione/no compression) in 1s
[RAM_GB=345.9 auto] cap=64 ok (proiezione picco 35.0 GB)
[stop] 1 token di stop: 163586
[KV] conversazione ripresa dal disco: 61 token in 0.0s (niente re-prefill)
floyd chat — :exit esce
```

`[KV] conversazione ripresa dal disco: 61 token in 0.0s (niente
re-prefill)` — the resumed token count (61) matches **exactly** the
`ctx 61/4096` the previous process ended with, and no `[prefill]`
lines are emitted before the prompt (no re-tokenization/re-forward of
the history) — this is the warm-restore evidence.

**(b) `--no-kvsave`** (after deleting `.coli_kv` and starting a fresh
one-turn conversation):

```
$ rm -f models/moonlight_i8/.coli_kv
$ printf 'Hi\n:exit\n' | ./floyd chat --model models/moonlight_i8 --no-kvsave 2>chat_nokv.err
◆ Hello! How can I assist you today?
$ ls models/moonlight_i8/.coli_kv
ls: models/moonlight_i8/.coli_kv: No such file or directory
```

Confirmed: with `--no-kvsave`, no `.coli_kv` file is written at all
(no `[KV]` lines on load either, since there's nothing to load).

`models/moonlight_i8/.coli_kv` was deleted after these checks so the
repo/model directory stays clean (it is gitignored via `models/` in
`.gitignore` regardless, but the brief asked for explicit cleanup).

## 4. `run` subcommand — one-shot English/Chinese

```
$ ./floyd run --model models/moonlight_i8 --prompt "What is the capital of France? Answer in one word." --ngen 16
prompt: 28 token | genero fino a 16 (stop EOS=163586) | draft n-gram=0
What is the capital of France? Answer in one word.Paris
---
1 token in 1.08s (0.93 tok/s) | hit-rate expert 100.0% | RSS 15.04 GB
```

```
$ ./floyd run --model models/moonlight_i8 --prompt "法国的首都是哪里？用一个词回答。" --ngen 16
prompt: 25 token | genero fino a 16 (stop EOS=163586) | draft n-gram=0
法国的首都是哪里？用一个词回答。巴黎
---
1 token in 1.01s (0.99 tok/s) | hit-rate expert 99.7% | RSS 15.06 GB
```

Both correctly answer with a single token, stopping on EOS right
after "Paris" / "巴黎" respectively — English and (separately
tokenized/detokenized, non-Latin) Chinese both round-trip cleanly
through the moonshot tokenizer + chat template + generation stack.

## 5. Full regression (Step 3 of the brief)

```
$ make clean && make
rm -f floyd *.o kernels_metal.h tests/test_json tests/test_st tests/test_backend_metal tests/test_tok_moon
clang -O3 -Xclang -fopenmp ... floyd.c -o floyd -lm ... -lomp
(zero warnings)

$ make test-c
json tests: ok
safetensors primitive tests: ok

$ make test-tok
tokenizer parity: 52/52
template build: ok (42/42 ids)

$ ./floyd tf --model fixture_tiny --ref ref_tiny.json --cap 8 --ebits 16 --dbits 16
PREFILL (teacher-forcing) C vs oracolo: 32/32 posizioni | 216.8 pos/s
```

All green, zero compiler warnings across the whole rebuild.

As a sanity check that the legacy env-var interface (documented in
`docs/parity-report.md`) still works unchanged, one line from that
report was re-run verbatim:

```
$ SNAP=fixture_tiny_i8 TF=1 REF=ref_tiny.json ./floyd 8 8 8
PREFILL (teacher-forcing) C vs oracolo: 24/32 posizioni | 218.5 pos/s
```

Matches the report's recorded `24/32` exactly — no `docs/parity-report.md`
changes were needed.

## Summary

| Check | Result |
|---|---|
| Tokenizer parity gate | 52/52 |
| Chat template gate | 42/42 |
| Reviewer adversarial sweep (ledger) | 50/50 |
| Multi-turn memory (2-turn chat, real i8) | Paris; second answer correctly references first question |
| `<|im_end|>` leak check | none found |
| KV warm restore after process restart | 61/61 token resumed, 0.0s, no re-prefill |
| `--no-kvsave` | confirmed no `.coli_kv` written |
| `run` English one-shot | "Paris" |
| `run` Chinese one-shot | "巴黎" |
| Full regression (`make clean && make && make test-c && make test-tok`) | all green, zero warnings |
| Fixture TF gate (`./floyd tf --model fixture_tiny ...`) | 32/32 |
| Legacy env-form command (parity-report.md) | reproduced 24/32 exactly |
