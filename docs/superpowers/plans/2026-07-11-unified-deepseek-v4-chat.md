# Unified DeepSeek V4 Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `floyd --model DIR` a single built-in chat entrypoint for Moonlight and DeepSeek V4 while replacing ambiguous project-owned `v4` names with `deepseek_v4`.

**Architecture:** Rename the verified DeepSeek V4 implementation without changing its numerical contracts, then extract its chat loop into a linkable module. `floyd` parses `config.json`, dispatches `model_type=deepseek_v4` before Moonlight initialization, and presents the same terminal shell for both runtimes.

**Tech Stack:** C11-style header implementation, Objective-C Metal wrapper, Make, POSIX shell tests, Python oracle builders, safetensors fixtures.

## Global Constraints

- `floyd` is the only supported chat executable; do not launch a subprocess.
- Use `deepseek_v4_*`, `DeepSeekV4*`, `DEEPSEEK_V4_*`, and `test-deepseek-v4-*` for project-owned DeepSeek V4 names.
- Preserve official `model_type=deepseek_v4` and upstream class names.
- Preserve Moonlight `run`, `tf`, `gen`, and legacy environment/positional behavior.
- Do not commit models, fixtures, binaries, logs, or `AGENTS.md`.
- Every behavior change follows RED, GREEN, regression, then a focused commit.

---

### Task 1: DeepSeek V4 Naming Migration

**Files:**
- Create: `tests/test_deepseek_v4_naming.sh`
- Rename: `v4_*.h` to `deepseek_v4_*.h`
- Rename: `tests/test_v4_*` and `tests/test_make_v4_oracle.py` to `tests/test_deepseek_v4_*` and `tests/test_make_deepseek_v4_oracle.py`
- Rename: `tools/make_v4_*` to `tools/make_deepseek_v4_*`
- Rename: DeepSeek V4 docs containing `v4` in their filename to `deepseek-v4`
- Modify: `Makefile`, `.gitignore`, `README.md`, renamed C/Python/shell files

**Interfaces:**
- Produces C prefix `deepseek_v4_*`, type prefix `DeepSeekV4`, macro/log/env prefix `DEEPSEEK_V4_`, and Make target prefix `test-deepseek-v4-`.
- Retains checkpoint field `model_type=deepseek_v4` and `DSPARK` model-path variable.

- [ ] **Step 1: Write the failing naming gate**

The shell test must fail when tracked source/build filenames match `(^|/)(v4_|test_v4_|make_v4_)`, when C files contain `\bv4_` or `\bV4_`, or when `Makefile` contains `test-v4-`.

- [ ] **Step 2: Run the gate and verify RED**

Run: `sh tests/test_deepseek_v4_naming.sh`

Expected: failure listing current `v4_quant.h`, `V4Runtime`, and `test-v4-*` names.

- [ ] **Step 3: Perform scoped mechanical renames**

Use `git mv` for tracked files. Apply token-aware replacements only to project-owned names:

```text
v4_ -> deepseek_v4_
V4_ -> DEEPSEEK_V4_
V4Runtime/V4Real/V4Quant -> DeepSeekV4Runtime/DeepSeekV4Real/DeepSeekV4Quant
test-v4- -> test-deepseek-v4-
fixture_tiny_v4 -> fixture_tiny_deepseek_v4
```

Do not rewrite upstream `DeepseekV4ForCausalLM` or the JSON value `deepseek_v4`.

- [ ] **Step 4: Verify GREEN and numerical parity**

Run:

```bash
sh tests/test_deepseek_v4_naming.sh
make test-deepseek-v4-oracle PYTHON=.venv/bin/python
make test-deepseek-v4-native-quant test-deepseek-v4-native-quant-metal DSPARK=models/DeepSeek-V4-Flash-DSpark
```

Expected: naming gate passes; native FP4/FP8 retain prior error thresholds and top-1 matches.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: use explicit DeepSeek V4 naming"
```

### Task 2: Built-in DeepSeek V4 Runtime Dispatch

**Files:**
- Create: `deepseek_v4_chat.h`
- Modify: `deepseek_v4_chat.c`, `floyd.c`, `Makefile`, `json.h`
- Create: `tests/test_deepseek_v4_chat_dispatch.sh`

**Interfaces:**
- Produces `DeepSeekV4ChatOptions { const char *model_dir; int max_context; int max_new_tokens; int use_spec; }`.
- Produces `int deepseek_v4_chat_run(const DeepSeekV4ChatOptions *options)`.
- Produces `int deepseek_v4_model_dir(const char *model_dir)`, using structured `json.h` parsing.

- [ ] **Step 1: Write the failing built-in dispatch test**

Build only `floyd`, assert no standalone DeepSeek V4 chat binary exists, pipe `:exit`, and require `floyd chat [DeepSeek V4]` plus the backend line:

```bash
printf ':exit\n' | ./floyd --model "$DSPARK" --ctx 64 --ngen 1
```

- [ ] **Step 2: Run and verify RED**

Run: `make test-deepseek-v4-chat-dispatch DSPARK=models/DeepSeek-V4-Flash-DSpark`

Expected: Moonlight loader rejects the DeepSeek V4 checkpoint or the built-in symbol is absent.

- [ ] **Step 3: Extract the application API**

Move the old standalone `main()` behavior behind `deepseek_v4_chat_run()`. Keep backend initialization, tokenizer loading, runtime state, greedy/spec generation, and cleanup inside this module. Remove the standalone binary target.

- [ ] **Step 4: Add structured model detection and early dispatch**

After `cli_adapt()` resolves `SNAP`, parse `<model>/config.json`. When chat mode and `model_type` is `deepseek_v4`, construct options from `CTX`, `NGEN`, and `DSPARK_SPEC`, call `deepseek_v4_chat_run()`, and return before Moonlight Metal/model initialization.

- [ ] **Step 5: Verify GREEN**

Run the dispatch test in CPU and `METAL=1` builds. `nm floyd` must contain `deepseek_v4_chat_run`; `otool -L floyd` in a Metal build must list Metal and Foundation.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: embed DeepSeek V4 chat in floyd"
```

### Task 3: Unified Chat Shell And Option Contract

**Files:**
- Modify: `deepseek_v4_chat.c`, `deepseek_v4_chat.h`, `floyd.c`, `README.md`
- Modify: `tests/test_cli_default_chat.sh`, `tests/test_deepseek_v4_chat_dispatch.sh`

**Interfaces:**
- Both runtimes display `floyd chat [MODEL]`, use input prompt `›`, output prompt `◆`, and advertise `:reset`/`:exit`.
- DeepSeek V4 accepts `/clear` and `/exit` only as unadvertised compatibility aliases.
- CLI markers distinguish explicitly supplied `--temp`, `--top-p`, and `--system`; DeepSeek V4 rejects them with exit code 2.

- [ ] **Step 1: Extend tests and verify RED**

Require the common banner/prompts and `:reset/:exit` for both official model paths. Invoke DeepSeek V4 with each unsupported sampling/system flag and require a model-specific error instead of inference.

- [ ] **Step 2: Implement the shared shell contract**

Update both loops to identical command strings and prompts. Keep model-specific tokenization, state reset, generation, and statistics behind their runtime boundaries.

- [ ] **Step 3: Implement explicit option rejection**

Have `cli_adapt()` record which optional flags were supplied. Before DeepSeek V4 initialization, reject the three unsupported flags; do not reject inherited debug environment variables unless the CLI flag was present.

- [ ] **Step 4: Verify GREEN and compatibility**

Run default and explicit chat smoke for both models, `./floyd --help`, Moonlight tiny `tf`/`gen`, and the legacy `SNAP=... ./floyd 8 16 16` command.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: unify chat UX across model runtimes"
```

### Task 4: Full Regression, Audit, And Push

**Files:**
- Modify only files required by failures found during verification.

- [ ] **Step 1: Run build and C/oracle suites**

```bash
make clean
make test-c all test-tok
make test-deepseek-v4-oracle PYTHON=.venv/bin/python
make test-deepseek-v4-attention test-deepseek-v4-compress test-deepseek-v4-indexer test-deepseek-v4-kv-cache test-deepseek-v4-moe test-deepseek-v4-hc
```

- [ ] **Step 2: Run Metal and real-model suites**

```bash
make METAL=1 metal-test test-deepseek-v4-native-quant-metal DSPARK=models/DeepSeek-V4-Flash-DSpark
make METAL=1 test-deepseek-v4-chat-dispatch DSPARK=models/DeepSeek-V4-Flash-DSpark
./floyd tf --model models/moonlight_i8 --ref ref_moonlight.json --metal
```

- [ ] **Step 3: Run official one-token DeepSeek V4 smoke through `floyd`**

Require the existing oracle token, positive Metal calls, and no standalone helper process or binary.

- [ ] **Step 4: Audit repository state**

Run `git diff --check`, `git status --short --untracked-files=all`, naming gate, tracked file-size inspection, and `git log`. Confirm no models, fixtures, binaries, logs, or `AGENTS.md` are staged.

- [ ] **Step 5: Commit fixes, push `master`, and verify remote HEAD**

Expected: local `master` and `origin/master` resolve to the same commit.
