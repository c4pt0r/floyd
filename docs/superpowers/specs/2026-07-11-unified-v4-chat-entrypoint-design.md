# Unified DeepSeek V4 Chat Entrypoint Design

## Goal

`floyd --model DIR` must start the correct built-in chat runtime for both
Moonlight/DeepSeek V3 and DeepSeek V4. DeepSeek V4 code must use the explicit
`deepseek_v4` name rather than the ambiguous `v4` abbreviation. The `floyd`
executable must not depend on launching or shipping a second executable.

## Architecture

Extract the existing DeepSeek V4 chat application into a linkable
`deepseek_v4_chat.c`/`.h` module exposing an options-based
`deepseek_v4_chat_run()` API. Link it directly into `floyd` in CPU and Metal
builds. The old standalone `v4_chat` executable is removed; `floyd` is the
single supported chat binary.

After CLI adaptation has resolved `--model`, `floyd` parses the checkpoint's
`config.json` through `json.h`. For default or explicit chat mode:

- `model_type == "deepseek_v4"` calls `deepseek_v4_chat_run()` before Moonlight model
  loading or Moonlight Metal initialization.
- Other model types continue through the existing Moonlight path unchanged.
- Explicit `run`, `tf`, and `gen` retain their current behavior and are not
  routed to V4 in this milestone.

## Naming Migration

Rename all DeepSeek V4-specific source, test, and oracle files from `v4_*` to
`deepseek_v4_*`. Rename C symbols and types from `v4_*`/`V4*` to
`deepseek_v4_*`/`DeepSeekV4*`, Make targets from `test-v4-*` to
`test-deepseek-v4-*`, and user-facing environment/log prefixes from `V4_*` to
`DEEPSEEK_V4_*`. Ignored fixture directories follow the same convention and
are regenerated locally; no fixture or model artifacts are committed.

No compatibility aliases are retained for internal headers, C symbols, test
targets, or the old standalone binary. The checkpoint's official
`model_type=deepseek_v4` remains the detection contract.

## CLI Contract

DeepSeek V4 receives `--ctx` and `--ngen`; absent values retain its current
defaults of 512 context tokens and 16 generated tokens per turn. A Metal build
retains the current policy of enabling GPU batch matmul by default.

Both runtimes expose the same shell: banner `floyd chat [MODEL]`, input prompt
`›`, output prompt `◆`, commands `:reset` and `:exit`, and per-turn statistics
on stderr. DeepSeek V4 may accept `/clear` and `/exit` as temporary input
aliases, but they are not advertised.

DeepSeek V4 remains greedy in this milestone. CLI-provided `--temp`, `--top-p`,
or `--system` must fail with a clear unsupported-option error instead of being
silently ignored. Moonlight continues to support those flags.

## Errors And Compatibility

Missing or malformed config files fall through to the existing model loader so
its diagnostics remain authoritative. A recognized DeepSeek V4 checkpoint
that cannot initialize reports a model-specific error and exits nonzero.

## Verification

TDD starts with an official-checkpoint CLI smoke that builds only `floyd`,
enters DeepSeek V4 chat, and exits. A one-token official smoke must match the
existing token oracle and report positive Metal calls. Naming tests reject
remaining source/build identifiers matching the old V4 prefixes. Regression
includes Moonlight default chat, tiny TF/greedy, DeepSeek V4 native quant/Metal
tests, `make test-c all`, link inspection, and clean Git status auditing.
