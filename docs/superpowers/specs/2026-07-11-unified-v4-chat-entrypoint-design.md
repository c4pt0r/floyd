# Unified V4 Chat Entrypoint Design

## Goal

`floyd --model DIR` must start the correct built-in chat runtime for both
Moonlight/DeepSeek V3 and DeepSeek V4. The `floyd` executable must not depend
on launching or shipping a separate `v4_chat` executable.

## Architecture

Extract the existing V4 chat application from `v4_chat.c` into a linkable
`v4_chat_app.c`/`.h` module exposing a small options-based `v4_chat_run()` API.
Keep `v4_chat.c` as an optional compatibility wrapper with only `main()`.
Link the application module directly into `floyd` in CPU and Metal builds.

After CLI adaptation has resolved `--model`, `floyd` parses the checkpoint's
`config.json` through `json.h`. For default or explicit chat mode:

- `model_type == "deepseek_v4"` calls `v4_chat_run()` before Moonlight model
  loading or Moonlight Metal initialization.
- Other model types continue through the existing Moonlight path unchanged.
- Explicit `run`, `tf`, and `gen` retain their current behavior and are not
  routed to V4 in this milestone.

## CLI Contract

V4 receives `--ctx` and `--ngen`; absent values retain its current defaults of
512 context tokens and 16 generated tokens per turn. A Metal build retains the
current V4 policy of enabling GPU batch matmul by default. The interactive V4
loop accepts both `:reset`/`:exit` and its existing `/clear`/`/exit` commands.

V4 remains greedy in this milestone. CLI-provided `--temp`, `--top-p`, or
`--system` must fail with a clear unsupported-option error instead of being
silently ignored. Moonlight continues to support those flags.

## Errors And Compatibility

Missing or malformed config files fall through to the existing model loader so
its diagnostics remain authoritative. A recognized V4 checkpoint that cannot
initialize reports a V4-specific error and exits nonzero. The standalone
`v4_chat` command remains available but shares the exact same implementation.

## Verification

TDD starts with an official-checkpoint CLI smoke that runs `floyd` without a
`v4_chat` binary, enters V4 chat, and exits. A one-token official smoke must
match the existing V4 token oracle and report positive Metal calls. Regression
includes Moonlight default chat, tiny TF/greedy, V4 native quant/Metal tests,
`make test-c all`, link inspection, and clean Git status auditing.
