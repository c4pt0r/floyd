# User-First README And English CLI Design

## Goal

Make the repository immediately usable from the README and present a consistent
English command-line interface. DeepSeek V4 is the primary quickstart because it
is the production chat/agent path; Moonlight remains documented as a supported
secondary runtime.

## README Structure

The first sections after the project summary will be:

1. Prerequisites and platform expectations.
2. DeepSeek V4 download and DS4 preparation.
3. Metal build.
4. Built-in `floyd chat` usage, including `:reset` and `:exit`.
5. One-shot `run` and persistent `serve --stdio` examples.
6. Memory guidance for `--ctx` and `--prefix-cache-mb`.

Commands will be copy-pasteable from the repository root and use CLI flags as
the public interface. Standard prepared DS4 filenames are auto-discovered;
explicit `--ds4-model` and `--ds4-support` examples are reserved for custom
locations. The JSONL request example will be short and show that the server
must remain resident for prefix-cache reuse.

Moonlight setup, conversion, and parity validation will move below the primary
quickstart. Architecture notes, contributor-oriented oracle commands, and
measured results remain available but do not block the first successful chat.

## CLI Language

Translate public text in `usage()` and `cli_adapt()` to English, including
command descriptions, required-argument errors, unknown flags, and
command-specific validation. Translate the built-in Moonlight, DeepSeek V4,
and ggml chat banners and reset confirmations where they are user-facing.
Command names, flags, environment compatibility, exit codes, and inference
behavior remain unchanged.

## Verification

Update shell assertions only where they intentionally depend on user-facing
text. Verify `floyd help`, malformed CLI invocations, the model-free serve CLI
test, CPU build, Metal build, and existing Moonlight/DeepSeek chat dispatch
tests. Do not commit models, fixtures, binaries, logs, or `AGENTS.md`.
