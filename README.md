# floyd

`floyd` is a native Apple Metal inference runtime for two MoE model families:

- Moonlight-16B-A3B-Instruct, using compressed MLA KV cache and q8/q4 weights.
- DeepSeek-V4-Flash-DSpark, using the resident DS4 runtime and speculative decode.

Inference is in-process. Python is used only to download, convert, and generate
developer oracles. The production binary does not start Python or another model
server.

## Requirements

- Apple Silicon Mac with enough unified memory for the selected model.
- Xcode Command Line Tools, Homebrew, `libomp`, `cmake`, `ninja`, and `xxd`.
- Python 3 for model preparation.

```bash
brew install libomp cmake ninja xxd
python3 -m venv .venv
.venv/bin/pip install -U torch transformers safetensors numpy tiktoken \
  blobfile "huggingface_hub[cli]"
make
```

Metal is mandatory and enabled by default. `make METAL=1` is equivalent to
`make`; `make METAL=0` is rejected.

## Moonlight Quickstart

Download and convert the official checkpoint. The converted directory keeps
the tokenizer and config beside the packed safetensors files.

```bash
.venv/bin/hf download moonshotai/Moonlight-16B-A3B-Instruct \
  --local-dir models/Moonlight-16B-A3B-Instruct

.venv/bin/python tools/convert_moonlight.py \
  --indir models/Moonlight-16B-A3B-Instruct \
  --outdir models/moonlight_i8 --ebits 8 --dbits 8
```

Start the built-in multi-turn chat:

```bash
./floyd chat --model models/moonlight_i8 --ctx 4096 --ngen 512
```

Inside chat, use `:reset` to clear the in-memory conversation and `:exit` to
quit. Moonlight prefill, incremental prompt append, compressed KV attention,
MoE, and decode all run through the persistent Metal session. Startup prints
`MOONLIGHT_BACKEND backend=metal-moonlight`; per-turn diagnostics report prompt
and decode throughput plus CPU fallback count.

For one-shot generation:

```bash
./floyd run --model models/moonlight_i8 \
  --prompt "Summarize why prefix caching helps agents." \
  --ctx 4096 --ngen 256 --temp 0.7 --top-p 0.9
```

Use a smaller `--ctx` when memory is tight. Context allocation includes the
compressed KV cache and Metal scratch buffers. q4 conversion is supported with
`--ebits 4 --dbits 4`, but validate output quality for your workload.

## DeepSeek V4 Quickstart

Download the official checkpoint, then prepare the ignored DS4 GGUF files:

```bash
.venv/bin/hf download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --local-dir models/DeepSeek-V4-Flash-DSpark

make prepare-deepseek-v4-ds4 \
  DSPARK="$PWD/models/DeepSeek-V4-Flash-DSpark"
```

Standard prepared filenames are discovered automatically:

```bash
./floyd chat --model models/DeepSeek-V4-Flash-DSpark \
  --ctx 4096 --ngen 512 --draft 3

./floyd run --model models/DeepSeek-V4-Flash-DSpark \
  --prompt "Write a concise release note." \
  --ctx 4096 --ngen 256 --draft 3
```

Use `--ds4-model FILE` and `--ds4-support FILE` only when the prepared files
are outside the standard directory. `--draft` applies to DeepSeek V4; Moonlight
has no speculative head and rejects a nonzero value.

## Agent Stdio Server

The persistent JSONL server currently uses the DeepSeek V4 DS4 runtime:

```bash
./floyd serve --stdio \
  --model models/DeepSeek-V4-Flash-DSpark \
  --ctx 8192 --ngen 512 --draft 3 --prefix-cache-mb 256
```

It reads one JSON request per line and writes one JSON response per line.
Diagnostics use stderr, leaving stdout valid JSONL.

```json
{"id":"a","system":"You are a coding agent.","prompt":"Inspect the build.","max_tokens":128,"draft":2}
{"id":"b","messages":[{"role":"user","content":"Summarize the tests."}],"max_tokens":128}
```

The prefix LRU is byte-budgeted. Reduce `--prefix-cache-mb` and `--ctx` on
memory-constrained systems; `--prefix-cache-mb 0` disables snapshots without
disabling inference.

## CLI

Run `./floyd help` for the current flags. Common forms are:

```bash
./floyd chat --model DIR [--ctx N] [--ngen N]
./floyd run --model DIR --prompt TEXT [--ctx N] [--ngen N]
./floyd serve --stdio --model DIR [--prefix-cache-mb N]
```

Model checkpoints, converted weights, generated fixtures, logs, and binaries
are ignored and must not be committed.

## Validation

Run the portable tests and Metal kernel checks:

```bash
make test-c
make metal-test
```

Moonlight numerical closure uses external ignored fixtures:

```bash
make test-moonlight-metal \
  MOONLIGHT_TINY=/path/to/moonlight_model \
  MOONLIGHT_ORACLE=/path/to/oracle
```

DeepSeek V4 hardware smoke and performance checks are opt-in:

```bash
make test-deepseek-v4-ds4-official \
  DSPARK="$PWD/models/DeepSeek-V4-Flash-DSpark" NGEN=128 MIN_TPS=30
```

See `docs/` for numerical contracts, implementation plans, and parity reports.
