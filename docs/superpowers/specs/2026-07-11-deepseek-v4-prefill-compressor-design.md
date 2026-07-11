# DeepSeek V4 Prefill Compressor Design

## Scope

Add an allocation-free F32 reference compressor for stateless, single-batch prefill. It covers the official DSpark ratios: non-overlapping HCA at `128` tokens and overlapping CSA at `4` tokens. KV cache, decode state, quantization simulation, and the Lightning Indexer remain out of scope.

## Oracle Contract

`tools/make_deepseek_v4_oracle.py` calls the native Transformers compressor modules with deterministic hidden states. For each mode, `oracle.safetensors` stores projected F32 KV and gate tensors, position bias, RMSNorm weight, and the final compressed tensor after pooling, RMSNorm, and compressed-position RoPE. HCA uses two 128-token windows; CSA uses three 4-token windows so the previous-Ca/current-Cb overlap is observable.

## C Interface

`deepseek_v4_compress.h` exposes one stateless forward function parameterized by ratio, head/rope dimensions, epsilon, RoPE theta, and overlap mode. Inputs are row-major projected KV/gate tensors plus position bias and norm weight. HCA expects width `head_dim`; CSA expects width `2 * head_dim`.

The function rejects null pointers, invalid dimensions, unsupported overlap values, and token counts not divisible by the ratio. It performs per-dimension stable softmax pooling in F32, weighted RMSNorm, then interleaved RoPE on the trailing rope channels at positions `window_index * ratio`.

## Acceptance

Both fixture cases must match Transformers with `max_abs < 3e-5`. Existing V4 and Moonlight parity gates must remain green. Generated fixtures, binaries, model files, logs, and `AGENTS.md` are never committed.
