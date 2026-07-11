# DeepSeek V4 Native Quantized Matmul Design

## Scope

Add scalar reference kernels that multiply F32 activations directly by packed DSpark weights without materializing a full F32 matrix. FP4 uses packed E2M1 with one E8M0 scale per row and 32 logical columns. FP8 uses E4M3 weights with an E8M0 scale grid over 128x128 blocks.

Activation quantization, SIMD/GPU optimization, fused SwiGLU, and production scheduling remain out of scope. These kernels establish the correctness boundary for later optimized implementations.

## Real Checkpoint Fixture

The test reads small complete blocks from the downloaded official checkpoint:

- FP8: `layers.0.attn.wq_a.weight/scale`, first 256x256 values and 2x2 scale blocks.
- FP4: `layers.0.ffn.experts.0.w1.weight/scale`, first 16 rows by 256 logical columns.

Raw row slices are loaded with a bounded safetensors byte-slice helper. Two deterministic F32 input rows exercise matmul layout. The reference fully dequantizes the extracted matrix with existing helpers, then performs ordinary F32 matmul.

## Acceptance

Native and reference outputs must have `max_abs < 3e-5`, with identical top-1 rows for both inputs. The real-checkpoint test is opt-in through `DSPARK=<model-dir>`. Existing V4 and Moonlight gates remain green, and no extracted weight artifact is written or committed.
