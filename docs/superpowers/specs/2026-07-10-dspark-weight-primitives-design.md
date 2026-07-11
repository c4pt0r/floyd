# DSpark Weight Primitives Design

## Scope

This milestone makes the existing safetensors indexer understand every scalar storage type present in the official `DeepSeek-V4-Flash-DSpark` checkpoint, then adds a CPU reference decoder for its routed-expert FP4 weights. It does not add V4 forward execution or optimized FP4 matrix multiplication.

The target checkpoint is revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`. Its 48 shards contain BF16, F32, I64, I8 packed FP4, F8_E4M3 dense weights, and F8_E8M0 scales. Existing Moonlight dtype numeric codes and BF16/F16/F32 conversion behavior must remain unchanged.

## Safetensors Boundary

`st.h` gains explicit dtype constants for I64, F8_E8M0, and F8_E4M3. These types are indexable and readable through `st_read_raw`, but `st_read_f32` and `st_read_slice_f32` reject them rather than silently interpreting their bytes as F16. This preserves the current loader while allowing a future V4 backend to consume raw quantized tensors.

## FP4 Reference

`deepseek_v4_quant.h` exposes allocation-free scalar and row decoding helpers. Packed E2M1 values use the official nibble table and low-nibble-first order. Each group of 32 logical FP4 values shares one E8M0 scale; E8M0 byte `e` represents `2^(e-127)`, with `255` treated as NaN.

The reference path favors clarity and exact format verification. Optimized NEON, Metal, or fused matmul kernels remain later milestones and must be tested against this implementation.

## Verification

Unit tests cover dtype recognition, all 16 FP4 codes, nibble order, E8M0 edge values, and scale boundaries at logical elements 31/32. A real-checkpoint probe must index the downloaded DSpark shards and confirm the expected dtype population without reading tensor bodies. Existing C tests and Moonlight parity remain required regression gates.
