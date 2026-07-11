# DeepSeek V4 Prefill Compressor Implementation Plan

**Goal:** Match Transformers HCA-128 and overlapping CSA-4 prefill compression in a standalone F32 C reference.

**Architecture:** Extend the existing tiny oracle with direct native-compressor captures, then implement the matching stateless C kernel behind a narrow validated interface.

## Task 1: Oracle Captures

- Extend `build_tiny()` to use HCA ratio 128.
- Add deterministic direct compressor calls and save `kv`, `gate`, `ape`, norm, and final output tensors for HCA and CSA.
- Update `tests/test_make_deepseek_v4_oracle.py`; run `make test-deepseek-v4-oracle` first to observe RED, then GREEN.
- Commit the oracle change without generated artifacts.

## Task 2: C Fixture and Kernel

- Add `tests/test_deepseek_v4_compress_fixture.c` and a `test-deepseek-v4-compress` Make target.
- Compile first and confirm RED because `deepseek_v4_compress.h` is missing.
- Implement validation, HCA/CSA window layout, stable per-dimension softmax pooling, weighted RMSNorm, and compressed-position interleaved RoPE in `deepseek_v4_compress.h`.
- Verify both modes are below `3e-5`; commit source and tests.

## Task 3: Regression

- Run oracle, compressor, existing V4 MoE/mHC/attention, all C unit tests, and Moonlight teacher-forcing/greedy parity.
- Run `git diff --check` and confirm only source, tests, Makefile, and concise docs are tracked.
