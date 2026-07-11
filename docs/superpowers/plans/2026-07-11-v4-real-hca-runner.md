# V4 Real HCA Runner Implementation Plan

**Goal:** Carry the official DSpark reference runner through layer 3 and establish
the state contract used by layers 3-42.

1. Extend `tools/make_v4_real_layer0_oracle.py` and
   `tests/test_v4_real_layer0.py` with a real 128-token layer-3 capture. Run the
   oracle test and confirm it fails before the new tensors exist; commit the test.
2. Capture HCA projections/compressed KV, learned-router values, and layer-3
   output. Regenerate only ignored fixtures, run the Python test, and commit the
   oracle implementation.
3. Extend `tests/test_v4_real_layer0.c` with the caller-owned layer state API and
   official layer-3 comparisons. Confirm compilation or runtime RED and commit.
4. Implement the ratio-driven compressor path, causal compressed-key visibility,
   and biased learned routing in the real runner. Run the real checkpoint test and
   related V4 regressions; commit GREEN.
5. Reuse the state API for representative CSA/HCA layers and then layers 3-42,
   preserving RED/GREEN commits for each boundary before adding final head and
   DSpark stages.

