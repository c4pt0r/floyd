# DeepSeek V4 Three-Stage DSpark Milestone

## Scope

The official checkpoint contains three DSpark stages (`mtp.0` through `mtp.2`),
while the current Metal integration consumes a one-stage MTP GGUF. The next
milestone is a standalone DSpark support GGUF and a reusable proposal API. It
must not change base-model loading or the verified non-speculative path.

## Weight Contract

The converter reads the official safetensors index and emits all three stage
blocks using DS4-compatible tensor names. It also emits `mtp.0.main_proj`,
`mtp.0.main_norm`, the final stage HC/norm weights, both Markov matrices, and
the confidence projection. Routed experts retain the existing Q4 path; dense
FP8 tensors use the existing DS4 bridge. Model files remain external artifacts.

## Verification

1. A metadata-only test audits the real checkpoint and rejects the current
   one-stage MTP GGUF as incomplete.
2. A tiny real oracle compares all three stage outputs, proposal IDs, and
   confidence values before Metal integration.
3. An acceptance test compares five-token proposals with the base greedy token
   stream and reports accepted tokens per round.
4. Metal integration proceeds only if measured acceptance can amortize three
   extra blocks; final acceptance requires exact tokens and both 32/128 token
   runs above 40 emitted tok/s.

## Boundaries

This milestone does not alter sampling semantics, base KV state, or Moonlight.
The one-stage MTP path remains available until the three-stage path passes the
official correctness and throughput gates.
